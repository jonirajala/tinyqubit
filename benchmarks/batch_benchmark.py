"""Batched evaluation benchmark — correctness + performance tracking.

Benchmarks batched expectation, parameter sweeps, kernel matrices, and
batched gradient computation across workloads from real QML/VQE examples.

Usage:
    ./venv/bin/python benchmarks/batch_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/batch_benchmark.py --save   # save new baseline
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, Parameter, simulate
from tinyqubit.measurement.observable import Observable, X, Y, Z, expectation, expectation_batch, expectation_sweep

BASELINE_PATH = Path(__file__).parent / "batch_baseline.json"
N_WARMUP = 1
N_RUNS = 5
N_RUNS_FAST = 15
CORRECTNESS_TOL = 1e-8


# --- Helpers ---

def _build_hea(n, layers, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for l in range(layers):
        for q in range(n):
            ny, nz = f"ry_{l}_{q}", f"rz_{l}_{q}"
            c.ry(q, Parameter(ny)); c.rz(q, Parameter(nz))
            params[ny] = rng.uniform(0, 2 * np.pi)
            params[nz] = rng.uniform(0, 2 * np.pi)
        for q in range(n - 1): c.cx(q, q + 1)
    c.param_values = params
    return c


def _build_zz_feature_map(n, reps=2):
    """ZZ feature map — standard kernel circuit for QML."""
    c = Circuit(n)
    for _ in range(reps):
        for q in range(n): c.h(q)
        for q in range(n):
            c.rz(q, Parameter(f"x_{_}_{q}"))
        for i in range(n):
            for j in range(i + 1, n):
                c.cz(i, j)
                c.rz(j, Parameter(f"xx_{_}_{i}_{j}"))
    return c


def _z_hamiltonian(n):
    return sum(Z(q) for q in range(n))


def _heisenberg_hamiltonian(n):
    terms = []
    for q in range(n - 1):
        for p in ['X', 'Y', 'Z']:
            terms.append((1.0, {q: p, q + 1: p}))
    return Observable(terms)


def _random_param_sets(circuit, n_sets, seed=42):
    rng = np.random.RandomState(seed)
    keys = sorted(circuit.param_values.keys())
    return [{k: rng.uniform(0, 2 * np.pi) for k in keys} for _ in range(n_sets)]


# --- Benchmark implementations ---

def bench_expectation_sweep(circuit, obs, param_name, n_points, base_params):
    """Sweep one parameter, measure expectation at each point."""
    values = np.linspace(0, 2 * np.pi, n_points)
    work = circuit.bind({})
    results = np.empty(n_points)
    for i, v in enumerate(values):
        work.bind_params({**base_params, param_name: v})
        results[i] = expectation(work, obs)
    return results


def bench_multi_param_eval(circuit, obs, param_sets):
    """Evaluate expectation for each parameter set — the VQE inner loop."""
    work = circuit.bind({})
    results = np.empty(len(param_sets))
    for i, params in enumerate(param_sets):
        work.bind_params(params)
        results[i] = expectation(work, obs)
    return results


def bench_kernel_matrix(feature_circuit, X_data, n_qubits):
    """Compute fidelity kernel matrix K[i,j] = |<phi(xi)|phi(xj)>|^2."""
    n = len(X_data)
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Bind both data points: encode x_i, then adjoint-encode x_j
            # For fidelity: simulate encoding circuit for x_i, compute overlap with x_j
            p_i = _zz_params(X_data[i], n_qubits)
            state_i, _ = simulate(feature_circuit.bind(p_i))
            p_j = _zz_params(X_data[j], n_qubits)
            state_j, _ = simulate(feature_circuit.bind(p_j))
            K[i, j] = K[j, i] = float(np.abs(np.vdot(state_i, state_j)) ** 2)
    return K


def _zz_params(x, n, reps=2):
    params = {}
    for r in range(reps):
        for q in range(n):
            params[f"x_{r}_{q}"] = float(x[q])
        for i in range(n):
            for j in range(i + 1, n):
                params[f"xx_{r}_{i}_{j}"] = float(x[i] * x[j])
    return params


def bench_batch_gradient_eval(circuit, obs, param_sets):
    """Evaluate adjoint gradient for multiple parameter sets — hyperparameter search."""
    from tinyqubit.qml.optim import adjoint_gradient
    results = []
    for params in param_sets:
        g = adjoint_gradient(circuit, obs, params)
        results.append(sum(v ** 2 for v in g.values()))  # gradient norm squared
    return np.array(results)


# --- Benchmark suite ---

BENCHMARKS = [
    # (name, runner_fn, setup_fn) — setup returns (args, expected_shape)

    # Expectation sweep: single param scan (like landscape plots)
    ("sweep_8q_100pts", lambda: _run_sweep(8, 3, 100)),
    ("sweep_14q_50pts", lambda: _run_sweep(14, 2, 50)),

    # Multi-param evaluation: VQE-style batch
    ("eval_8q_50sets",  lambda: _run_eval(8, 3, 50, _z_hamiltonian)),
    ("eval_12q_30sets", lambda: _run_eval(12, 2, 30, _z_hamiltonian)),
    ("eval_8q_heisen",  lambda: _run_eval(8, 3, 30, _heisenberg_hamiltonian)),

    # Kernel matrix: fidelity kernel (symmetric, upper-triangle opportunity)
    ("kernel_6q_20x20", lambda: _run_kernel(6, 20)),
    ("kernel_8q_15x15", lambda: _run_kernel(8, 15)),

    # Batch gradient: evaluate gradient at multiple parameter points
    ("grad_batch_8q_20", lambda: _run_grad_batch(8, 3, 20)),
    ("grad_batch_12q_10", lambda: _run_grad_batch(12, 2, 10)),
]


def _run_sweep(n, layers, n_points):
    circuit = _build_hea(n, layers)
    obs = _z_hamiltonian(n)
    params = circuit.param_values
    first_param = sorted(params.keys())[0]
    result = bench_expectation_sweep(circuit, obs, first_param, n_points, params)
    return result, n, len(params)


def _run_eval(n, layers, n_sets, obs_fn):
    circuit = _build_hea(n, layers)
    obs = obs_fn(n)
    param_sets = _random_param_sets(circuit, n_sets)
    result = bench_multi_param_eval(circuit, obs, param_sets)
    return result, n, len(circuit.param_values)


def _run_kernel(n, n_samples):
    rng = np.random.RandomState(42)
    X_data = rng.uniform(0, np.pi, (n_samples, n))
    feature_circuit = _build_zz_feature_map(n)
    K = bench_kernel_matrix(feature_circuit, X_data, n)
    return K, n, n_samples


def _run_grad_batch(n, layers, n_sets):
    circuit = _build_hea(n, layers)
    obs = _z_hamiltonian(n)
    param_sets = _random_param_sets(circuit, n_sets)
    result = bench_batch_gradient_eval(circuit, obs, param_sets)
    return result, n, len(circuit.param_values)


def run_benchmark(name, runner):
    """Run a single benchmark: correctness check + timing."""
    # Correctness: run once, get result + metadata
    result, n_qubits, n_params_or_samples = runner()
    checksum = float(np.sum(np.abs(result)))
    result_shape = result.shape

    # Timing
    n_runs = N_RUNS_FAST if n_qubits <= 8 else N_RUNS
    for _ in range(N_WARMUP):
        runner()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner()
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {
        "checksum": round(checksum, 8), "shape": list(result_shape),
        "n_qubits": n_qubits, "n_params_or_samples": n_params_or_samples,
        "time_ms": round(time_ms, 3),
    }


def main():
    save_mode = "--save" in sys.argv

    print("=== TinyQubit Batch Evaluation Benchmark ===\n")

    baseline = {}
    if BASELINE_PATH.exists() and not save_mode:
        baseline = json.loads(BASELINE_PATH.read_text())
        print(f"Baseline loaded: {BASELINE_PATH.name} ({len(baseline)} benchmarks)\n")
    elif not save_mode:
        print("No baseline found. Run with --save to create one.\n")

    results = {}
    correctness_pass = correctness_fail = perf_faster = perf_regression = 0

    print("Correctness:")
    for name, runner in BENCHMARKS:
        r = run_benchmark(name, runner)
        results[name] = r

        if name in baseline:
            b = baseline[name]
            chk_ok = abs(r["checksum"] - b["checksum"]) < CORRECTNESS_TOL * max(abs(b["checksum"]), 1)
            shape_ok = r["shape"] == b["shape"]

            if chk_ok and shape_ok:
                print(f"  {name:<22s} PASS  (chk={r['checksum']:.4f} shape={r['shape']})")
                correctness_pass += 1
            else:
                fails = []
                if not chk_ok: fails.append(f"chk {b['checksum']:.4f}→{r['checksum']:.4f}")
                if not shape_ok: fails.append(f"shape {b['shape']}→{r['shape']}")
                print(f"  {name:<22s} FAIL  ({', '.join(fails)})")
                correctness_fail += 1
        else:
            print(f"  {name:<22s} NEW   (chk={r['checksum']:.4f} shape={r['shape']})")
            correctness_pass += 1

    # Performance
    print(f"\nPerformance:")
    print(f"  {'Benchmark':<22s} {'Qubits':>6s} {'N':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (46 + (22 if baseline else 0)))

    for name, _ in BENCHMARKS:
        r = results[name]
        line = f"  {name:<22s} {r['n_qubits']:>6d} {r['n_params_or_samples']:>6d} {r['time_ms']:>10.3f}"
        if name in baseline:
            b_time = baseline[name]["time_ms"]
            change = (r["time_ms"] - b_time) / b_time
            if change < -0.05:
                tag = f"{change:+.0%} FASTER"; perf_faster += 1
            elif change > 0.30:
                tag = f"{change:+.0%} REGR!"; perf_regression += 1
            elif change > 0.15:
                tag = f"{change:+.0%} warn"; perf_regression += 1
            else:
                tag = f"{change:+.0%}"
            line += f" {b_time:>10.3f} {tag:>10s}"
        print(line)

    total = correctness_pass + correctness_fail
    print(f"\nSummary: {correctness_pass}/{total} correctness PASS", end="")
    if correctness_fail: print(f", {correctness_fail} FAIL", end="")
    if baseline:
        print(f", {perf_faster} faster, {perf_regression} regressions", end="")
    print()

    if save_mode:
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nBaseline saved to {BASELINE_PATH}")

    return 1 if correctness_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
