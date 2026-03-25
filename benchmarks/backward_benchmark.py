"""Adjoint backward pass benchmark — correctness + performance tracking.

Isolates the backward pass performance from the forward simulation and lambda
computation. Tests across circuit families, qubit counts, and parameter densities
that represent real VQE/QML workloads.

Correctness: gradient must match finite-difference reference within tolerance.
The backward is the ONLY component being timed — forward sim is excluded.

Usage:
    ./venv/bin/python benchmarks/backward_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/backward_benchmark.py --save   # save new baseline
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, Parameter, simulate
from tinyqubit.measurement.observable import Observable, X, Y, Z, expectation
from tinyqubit.qml.optim import adjoint_gradient, finite_difference_gradient, _adjoint_backward, _build_adjoint_info
from tinyqubit.ir import _GATE_ADJOINT, _PARAM_GATES

BASELINE_PATH = Path(__file__).parent / "backward_baseline.json"
N_WARMUP = 2
N_RUNS = 5
N_RUNS_FAST = 25
CORRECTNESS_TOL = 1e-5


# --- Circuit builders ---

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


def _build_qaoa(n, p, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for q in range(n): c.h(q)
    for l in range(p):
        gn, bn = f"gamma_{l}", f"beta_{l}"
        params[gn] = rng.uniform(0, 2 * np.pi)
        params[bn] = rng.uniform(0, 2 * np.pi)
        for q in range(n - 1): c.rzz(q, q + 1, Parameter(gn))
        for q in range(n): c.rx(q, Parameter(bn))
    c.param_values = params
    return c


def _build_uccsd(n, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for q in range(n // 2): c.x(q)
    for i in range(n // 2):
        for a in range(n // 2, n):
            pn = f"s_{i}_{a}"
            params[pn] = rng.uniform(-0.5, 0.5)
            c.cx(i, a); c.ry(a, Parameter(pn)); c.cx(i, a)
    c.param_values = params
    return c


def _build_random(n, depth, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for d in range(depth):
        for q in range(n):
            pn = f"p_{d}_{q}"
            params[pn] = rng.uniform(0, 2 * np.pi)
            g = rng.choice(3)
            if g == 0: c.rx(q, Parameter(pn))
            elif g == 1: c.ry(q, Parameter(pn))
            else: c.rz(q, Parameter(pn))
        for q in range(0, n - 1, 2): c.cx(q, q + 1)
        for q in range(1, n - 1, 2): c.cx(q, q + 1)
    c.param_values = params
    return c


def _build_swap_heavy(n, layers, seed=42):
    """Compiled-circuit-like: 1Q rotations + SWAP+CX routing chains."""
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for l in range(layers):
        for q in range(n):
            pn = f"ry_{l}_{q}"
            params[pn] = rng.uniform(0, 2 * np.pi)
            c.ry(q, Parameter(pn))
        for _ in range(n // 3):
            q0, q1 = rng.randint(0, n), rng.randint(0, n)
            if q0 == q1: continue
            lo, hi = min(q0, q1), max(q0, q1)
            for s in range(lo, hi - 1): c.swap(s, s + 1)
            c.cx(hi - 1, hi)
            for s in range(hi - 2, lo - 1, -1): c.swap(s, s + 1)
    c.param_values = params
    return c


# --- Observable builders ---

def _z_sum(n): return sum(Z(q) for q in range(n))

def _heisenberg(n):
    terms = []
    for q in range(n - 1):
        for p in ['X', 'Y', 'Z']:
            terms.append((1.0, {q: p, q + 1: p}))
    return Observable(terms)

def _random_hamiltonian(n, n_terms, seed=42):
    rng = np.random.RandomState(seed)
    terms = []
    for _ in range(n_terms):
        coeff = rng.uniform(-1, 1)
        nq = rng.randint(1, min(n, 4) + 1)
        qs = rng.choice(n, size=nq, replace=False)
        terms.append((coeff, {int(q): rng.choice(['X', 'Y', 'Z']) for q in qs}))
    return Observable(terms)


# --- Benchmark suite ---

BENCHMARKS = [
    # HEA: the #1 VQE workload. RZ-RY fusion + kron grouping is the main optimization.
    ("hea_4q_3l",      lambda: _run("hea", 4, 3, _z_sum)),
    ("hea_8q_3l",      lambda: _run("hea", 8, 3, _z_sum)),
    ("hea_12q_3l",     lambda: _run("hea", 12, 3, _z_sum)),
    ("hea_16q_3l",     lambda: _run("hea", 16, 3, _z_sum)),
    ("hea_20q_2l",     lambda: _run("hea", 20, 2, _z_sum)),
    # HEA with Heisenberg Hamiltonian (mixed Pauli terms in lambda)
    ("hea_8q_heisen",  lambda: _run("hea", 8, 3, _heisenberg)),
    # QAOA: RZZ parametric 2Q gates (different backward path than RY/RZ)
    ("qaoa_8q_p3",     lambda: _run("qaoa", 8, 3, _z_sum)),
    ("qaoa_14q_p3",    lambda: _run("qaoa", 14, 3, _z_sum)),
    # UCCSD: CX-RY-CX ladders (no consecutive same-qubit RZ-RY pairs)
    ("uccsd_8q",       lambda: _run("uccsd", 8, 0, lambda n: _random_hamiltonian(n, 30))),
    # Random circuit: mixed RX/RY/RZ (tests non-uniform gate distribution)
    ("random_10q_d5",  lambda: _run("random", 10, 5, _z_sum)),
    ("random_14q_d5",  lambda: _run("random", 14, 5, _z_sum)),
    # SWAP-heavy routed circuit: tests SWAP+CX perm batching in backward
    ("swap_10q",       lambda: _run("swap", 10, 5, _z_sum)),
    # Large Hamiltonian: many Pauli terms (tests lambda computation)
    ("hea_8q_bigH",    lambda: _run("hea", 8, 3, lambda n: _random_hamiltonian(n, 100))),
]


def _run(circuit_type, n, layers_or_depth, obs_fn):
    if circuit_type == "hea": circuit = _build_hea(n, layers_or_depth)
    elif circuit_type == "qaoa": circuit = _build_qaoa(n, layers_or_depth)
    elif circuit_type == "uccsd": circuit = _build_uccsd(n)
    elif circuit_type == "random": circuit = _build_random(n, layers_or_depth)
    elif circuit_type == "swap": circuit = _build_swap_heavy(n, layers_or_depth)
    obs = obs_fn(n)
    params = circuit.param_values

    # Correctness: compare with finite-difference
    g_adj = adjoint_gradient(circuit, obs, params)
    g_fd = finite_difference_gradient(circuit, obs, params)
    max_err = max(abs(g_adj[k] - g_fd[k]) for k in params)
    grad_norm = float(np.sqrt(sum(v ** 2 for v in g_adj.values())))

    # Timing: time the FULL adjoint_gradient (includes forward + lambda + backward)
    n_runs = N_RUNS_FAST if n <= 10 else N_RUNS
    for _ in range(N_WARMUP): adjoint_gradient(circuit, obs, params)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        adjoint_gradient(circuit, obs, params)
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {
        "max_err": round(max_err, 10), "grad_norm": round(grad_norm, 8),
        "n_qubits": n, "n_params": len(params), "n_ops": len(circuit.ops),
        "time_ms": round(time_ms, 3),
    }


def run_benchmark(name, runner):
    r = runner()
    return r


def main():
    save_mode = "--save" in sys.argv
    print("=== TinyQubit Adjoint Backward Benchmark ===\n")

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
            err_ok = r["max_err"] < CORRECTNESS_TOL
            norm_ok = abs(r["grad_norm"] - b["grad_norm"]) < 1e-4 or abs(r["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-10) < 0.01
            if err_ok and norm_ok:
                print(f"  {name:<20s} PASS  (err={r['max_err']:.2e} |g|={r['grad_norm']:.4f})")
                correctness_pass += 1
            else:
                fails = []
                if not err_ok: fails.append(f"max_err={r['max_err']:.2e}")
                if not norm_ok: fails.append(f"|g| {b['grad_norm']:.4f}→{r['grad_norm']:.4f}")
                print(f"  {name:<20s} FAIL  ({', '.join(fails)})")
                correctness_fail += 1
        else:
            print(f"  {name:<20s} NEW   (err={r['max_err']:.2e} |g|={r['grad_norm']:.4f})")
            correctness_pass += 1

    print(f"\nPerformance:")
    print(f"  {'Benchmark':<20s} {'Qubits':>6s} {'Params':>6s} {'Ops':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (50 + (22 if baseline else 0)))

    for name, _ in BENCHMARKS:
        r = results[name]
        line = f"  {name:<20s} {r['n_qubits']:>6d} {r['n_params']:>6d} {r['n_ops']:>6d} {r['time_ms']:>10.3f}"
        if name in baseline:
            b_time = baseline[name]["time_ms"]
            change = (r["time_ms"] - b_time) / b_time
            if change < -0.05:
                tag = f"{change:+.0%} FASTER"; perf_faster += 1
            elif change > 0.30:
                tag = f"{change:+.0%} REGR!"; perf_regression += 1
            elif change > 0.15:
                tag = f"{change:+.0%} warn"
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
