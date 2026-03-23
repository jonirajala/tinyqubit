"""Noisy simulation benchmark — correctness + performance tracking.

Benchmarks noisy simulation (Monte Carlo trajectories), noise channel application,
and ZNE error mitigation across circuit families from real chemistry/QML workloads.

Usage:
    ./venv/bin/python benchmarks/noisy_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/noisy_benchmark.py --save   # save new baseline
"""
import sys, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, Parameter, simulate
from tinyqubit.measurement.observable import Observable, Z, expectation
from tinyqubit.simulator.noise import (NoiseModel, depolarizing, amplitude_damping, phase_damping, realistic_noise)

BASELINE_PATH = Path(__file__).parent / "noisy_baseline.json"
N_WARMUP = 1
N_RUNS = 5
N_RUNS_FAST = 15
CORRECTNESS_TOL = 0.05  # noisy simulation is stochastic — needs wider tolerance
SEED = 42


# --- Circuit builders ---

def _build_hea(n, layers, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    for l in range(layers):
        for q in range(n):
            c.ry(q, rng.uniform(0, 2 * np.pi))
            c.rz(q, rng.uniform(0, 2 * np.pi))
        for q in range(n - 1): c.cx(q, q + 1)
    return c


def _build_ghz(n):
    c = Circuit(n)
    c.h(0)
    for q in range(n - 1): c.cx(q, q + 1)
    return c


def _build_qaoa(n, p, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    for q in range(n): c.h(q)
    for l in range(p):
        gamma, beta = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        for q in range(n - 1): c.rzz(q, q + 1, gamma)
        for q in range(n): c.rx(q, beta)
    return c


def _build_uccsd(n, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    for q in range(n // 2): c.x(q)
    for i in range(n // 2):
        for a in range(n // 2, n):
            c.cx(i, a); c.ry(a, rng.uniform(-0.5, 0.5)); c.cx(i, a)
    return c


# --- Noise models ---

def _depol_only(p):
    return NoiseModel().add_depolarizing(p)


def _amp_damp_only(gamma):
    return NoiseModel().add_amplitude_damping(gamma)


def _full_realistic():
    return realistic_noise(depolarizing_1q=0.001, depolarizing_2q=0.01, readout_err=0.0)


# --- Benchmark suite ---

BENCHMARKS = [
    # Depolarizing noise — the most common model
    ("depol_hea_4q",        lambda: _run_noisy(_build_hea(4, 3), _depol_only(0.001), 4, 100)),
    ("depol_hea_8q",        lambda: _run_noisy(_build_hea(8, 3), _depol_only(0.001), 8, 50)),
    ("depol_hea_12q",       lambda: _run_noisy(_build_hea(12, 2), _depol_only(0.001), 12, 20)),
    ("depol_hea_16q",       lambda: _run_noisy(_build_hea(16, 2), _depol_only(0.001), 16, 10)),
    # Amplitude damping — T1 decay
    ("ampdamp_ghz_8q",      lambda: _run_noisy(_build_ghz(8), _amp_damp_only(0.01), 8, 50)),
    # Realistic combined noise (depol + amp damp + phase damp)
    ("realistic_hea_8q",    lambda: _run_noisy(_build_hea(8, 3), _full_realistic(), 8, 30)),
    ("realistic_uccsd_8q",  lambda: _run_noisy(_build_uccsd(8), _full_realistic(), 8, 30)),
    ("realistic_qaoa_10q",  lambda: _run_noisy(_build_qaoa(10, 3), _full_realistic(), 10, 20)),
    # ZNE workload: multiple simulations at different noise scales
    ("zne_hea_8q",          lambda: _run_zne(_build_hea(8, 3), 8)),
    # High-noise regime (common in NISQ error mitigation research)
    ("highnoise_hea_8q",    lambda: _run_noisy(_build_hea(8, 3), _depol_only(0.01), 8, 50)),
    # Many-shot noisy expectation (statistical averaging)
    ("noisy_exp_8q_200",    lambda: _run_noisy_expectation(_build_hea(8, 3), _depol_only(0.001), 8, 200)),
]


def _run_noisy(circuit, noise, n, n_sims):
    """Run n_sims noisy simulations, return array of Z(0) expectations."""
    rng = np.random.default_rng(SEED)
    results = np.empty(n_sims)
    obs = Z(0)
    for i in range(n_sims):
        state, _ = simulate(circuit, noise_model=noise, seed=int(rng.integers(2**32)))
        results[i] = expectation(state, obs, n_qubits=n)
    return results, n, n_sims


def _run_zne(circuit, n):
    """ZNE-style workload: simulate at 3 noise scales x 30 shots each."""
    from tinyqubit.measurement.mitigation import _fold_circuit
    rng = np.random.default_rng(SEED)
    noise = _depol_only(0.005)
    obs = Z(0)
    results = []
    for scale in [1, 3, 5]:
        folded = _fold_circuit(circuit, scale)
        vals = []
        for _ in range(30):
            state, _ = simulate(folded, noise_model=noise, seed=int(rng.integers(2**32)))
            vals.append(expectation(state, obs, n_qubits=n))
        results.append(np.mean(vals))
    return np.array(results), n, 90  # 3 scales × 30 shots


def _run_noisy_expectation(circuit, noise, n, n_shots):
    """Noisy expectation via shot averaging — VQE cost function pattern."""
    rng = np.random.default_rng(SEED)
    obs = sum(Z(q) for q in range(n))
    vals = np.empty(n_shots)
    for i in range(n_shots):
        state, _ = simulate(circuit, noise_model=noise, seed=int(rng.integers(2**32)))
        vals[i] = expectation(state, obs, n_qubits=n)
    return vals, n, n_shots


def run_benchmark(name, runner):
    result, n_qubits, n_evals = runner()
    mean_val = float(np.mean(result))
    std_val = float(np.std(result))

    n_runs = N_RUNS_FAST if n_qubits <= 8 and n_evals <= 50 else N_RUNS
    for _ in range(N_WARMUP): runner()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner()
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {
        "mean": round(mean_val, 6), "std": round(std_val, 6),
        "n_qubits": n_qubits, "n_evals": n_evals,
        "time_ms": round(time_ms, 3),
    }


def main():
    save_mode = "--save" in sys.argv

    print("=== TinyQubit Noisy Simulation Benchmark ===\n")

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
            # Stochastic: check mean is within tolerance (wider than deterministic tests)
            mean_ok = abs(r["mean"] - b["mean"]) < CORRECTNESS_TOL
            if mean_ok:
                print(f"  {name:<24s} PASS  (mean={r['mean']:.4f}±{r['std']:.4f})")
                correctness_pass += 1
            else:
                print(f"  {name:<24s} FAIL  (mean {b['mean']:.4f}→{r['mean']:.4f})")
                correctness_fail += 1
        else:
            print(f"  {name:<24s} NEW   (mean={r['mean']:.4f}±{r['std']:.4f})")
            correctness_pass += 1

    print(f"\nPerformance:")
    print(f"  {'Benchmark':<24s} {'Qubits':>6s} {'Evals':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (48 + (22 if baseline else 0)))

    for name, _ in BENCHMARKS:
        r = results[name]
        line = f"  {name:<24s} {r['n_qubits']:>6d} {r['n_evals']:>6d} {r['time_ms']:>10.3f}"
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
