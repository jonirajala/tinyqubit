"""Statevector simulator benchmark — correctness + performance tracking.

Tracks golden outputs (state hashes, expectation values) to catch regressions,
and times each circuit at multiple qubit counts to measure optimization progress.

Usage:
    ./venv/bin/python benchmarks/simulator_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/simulator_benchmark.py --save   # save new baseline
"""
import sys, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, simulate
from tinyqubit.measurement.observable import expectation, Z

BASELINE_PATH = Path(__file__).parent / "simulator_baseline.json"
N_WARMUP = 1
N_RUNS = 5
N_RUNS_FAST = 25  # more samples for sub-ms circuits (n <= 12)
PERF_WARN = 0.15   # 15% slower = warning
PERF_FAIL = 0.30   # 30% slower = fail
CORRECTNESS_TOL = 1e-10


# --- Circuit builders (TQ-only, no Qiskit/PL overhead) ---

def build_ghz(n):
    c = Circuit(n)
    c.h(0)
    for q in range(n - 1): c.cx(q, q + 1)
    return c

def build_qft(n):
    c = Circuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.cp(j, i, np.pi / (2 ** (j - i)))
    return c

def build_hea(n, layers=5):
    c = Circuit(n)
    rng = np.random.RandomState(42)
    for _ in range(layers):
        for q in range(n):
            c.ry(q, rng.uniform(0, 2 * np.pi))
            c.rz(q, rng.uniform(0, 2 * np.pi))
        for q in range(n - 1):
            c.cx(q, q + 1)
    return c

def build_clifford_t(n, depth=20):
    c = Circuit(n)
    rng = np.random.RandomState(42)
    for _ in range(depth):
        for q in range(n):
            g = rng.choice(3)
            if g == 0: c.h(q)
            elif g == 1: c.s(q)
            else: c.t(q)
        for q in range(0, n - 1, 2):
            if rng.random() < 0.5: c.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            if rng.random() < 0.5: c.cx(q, q + 1)
    return c

def build_toffoli_chain(n):
    c = Circuit(n)
    for q in range(n): c.h(q)  # non-trivial input state
    for q in range(n - 2): c.ccx(q, q + 1, q + 2)
    return c

def build_qaoa(n, layers=5):
    """QAOA-style: RX rotations + RZZ interactions — common in optimization."""
    c = Circuit(n)
    rng = np.random.RandomState(42)
    for q in range(n): c.h(q)
    for _ in range(layers):
        for q in range(n - 1):
            c.rzz(q, q + 1, rng.uniform(0, 2 * np.pi))
        for q in range(n):
            c.rx(q, rng.uniform(0, 2 * np.pi))
    return c

def build_random_circuit(n, depth=20):
    """Random 1Q rotations + random CX pairs — general stress test."""
    c = Circuit(n)
    rng = np.random.RandomState(42)
    for _ in range(depth):
        for q in range(n):
            axis = rng.choice(3)
            angle = rng.uniform(0, 2 * np.pi)
            if axis == 0: c.rx(q, angle)
            elif axis == 1: c.ry(q, angle)
            else: c.rz(q, angle)
        for q in range(0, n - 1, 2):
            if rng.random() < 0.6: c.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            if rng.random() < 0.6: c.cx(q, q + 1)
    return c

def build_diagonal_heavy(n, layers=10):
    """Heavy RZ/S/T/CZ circuit — tests diagonal gate optimizations."""
    c = Circuit(n)
    for q in range(n): c.h(q)
    rng = np.random.RandomState(42)
    for _ in range(layers):
        for q in range(n):
            c.rz(q, rng.uniform(0, 2 * np.pi))
            c.s(q)
        for q in range(0, n - 1, 2):
            c.cz(q, q + 1)
        for q in range(1, n - 1, 2):
            if q + 1 < n: c.cz(q, q + 1)
    return c


# --- Circuit suite ---
CIRCUITS = [
    ("ghz_4",         lambda: build_ghz(4)),
    ("ghz_12",        lambda: build_ghz(12)),
    ("ghz_20",        lambda: build_ghz(20)),
    ("ghz_23",        lambda: build_ghz(23)),
    ("qft_4",         lambda: build_qft(4)),
    ("qft_10",        lambda: build_qft(10)),
    ("qft_16",        lambda: build_qft(16)),
    ("hea_8",         lambda: build_hea(8)),
    ("hea_14",        lambda: build_hea(14)),
    ("hea_20",        lambda: build_hea(20)),
    ("clifford_t_10", lambda: build_clifford_t(10)),
    ("clifford_t_16", lambda: build_clifford_t(16)),
    ("toffoli_6",     lambda: build_toffoli_chain(6)),
    ("toffoli_10",    lambda: build_toffoli_chain(10)),
    ("diag_12",       lambda: build_diagonal_heavy(12)),
    ("diag_20",       lambda: build_diagonal_heavy(20)),
    ("qaoa_12",       lambda: build_qaoa(12)),
    ("qaoa_18",       lambda: build_qaoa(18)),
    ("random_12",     lambda: build_random_circuit(12)),
    ("random_18",     lambda: build_random_circuit(18)),
]


def state_hash(state):
    return hashlib.sha256(state.tobytes()).hexdigest()[:16]


def run_benchmark(name, builder):
    """Run a single benchmark: correctness check + timing."""
    circuit = builder()
    n = circuit.n_qubits

    # Correctness: simulate once, freeze outputs
    state, _ = simulate(circuit)
    h = state_hash(state)
    norm = float(np.linalg.norm(state))
    ez0 = float(expectation(state, Z(0), n_qubits=n))
    p0 = float(np.abs(state[0]) ** 2)

    # Performance: more runs for fast circuits to stabilize the median
    n_runs = N_RUNS_FAST if n <= 12 else N_RUNS
    for _ in range(N_WARMUP):
        simulate(builder())

    times = []
    for _ in range(n_runs):
        c = builder()
        t0 = time.perf_counter()
        simulate(c)
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {"hash": h, "norm": norm, "ez0": round(ez0, 12), "p0": round(p0, 12),
            "time_ms": round(time_ms, 3), "n_qubits": n, "n_ops": len(circuit.ops)}


def main():
    save_mode = "--save" in sys.argv

    print("=== TinyQubit Statevector Benchmark ===\n")

    # Load baseline if exists
    baseline = {}
    if BASELINE_PATH.exists() and not save_mode:
        baseline = json.loads(BASELINE_PATH.read_text())
        print(f"Baseline loaded: {BASELINE_PATH.name} ({len(baseline)} circuits)\n")
    elif not save_mode:
        print("No baseline found. Run with --save to create one.\n")

    results = {}
    correctness_pass = 0
    correctness_fail = 0
    perf_faster = 0
    perf_regression = 0

    # --- Correctness ---
    print("Correctness:")
    for name, builder in CIRCUITS:
        r = run_benchmark(name, builder)
        results[name] = r

        if name in baseline:
            b = baseline[name]
            norm_ok = abs(r["norm"] - 1.0) < CORRECTNESS_TOL
            ez0_ok = abs(r["ez0"] - b["ez0"]) < CORRECTNESS_TOL
            p0_ok = abs(r["p0"] - b["p0"]) < CORRECTNESS_TOL

            if norm_ok and ez0_ok and p0_ok:
                print(f"  {name:<20s} PASS  (norm={r['norm']:.10f} ez0={r['ez0']:.6f})")
                correctness_pass += 1
            else:
                fails = []
                if not norm_ok: fails.append(f"norm={r['norm']}")
                if not ez0_ok: fails.append(f"ez0 {b['ez0']}→{r['ez0']}")
                if not p0_ok: fails.append(f"p0 {b['p0']}→{r['p0']}")
                print(f"  {name:<20s} FAIL  ({', '.join(fails)})")
                correctness_fail += 1
        else:
            print(f"  {name:<20s} NEW   (hash={r['hash'][:8]}.. norm={r['norm']:.10f})")
            correctness_pass += 1

    # --- Performance ---
    print(f"\nPerformance:")
    print(f"  {'Circuit':<20s} {'Qubits':>6s} {'Ops':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (54 + (22 if baseline else 0)))

    for name, _ in CIRCUITS:
        r = results[name]
        line = f"  {name:<20s} {r['n_qubits']:>6d} {r['n_ops']:>6d} {r['time_ms']:>10.3f}"

        if name in baseline:
            b_time = baseline[name]["time_ms"]
            change = (r["time_ms"] - b_time) / b_time
            if change < -0.05:
                tag = f"{change:+.0%} FASTER"
                perf_faster += 1
            elif change > PERF_FAIL:
                tag = f"{change:+.0%} REGR!"
                perf_regression += 1
            elif change > PERF_WARN:
                tag = f"{change:+.0%} warn"
                perf_regression += 1
            else:
                tag = f"{change:+.0%}"
            line += f" {b_time:>10.3f} {tag:>10s}"
        print(line)

    # --- Summary ---
    total = correctness_pass + correctness_fail
    print(f"\nSummary: {correctness_pass}/{total} correctness PASS", end="")
    if correctness_fail: print(f", {correctness_fail} FAIL", end="")
    if baseline:
        print(f", {perf_faster} faster, {perf_regression} regressions", end="")
    print()

    # --- Save baseline ---
    if save_mode:
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nBaseline saved to {BASELINE_PATH}")

    return 1 if correctness_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
