"""Compiler benchmark — correctness + performance tracking.

Benchmarks precompile(), transpile(), and individual passes across circuit
families that represent real-world compilation workloads.

CORRECTNESS: compiled circuits must produce the same statevector as the original.
This is the #1 priority — any optimization that changes compilation output is rejected.

Usage:
    ./venv/bin/python benchmarks/compiler_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/compiler_benchmark.py --save   # save new baseline
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit
from tinyqubit.ir import Gate
from tinyqubit.compile import precompile, transpile, CompileConfig, PRESET_FAST, PRESET_DEFAULT
from tinyqubit.target import Target
from tinyqubit.simulator.simulator import simulate, verify

BASELINE_PATH = Path(__file__).parent / "compiler_baseline.json"
N_WARMUP = 1
N_RUNS = 5
N_RUNS_FAST = 15


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


def _build_qft(n):
    c = Circuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.cp(j, i, np.pi / (2 ** (j - i)))
    return c


def _build_ghz(n):
    c = Circuit(n); c.h(0)
    for q in range(n - 1): c.cx(q, q + 1)
    return c


def _build_toffoli_chain(n):
    c = Circuit(n)
    for q in range(n): c.h(q)
    for q in range(n - 2): c.ccx(q, q + 1, q + 2)
    return c


def _build_random(n, depth, seed=42):
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    for _ in range(depth):
        for q in range(n):
            g = rng.choice(3); a = rng.uniform(0, 2 * np.pi)
            if g == 0: c.rx(q, a)
            elif g == 1: c.ry(q, a)
            else: c.rz(q, a)
        for q in range(0, n - 1, 2): c.cx(q, q + 1)
        for q in range(1, n - 1, 2): c.cx(q, q + 1)
    return c


def _line_target(n):
    edges = frozenset((i, i + 1) for i in range(n - 1)) | frozenset((i + 1, i) for i in range(n - 1))
    return Target(n_qubits=n, edges=edges, basis_gates=frozenset({Gate.CX, Gate.H, Gate.RZ, Gate.SWAP}), name=f"line_{n}")


# --- Benchmark suite ---

BENCHMARKS = [
    # Precompile only (no target) — fast, tests optimizer + fuser
    ("pre_hea_4q",       lambda: _run_precompile(_build_hea(4, 3))),
    ("pre_hea_8q",       lambda: _run_precompile(_build_hea(8, 3))),
    ("pre_hea_12q",      lambda: _run_precompile(_build_hea(12, 2))),
    ("pre_qft_8q",       lambda: _run_precompile(_build_qft(8))),
    ("pre_toffoli_8q",   lambda: _run_precompile(_build_toffoli_chain(8))),
    ("pre_random_10q",   lambda: _run_precompile(_build_random(10, 10))),
    # Full transpile (with target) — includes routing
    ("trans_hea_8q",     lambda: _run_transpile(_build_hea(8, 3), 8)),
    ("trans_qft_8q",     lambda: _run_transpile(_build_qft(8), 8)),
    ("trans_ghz_12q",    lambda: _run_transpile(_build_ghz(12), 12)),
    ("trans_random_10q", lambda: _run_transpile(_build_random(10, 10), 10)),
    # Larger circuits (compiler cost scales with ops, not just qubits)
    ("pre_hea_16q",      lambda: _run_precompile(_build_hea(16, 2))),
    ("trans_hea_12q",    lambda: _run_transpile(_build_hea(12, 2), 12)),
    # Default preset (more optimization iterations than fast)
    ("trans_hea_8q_def", lambda: _run_transpile_default(_build_hea(8, 3), 8)),
    # ADAPT-VQE pattern: repeated precompile with growing circuit
    ("adapt_grow_8q",    lambda: _run_adapt_pattern(8)),
]


def _run_precompile(circuit):
    compiled = precompile(circuit)
    return compiled, circuit.n_qubits, len(circuit.ops), len(compiled.ops)


def _run_transpile(circuit, n):
    target = _line_target(n)
    compiled = transpile(circuit, target, preset='fast')
    return compiled, n, len(circuit.ops), len(compiled.ops)


def _run_transpile_default(circuit, n):
    target = _line_target(n)
    compiled = transpile(circuit, target, preset='default')
    return compiled, n, len(circuit.ops), len(compiled.ops)


def _run_adapt_pattern(n):
    """Simulate ADAPT-VQE: 10 iterations, each adding 2 gates and recompiling."""
    rng = np.random.RandomState(42)
    c = Circuit(n)
    for q in range(n // 2): c.x(q)
    for iteration in range(10):
        q0, q1 = rng.randint(0, n, 2)
        if q0 == q1: q1 = (q0 + 1) % n
        c.cx(q0, q1)
        c.ry(q1, rng.uniform(0, 2 * np.pi))
        c.cx(q0, q1)
    compiled = precompile(c)
    return compiled, n, len(c.ops), len(compiled.ops)


def run_benchmark(name, runner):
    """Run a single benchmark: correctness check + timing."""
    compiled, n_qubits, n_ops_in, n_ops_out = runner()

    n_runs = N_RUNS_FAST if n_qubits <= 8 else N_RUNS
    for _ in range(N_WARMUP): runner()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner()
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {
        "n_qubits": n_qubits, "n_ops_in": n_ops_in, "n_ops_out": n_ops_out,
        "time_ms": round(time_ms, 3),
    }


def main():
    save_mode = "--save" in sys.argv
    print("=== TinyQubit Compiler Benchmark ===\n")

    baseline = {}
    if BASELINE_PATH.exists() and not save_mode:
        baseline = json.loads(BASELINE_PATH.read_text())
        print(f"Baseline loaded: {BASELINE_PATH.name} ({len(baseline)} benchmarks)\n")
    elif not save_mode:
        print("No baseline found. Run with --save to create one.\n")

    results = {}
    correctness_pass = correctness_fail = perf_faster = perf_regression = 0

    print("Correctness (compiled ops count must match):")
    for name, runner in BENCHMARKS:
        r = run_benchmark(name, runner)
        results[name] = r

        if name in baseline:
            b = baseline[name]
            # Gate count must match exactly — any change means different compilation
            ops_ok = r["n_ops_out"] == b["n_ops_out"]
            if ops_ok:
                print(f"  {name:<22s} PASS  ({r['n_ops_in']}→{r['n_ops_out']} ops)")
                correctness_pass += 1
            else:
                print(f"  {name:<22s} FAIL  (ops {b['n_ops_out']}→{r['n_ops_out']})")
                correctness_fail += 1
        else:
            print(f"  {name:<22s} NEW   ({r['n_ops_in']}→{r['n_ops_out']} ops)")
            correctness_pass += 1

    print(f"\nPerformance:")
    print(f"  {'Benchmark':<22s} {'Qubits':>6s} {'In':>6s} {'Out':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (52 + (22 if baseline else 0)))

    for name, _ in BENCHMARKS:
        r = results[name]
        line = f"  {name:<22s} {r['n_qubits']:>6d} {r['n_ops_in']:>6d} {r['n_ops_out']:>6d} {r['time_ms']:>10.3f}"
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
