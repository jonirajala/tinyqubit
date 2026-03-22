"""Gradient computation benchmark — correctness + performance tracking.

Benchmarks adjoint_gradient, parameter_shift_gradient, and backprop_gradient
across circuit families that represent real-world QML/VQE workloads.

Usage:
    ./venv/bin/python benchmarks/gradient_benchmark.py          # compare against baseline
    ./venv/bin/python benchmarks/gradient_benchmark.py --save   # save new baseline
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, Parameter
from tinyqubit.measurement.observable import Observable, X, Y, Z, expectation
from tinyqubit.qml.optim import adjoint_gradient, parameter_shift_gradient, backprop_gradient

BASELINE_PATH = Path(__file__).parent / "gradient_baseline.json"
N_WARMUP = 1
N_RUNS = 5
N_RUNS_FAST = 25  # more samples for fast circuits (n <= 10)
CORRECTNESS_TOL = 1e-5  # backprop uses finite-diff internally, needs more slack
PERF_WARN = 0.15
PERF_FAIL = 0.30


# --- Observable builders ---

def _random_hamiltonian(n, n_terms, seed=42):
    """Random Pauli-sum Hamiltonian with n_terms terms."""
    rng = np.random.RandomState(seed)
    terms = []
    paulis_list = ['X', 'Y', 'Z']
    for _ in range(n_terms):
        coeff = rng.uniform(-1, 1)
        n_paulis = rng.randint(1, min(n, 4) + 1)
        qubits = rng.choice(n, size=n_paulis, replace=False)
        paulis = {int(q): rng.choice(paulis_list) for q in qubits}
        terms.append((coeff, paulis))
    return Observable(terms)


def _z_hamiltonian(n):
    """Sum of Z on all qubits — simple diagonal Hamiltonian."""
    return sum(Z(q) for q in range(n))


def _heisenberg_hamiltonian(n):
    """1D Heisenberg chain: sum of XX + YY + ZZ on adjacent pairs."""
    terms = []
    for q in range(n - 1):
        for pauli in ['X', 'Y', 'Z']:
            terms.append((1.0, {q: pauli, q + 1: pauli}))
    return Observable(terms)


# --- Circuit builders ---

def build_hea(n, layers, seed=42):
    """Hardware-efficient ansatz: RY+RZ rotations + CX entanglement."""
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for l in range(layers):
        for q in range(n):
            name_y = f"ry_{l}_{q}"
            name_z = f"rz_{l}_{q}"
            c.ry(q, Parameter(name_y))
            c.rz(q, Parameter(name_z))
            params[name_y] = rng.uniform(0, 2 * np.pi)
            params[name_z] = rng.uniform(0, 2 * np.pi)
        for q in range(n - 1):
            c.cx(q, q + 1)
    c.param_values = params
    return c


def build_qaoa(n, p, seed=42):
    """QAOA ansatz: RZZ interactions + RX mixing."""
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for q in range(n): c.h(q)
    for l in range(p):
        gamma = f"gamma_{l}"
        beta = f"beta_{l}"
        params[gamma] = rng.uniform(0, 2 * np.pi)
        params[beta] = rng.uniform(0, 2 * np.pi)
        for q in range(n - 1):
            c.rzz(q, q + 1, Parameter(gamma))
        for q in range(n):
            c.rx(q, Parameter(beta))
    c.param_values = params
    return c


def build_uccsd_like(n, seed=42):
    """UCCSD-inspired ansatz: RZ+RY rotations + CX ladders (simulates excitation operators)."""
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    # Hartree-Fock init: half qubits in |1⟩
    for q in range(n // 2): c.x(q)
    # Single excitations
    for i in range(n // 2):
        for a in range(n // 2, n):
            name = f"s_{i}_{a}"
            params[name] = rng.uniform(-0.5, 0.5)
            c.cx(i, a)
            c.ry(a, Parameter(name))
            c.cx(i, a)
    # Double excitations (simplified)
    for i in range(0, n // 2 - 1):
        name = f"d_{i}"
        params[name] = rng.uniform(-0.5, 0.5)
        c.cx(i, i + 1)
        c.cx(i + 1, n // 2 + i)
        c.ry(n // 2 + i, Parameter(name))
        c.cx(i + 1, n // 2 + i)
        c.cx(i, i + 1)
    c.param_values = params
    return c


def build_data_reuploading(n_features, n_layers, seed=42):
    """Data re-uploading classifier: feature encoding + trainable rotations."""
    c = Circuit(1)
    rng = np.random.RandomState(seed)
    params = {}
    for l in range(n_layers):
        for f in range(n_features):
            fname = f"x_{l}_{f}"
            c.rx(0, Parameter(fname))
            params[fname] = rng.uniform(0, 2 * np.pi)
        wname = f"w_{l}"
        c.ry(0, Parameter(wname))
        params[wname] = rng.uniform(0, 2 * np.pi)
    c.param_values = params
    return c


def build_random_circuit(n, depth, seed=42):
    """Random parametric circuit — stress test for general gradient computation."""
    c = Circuit(n)
    rng = np.random.RandomState(seed)
    params = {}
    for d in range(depth):
        for q in range(n):
            gate = rng.choice(3)
            name = f"p_{d}_{q}"
            params[name] = rng.uniform(0, 2 * np.pi)
            if gate == 0: c.rx(q, Parameter(name))
            elif gate == 1: c.ry(q, Parameter(name))
            else: c.rz(q, Parameter(name))
        for q in range(0, n - 1, 2):
            c.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            c.cx(q, q + 1)
    c.param_values = params
    return c


# --- Benchmark suite ---

BENCHMARKS = [
    # (name, circuit_builder, observable_builder, gradient_fn)
    # VQE-style: adjoint gradient on Hamiltonian expectation
    ("vqe_hea_4q_3l",    lambda: build_hea(4, 3),              lambda n: _z_hamiltonian(n),          adjoint_gradient),
    ("vqe_hea_8q_3l",    lambda: build_hea(8, 3),              lambda n: _z_hamiltonian(n),          adjoint_gradient),
    ("vqe_hea_12q_3l",   lambda: build_hea(12, 3),             lambda n: _z_hamiltonian(n),          adjoint_gradient),
    ("vqe_hea_16q_3l",   lambda: build_hea(16, 3),             lambda n: _z_hamiltonian(n),          adjoint_gradient),
    ("vqe_hea_8q_heisen", lambda: build_hea(8, 3),             lambda n: _heisenberg_hamiltonian(n), adjoint_gradient),
    # UCCSD-style: chemistry with mixed Pauli Hamiltonian
    ("uccsd_4q",         lambda: build_uccsd_like(4),           lambda n: _random_hamiltonian(n, 10), adjoint_gradient),
    ("uccsd_8q",         lambda: build_uccsd_like(8),           lambda n: _random_hamiltonian(n, 30), adjoint_gradient),
    # QAOA-style: RZZ + RX with diagonal Hamiltonian
    ("qaoa_8q_p3",       lambda: build_qaoa(8, 3),              lambda n: _z_hamiltonian(n),          adjoint_gradient),
    ("qaoa_14q_p3",      lambda: build_qaoa(14, 3),             lambda n: _z_hamiltonian(n),          adjoint_gradient),
    # QML: data re-uploading (many params, 1 qubit)
    ("reupl_1q_4f_6l",   lambda: build_data_reuploading(4, 6),  lambda n: Z(0),                      adjoint_gradient),
    # Random circuit: stress test
    ("random_8q_d5",     lambda: build_random_circuit(8, 5),    lambda n: _random_hamiltonian(n, 15), adjoint_gradient),
    ("random_14q_d5",    lambda: build_random_circuit(14, 5),   lambda n: _random_hamiltonian(n, 15), adjoint_gradient),
    # Parameter-shift: compare overhead vs adjoint
    ("pshift_hea_8q_3l", lambda: build_hea(8, 3),              lambda n: _z_hamiltonian(n),          parameter_shift_gradient),
    # Backprop: loss(probabilities) gradient
    ("backprop_hea_8q",  lambda: build_hea(8, 3),              None,                                 "backprop"),
]


def run_benchmark(name, circuit_builder, obs_builder, grad_fn):
    """Run a single gradient benchmark: correctness + timing."""
    circuit = circuit_builder()
    n = circuit.n_qubits
    params = circuit.param_values

    # Correctness: compute gradient, verify against finite-difference
    if grad_fn == "backprop":
        loss_fn = lambda p: float(np.sum(p[:len(p)//2]))  # sum of first half probabilities
        grad = backprop_gradient(circuit, loss_fn, params)
        # Finite-diff reference
        ref_grad = {}
        bound = circuit.bind(params)
        state0, _ = __import__('tinyqubit').simulate(bound)
        probs0 = np.abs(state0) ** 2
        cost0 = loss_fn(probs0)
        for k in params:
            pp = {**params, k: params[k] + 1e-5}
            sp, _ = __import__('tinyqubit').simulate(circuit.bind(pp))
            ref_grad[k] = (loss_fn(np.abs(sp)**2) - cost0) / 1e-5
    else:
        obs = obs_builder(n)
        grad = grad_fn(circuit, obs, params)
        # Finite-diff reference
        ref_grad = {}
        for k in sorted(params):
            pp = {**params, k: params[k] + 1e-5}
            pm = {**params, k: params[k] - 1e-5}
            ref_grad[k] = (expectation(circuit.bind(pp), obs) - expectation(circuit.bind(pm), obs)) / 2e-5

    # Check gradient correctness
    max_err = max(abs(grad[k] - ref_grad[k]) for k in params)
    grad_norm = np.sqrt(sum(g ** 2 for g in grad.values()))

    # Timing
    n_runs = N_RUNS_FAST if n <= 10 else N_RUNS
    for _ in range(N_WARMUP):
        if grad_fn == "backprop":
            backprop_gradient(circuit, loss_fn, params)
        else:
            grad_fn(circuit, obs, params)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if grad_fn == "backprop":
            backprop_gradient(circuit, loss_fn, params)
        else:
            grad_fn(circuit, obs, params)
        times.append((time.perf_counter() - t0) * 1000)
    time_ms = float(np.median(times))

    return {
        "max_err": round(max_err, 10), "grad_norm": round(float(grad_norm), 10),
        "n_params": len(params), "n_qubits": n, "n_ops": len(circuit.ops),
        "time_ms": round(time_ms, 3),
    }


def main():
    save_mode = "--save" in sys.argv

    print("=== TinyQubit Gradient Benchmark ===\n")

    baseline = {}
    if BASELINE_PATH.exists() and not save_mode:
        baseline = json.loads(BASELINE_PATH.read_text())
        print(f"Baseline loaded: {BASELINE_PATH.name} ({len(baseline)} benchmarks)\n")
    elif not save_mode:
        print("No baseline found. Run with --save to create one.\n")

    results = {}
    correctness_pass = correctness_fail = perf_faster = perf_regression = 0

    print("Correctness:")
    for name, circ_builder, obs_builder, grad_fn in BENCHMARKS:
        r = run_benchmark(name, circ_builder, obs_builder, grad_fn)
        results[name] = r

        if name in baseline:
            b = baseline[name]
            err_ok = r["max_err"] < CORRECTNESS_TOL
            norm_ok = abs(r["grad_norm"] - b["grad_norm"]) < 1e-4 or abs(r["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-10) < 0.01

            if err_ok and norm_ok:
                print(f"  {name:<22s} PASS  (err={r['max_err']:.2e} |g|={r['grad_norm']:.6f})")
                correctness_pass += 1
            else:
                fails = []
                if not err_ok: fails.append(f"max_err={r['max_err']:.2e}")
                if not norm_ok: fails.append(f"|g| {b['grad_norm']:.6f}→{r['grad_norm']:.6f}")
                print(f"  {name:<22s} FAIL  ({', '.join(fails)})")
                correctness_fail += 1
        else:
            print(f"  {name:<22s} NEW   (err={r['max_err']:.2e} |g|={r['grad_norm']:.6f})")
            correctness_pass += 1

    # Performance
    print(f"\nPerformance:")
    print(f"  {'Benchmark':<22s} {'Qubits':>6s} {'Params':>6s} {'Ops':>6s} {'Time(ms)':>10s}", end="")
    if baseline:
        print(f" {'Baseline':>10s} {'Change':>10s}", end="")
    print()
    print("  " + "-" * (52 + (22 if baseline else 0)))

    for name, _, _, _ in BENCHMARKS:
        r = results[name]
        line = f"  {name:<22s} {r['n_qubits']:>6d} {r['n_params']:>6d} {r['n_ops']:>6d} {r['time_ms']:>10.3f}"
        if name in baseline:
            b_time = baseline[name]["time_ms"]
            change = (r["time_ms"] - b_time) / b_time
            if change < -0.05:
                tag = f"{change:+.0%} FASTER"; perf_faster += 1
            elif change > PERF_FAIL:
                tag = f"{change:+.0%} REGR!"; perf_regression += 1
            elif change > PERF_WARN:
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
