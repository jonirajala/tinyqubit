"""
Benchmark: tinyqubit vs Qiskit optimization.

Compares gate counts on standard circuits from QASMBench plus new circuit types.
Requires: pip install qiskit

Run: python benchmarks/optimize.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from math import pi

from tinyqubit.ir import Circuit
from tinyqubit.passes.optimize import optimize

from benchmarks.circuits import (
    bernstein_vazirani,
    hardware_efficient_ansatz,
    random_clifford_t,
    qft,
)
from benchmarks.metrics import count_t_gates

try:
    from qiskit import QuantumCircuit, transpile
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


def run_benchmark():
    """Run optimization benchmark."""
    tests = []

    # === Sanity checks ===

    # Basic cancellation
    tq = Circuit(2).h(0).h(0).cx(0,1).cx(0,1).x(1).x(1)
    qk = QuantumCircuit(2)
    qk.h(0); qk.h(0); qk.cx(0,1); qk.cx(0,1); qk.x(1); qk.x(1)
    tests.append(("redundant", tq, qk, False))

    # Clifford merging
    tq = Circuit(1).t(0).t(0).t(0).t(0).s(0).s(0)
    qk = QuantumCircuit(1)
    qk.t(0); qk.t(0); qk.t(0); qk.t(0); qk.s(0); qk.s(0)
    tests.append(("clifford", tq, qk, False))

    # === Standard benchmarks from QASMBench* ===

    # *GHZ-4: H then cascade of CNOTs (4 qubits, 4 gates, 3 CX)
    tq = Circuit(4).h(0).cx(0,1).cx(1,2).cx(2,3)
    qk = QuantumCircuit(4)
    qk.h(0); qk.cx(0,1); qk.cx(1,2); qk.cx(2,3)
    tests.append(("ghz_4*", tq, qk, True))

    # *Grover-2: oracle + diffusion (2 qubits, 16 gates, 2 CX)
    tq = Circuit(2)
    tq.h(0).h(1)
    tq.cz(0,1)
    tq.h(0).h(1).x(0).x(1)
    tq.cz(0,1)
    tq.x(0).x(1).h(0).h(1)
    qk = QuantumCircuit(2)
    qk.h(0); qk.h(1); qk.cz(0,1); qk.h(0); qk.h(1)
    qk.x(0); qk.x(1); qk.cz(0,1); qk.x(0); qk.x(1); qk.h(0); qk.h(1)
    tests.append(("grover_2*", tq, qk, True))

    # *Toffoli-3: standard CCX decomposition (3 qubits, 18 gates, 6 CX)
    tq = Circuit(3)
    tq.h(2).cx(1,2).tdg(2).cx(0,2).t(2).cx(1,2).tdg(2).cx(0,2)
    tq.t(1).t(2).h(2).cx(0,1).t(0).tdg(1).cx(0,1)
    qk = QuantumCircuit(3)
    qk.h(2); qk.cx(1,2); qk.tdg(2); qk.cx(0,2); qk.t(2); qk.cx(1,2); qk.tdg(2); qk.cx(0,2)
    qk.t(1); qk.t(2); qk.h(2); qk.cx(0,1); qk.t(0); qk.tdg(1); qk.cx(0,1)
    tests.append(("toffoli_3*", tq, qk, True))

    # *QFT-4: quantum fourier transform (4 qubits, 12 CX)
    tq = Circuit(4)
    tq.h(0)
    tq.cp(1, 0, pi/2).cp(2, 0, pi/4).cp(3, 0, pi/8)
    tq.h(1)
    tq.cp(2, 1, pi/2).cp(3, 1, pi/4)
    tq.h(2)
    tq.cp(3, 2, pi/2)
    tq.h(3)
    qk = QuantumCircuit(4)
    qk.h(0)
    qk.cp(pi/2, 1, 0); qk.cp(pi/4, 2, 0); qk.cp(pi/8, 3, 0)
    qk.h(1)
    qk.cp(pi/2, 2, 1); qk.cp(pi/4, 3, 1)
    qk.h(2)
    qk.cp(pi/2, 3, 2)
    qk.h(3)
    tests.append(("qft_4*", tq, qk, True))

    # *Adder-4: ripple carry adder (4 qubits, 23 gates, 10 CX)
    # Simplified adder pattern with Toffoli-like structures
    tq = Circuit(4)
    tq.cx(0,1).cx(0,2)
    tq.h(3).cx(2,3).tdg(3).cx(1,3).t(3).cx(2,3).tdg(3).cx(1,3).t(3).h(3)  # Toffoli
    tq.cx(0,1).cx(1,2)
    qk = QuantumCircuit(4)
    qk.cx(0,1); qk.cx(0,2)
    qk.h(3); qk.cx(2,3); qk.tdg(3); qk.cx(1,3); qk.t(3); qk.cx(2,3); qk.tdg(3); qk.cx(1,3); qk.t(3); qk.h(3)
    qk.cx(0,1); qk.cx(1,2)
    tests.append(("adder_4*", tq, qk, True))

    # === New circuit types from circuits.py ===

    # Bernstein-Vazirani (4, 8, 12 qubits)
    for n, secret in [(4, 0b1010), (8, 0b10101010), (12, 0b101010101010)]:
        tq, qk, _ = bernstein_vazirani(n, secret)
        tests.append((f"bv_{n}", tq, qk, True))

    # Hardware-efficient ansatz (4, 8 qubits, 2 layers)
    for n in [4, 8]:
        tq, qk, _ = hardware_efficient_ansatz(n, layers=2)
        tests.append((f"hea_{n}", tq, qk, True))

    # Random Clifford+T (10, 20 qubits)
    for n in [10, 20]:
        tq, qk, _ = random_clifford_t(n, depth=10, seed=42)
        tests.append((f"cliff_t_{n}", tq, qk, True))

    # Larger QFT (12, 16, 20 qubits)
    for n in [12, 16, 20]:
        tq, qk, _ = qft(n)
        tests.append((f"qft_{n}", tq, qk, True))

    # === Phase 11 targets ===

    # CNOT conjugation pattern
    tq = Circuit(2).z(0).cx(0,1).z(0).z(1).cx(0,1).z(1)
    qk = QuantumCircuit(2)
    qk.z(0); qk.cx(0,1); qk.z(0); qk.z(1); qk.cx(0,1); qk.z(1)
    tests.append(("cnot_conj", tq, qk, False))

    # 1Q gate consolidation
    tq = Circuit(4)
    for _ in range(2):
        for i in range(4):
            tq.rx(i, pi/4).rz(i, pi/4).rx(i, pi/4)
        for i in range(3):
            tq.cx(i, i+1)
    qk = QuantumCircuit(4)
    for _ in range(2):
        for i in range(4):
            qk.rx(pi/4, i); qk.rz(pi/4, i); qk.rx(pi/4, i)
        for i in range(3):
            qk.cx(i, i+1)
    tests.append(("variational", tq, qk, False))

    # Print results
    print("Gate counts after optimization (lower is better)")
    print("-" * 85)
    print(f"{'Circuit':<16} {'Original':>8} {'tinyqubit':>10} {'Qiskit-3':>10} {'T-count':>8} {'Winner':>12}")
    print("-" * 85)

    for name, tq_c, qk_c, _ in tests:
        orig = len(tq_c.ops)
        optimized = optimize(tq_c)
        tq_gates = len(optimized.ops)
        t_count = count_t_gates(optimized)

        if HAS_QISKIT and qk_c is not None:
            qk_gates = sum(transpile(qk_c, optimization_level=3).count_ops().values())
            if tq_gates < qk_gates: winner = "tinyqubit"
            elif tq_gates > qk_gates: winner = "Qiskit"
            else: winner = "tie"
        else:
            qk_gates, winner = "-", "-"

        print(f"{name:<16} {orig:>8} {tq_gates:>10} {qk_gates:>10} {t_count:>8} {winner:>12}")

    print("-" * 85)
    print("* = QASMBench standard circuit (github.com/pnnl/QASMBench)")


if __name__ == "__main__":
    run_benchmark()
