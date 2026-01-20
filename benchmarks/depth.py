"""
Benchmark: tinyqubit vs Qiskit circuit depth.

Compares circuit depth after optimization.
Requires: pip install qiskit

Run: python benchmarks/depth.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from math import pi

from tinyqubit.ir import Circuit, Gate
from tinyqubit.passes.optimize import optimize

try:
    from qiskit import QuantumCircuit, transpile
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


def calc_depth(circuit):
    """Calculate circuit depth."""
    free = {}
    for op in circuit.ops:
        t = max((free.get(q, 0) for q in op.qubits), default=0) + 1
        for q in op.qubits:
            free[q] = t
    return max(free.values(), default=0)


def run_benchmark():
    tests = []

    # *GHZ-4
    tq = Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    qk = QuantumCircuit(4)
    qk.h(0); qk.cx(0, 1); qk.cx(1, 2); qk.cx(2, 3)
    tests.append(("ghz_4*", tq, qk))

    # *Grover-2
    tq = Circuit(2)
    tq.h(0).h(1).cz(0, 1).h(0).h(1).x(0).x(1).cz(0, 1).x(0).x(1).h(0).h(1)
    qk = QuantumCircuit(2)
    qk.h(0); qk.h(1); qk.cz(0, 1); qk.h(0); qk.h(1)
    qk.x(0); qk.x(1); qk.cz(0, 1); qk.x(0); qk.x(1); qk.h(0); qk.h(1)
    tests.append(("grover_2*", tq, qk))

    # *Toffoli-3
    tq = Circuit(3)
    tq.h(2).cx(1, 2).tdg(2).cx(0, 2).t(2).cx(1, 2).tdg(2).cx(0, 2)
    tq.t(1).t(2).h(2).cx(0, 1).t(0).tdg(1).cx(0, 1)
    qk = QuantumCircuit(3)
    qk.h(2); qk.cx(1, 2); qk.tdg(2); qk.cx(0, 2); qk.t(2); qk.cx(1, 2); qk.tdg(2); qk.cx(0, 2)
    qk.t(1); qk.t(2); qk.h(2); qk.cx(0, 1); qk.t(0); qk.tdg(1); qk.cx(0, 1)
    tests.append(("toffoli_3*", tq, qk))

    # *QFT-4
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
    tests.append(("qft_4*", tq, qk))

    # Deep serial circuit
    tq = Circuit(1)
    for _ in range(20):
        tq.h(0).t(0).s(0)
    qk = QuantumCircuit(1)
    for _ in range(20):
        qk.h(0); qk.t(0); qk.s(0)
    tests.append(("serial_60", tq, qk))

    # Wide parallel circuit
    tq = Circuit(8)
    for i in range(8):
        tq.h(i).t(i)
    qk = QuantumCircuit(8)
    for i in range(8):
        qk.h(i); qk.t(i)
    tests.append(("parallel_8", tq, qk))

    print("Circuit depth after optimization (lower is better)")
    print("-" * 70)
    print(f"{'Circuit':<14} {'Original':>10} {'tinyqubit':>12} {'Qiskit-3':>12} {'Winner':>12}")
    print("-" * 70)

    for name, tq_c, qk_c in tests:
        orig = calc_depth(tq_c)
        tq_depth = calc_depth(optimize(tq_c))

        if HAS_QISKIT:
            qk_depth = transpile(qk_c, optimization_level=3).depth()
            if tq_depth < qk_depth: winner = "tinyqubit"
            elif tq_depth > qk_depth: winner = "Qiskit"
            else: winner = "tie"
        else:
            qk_depth, winner = "-", "-"

        print(f"{name:<14} {orig:>10} {tq_depth:>12} {qk_depth:>12} {winner:>12}")

    print("-" * 70)
    print("* = QASMBench standard circuit")


if __name__ == "__main__":
    run_benchmark()
