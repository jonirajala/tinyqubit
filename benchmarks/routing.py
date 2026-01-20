"""
Benchmark: tinyqubit vs Qiskit routing.

Compares 2Q gate counts after routing for line topology.
Requires: pip install qiskit

Run: python benchmarks/routing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit, Gate
from tinyqubit.target import Target
from tinyqubit.passes.route import route

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


def line_topology(n):
    return frozenset((i, i+1) for i in range(n-1)) | frozenset((i+1, i) for i in range(n-1))


def count_2q(circuit):
    return sum(1 for op in circuit.ops if op.gate in {Gate.CX, Gate.CZ, Gate.SWAP})


def run_benchmark():
    tests = []

    # CX between non-adjacent qubits
    tq = Circuit(5).h(0).cx(0, 4)
    qk = QuantumCircuit(5); qk.h(0); qk.cx(0, 4)
    tests.append(("cx_0_4", tq, qk, 1))

    # Multiple non-adjacent CX
    tq = Circuit(5).cx(0, 4).cx(1, 3).cx(0, 2)
    qk = QuantumCircuit(5); qk.cx(0, 4); qk.cx(1, 3); qk.cx(0, 2)
    tests.append(("multi_cx", tq, qk, 3))

    # GHZ on line
    tq = Circuit(5).h(0).cx(0, 1).cx(0, 2).cx(0, 3).cx(0, 4)
    qk = QuantumCircuit(5); qk.h(0); qk.cx(0, 1); qk.cx(0, 2); qk.cx(0, 3); qk.cx(0, 4)
    tests.append(("ghz_5", tq, qk, 4))

    # QFT-like pattern
    tq = Circuit(4).h(0).cx(0, 1).cx(0, 2).cx(0, 3).h(1).cx(1, 2).cx(1, 3)
    qk = QuantumCircuit(4); qk.h(0); qk.cx(0, 1); qk.cx(0, 2); qk.cx(0, 3); qk.h(1); qk.cx(1, 2); qk.cx(1, 3)
    tests.append(("qft_like", tq, qk, 6))

    target = Target(n_qubits=5, edges=line_topology(5), basis_gates=frozenset({Gate.CX, Gate.H, Gate.SWAP}), name="line_5")
    coupling = CouplingMap.from_line(5) if HAS_QISKIT else None

    print("2Q gate counts after routing for line topology (lower is better)")
    print("-" * 65)
    print(f"{'Circuit':<12} {'Original':>10} {'tinyqubit':>12} {'Qiskit-3':>12} {'Winner':>12}")
    print("-" * 65)

    for name, tq_c, qk_c, orig_2q in tests:
        routed = route(tq_c, target)
        tq_2q = count_2q(routed)

        if HAS_QISKIT:
            qk_routed = transpile(qk_c, coupling_map=coupling, optimization_level=3)
            qk_2q = qk_routed.count_ops().get('cx', 0) + qk_routed.count_ops().get('swap', 0) * 3
            if tq_2q < qk_2q: winner = "tinyqubit"
            elif tq_2q > qk_2q: winner = "Qiskit"
            else: winner = "tie"
        else:
            qk_2q, winner = "-", "-"

        print(f"{name:<12} {orig_2q:>10} {tq_2q:>12} {qk_2q:>12} {winner:>12}")

    print("-" * 65)


if __name__ == "__main__":
    run_benchmark()
