"""
Benchmark: tinyqubit vs Qiskit compile time.

Compares transpilation speed.
Requires: pip install qiskit

Run: python benchmarks/compile_time.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from math import pi

from tinyqubit.ir import Circuit, Gate
from tinyqubit.target import Target
from tinyqubit.compile import transpile as tq_transpile

try:
    from qiskit import QuantumCircuit, transpile as qk_transpile
    from qiskit.transpiler import CouplingMap
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


def line_topology(n):
    return frozenset((i, i+1) for i in range(n-1)) | frozenset((i+1, i) for i in range(n-1))


def time_func(func, runs=5):
    """Time a function, return average ms."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


def run_benchmark():
    tests = []

    # Small: Bell state
    def make_bell_tq():
        return Circuit(2).h(0).cx(0, 1)
    def make_bell_qk():
        qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
        return qc
    tests.append(("bell_2", 2, make_bell_tq, make_bell_qk))

    # Medium: GHZ-10
    def make_ghz_tq():
        c = Circuit(10).h(0)
        for i in range(9):
            c.cx(i, i+1)
        return c
    def make_ghz_qk():
        qc = QuantumCircuit(10); qc.h(0)
        for i in range(9):
            qc.cx(i, i+1)
        return qc
    tests.append(("ghz_10", 10, make_ghz_tq, make_ghz_qk))

    # Medium: QFT-8
    def make_qft_tq():
        c = Circuit(8)
        for i in range(8):
            c.h(i)
            for j in range(i+1, 8):
                c.cp(j, i, pi / (2 ** (j - i)))
        return c
    def make_qft_qk():
        qc = QuantumCircuit(8)
        for i in range(8):
            qc.h(i)
            for j in range(i+1, 8):
                qc.cp(pi / (2 ** (j - i)), j, i)
        return qc
    tests.append(("qft_8", 8, make_qft_tq, make_qft_qk))

    # Medium: Toffoli chain (3Q gates need decomposition)
    def make_toffoli_tq():
        c = Circuit(10)
        for i in range(8):
            c.ccx(i, i+1, i+2)
        return c
    def make_toffoli_qk():
        qc = QuantumCircuit(10)
        for i in range(8):
            qc.ccx(i, i+1, i+2)
        return qc
    tests.append(("toffoli_10", 10, make_toffoli_tq, make_toffoli_qk))

    # Large: Random-like pattern
    def make_random_tq():
        c = Circuit(20)
        for layer in range(10):
            for i in range(20):
                c.h(i)
            for i in range(0, 19, 2):
                c.cx(i, i+1)
            for i in range(1, 19, 2):
                c.cx(i, i+1)
        return c
    def make_random_qk():
        qc = QuantumCircuit(20)
        for layer in range(10):
            for i in range(20):
                qc.h(i)
            for i in range(0, 19, 2):
                qc.cx(i, i+1)
            for i in range(1, 19, 2):
                qc.cx(i, i+1)
        return qc
    tests.append(("layers_20", 20, make_random_tq, make_random_qk))

    print("Compile time in milliseconds (lower is better)")
    print("-" * 70)
    print(f"{'Circuit':<12} {'Qubits':>8} {'tinyqubit':>12} {'Qiskit-3':>12} {'Winner':>12}")
    print("-" * 70)

    for name, n_qubits, make_tq, make_qk in tests:
        target = Target(
            n_qubits=n_qubits,
            edges=line_topology(n_qubits),
            basis_gates=frozenset({Gate.CX, Gate.H, Gate.RZ, Gate.SWAP}),
            name=f"line_{n_qubits}"
        )
        coupling = CouplingMap.from_line(n_qubits) if HAS_QISKIT else None

        tq_time = time_func(lambda: tq_transpile(make_tq(), target))

        if HAS_QISKIT:
            qk_time = time_func(lambda: qk_transpile(make_qk(), coupling_map=coupling, optimization_level=3))
            if tq_time < qk_time: winner = "tinyqubit"
            elif tq_time > qk_time: winner = "Qiskit"
            else: winner = "tie"
        else:
            qk_time, winner = "-", "-"

        print(f"{name:<12} {n_qubits:>8} {tq_time:>10.1f}ms {qk_time:>10.1f}ms {winner:>12}" if HAS_QISKIT else f"{name:<12} {n_qubits:>8} {tq_time:>10.1f}ms {qk_time:>12} {winner:>12}")

    print("-" * 70)


if __name__ == "__main__":
    run_benchmark()
