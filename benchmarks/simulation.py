"""
Benchmark: tinyqubit vs Qiskit simulation speed.

Compares statevector simulation time.
Requires: pip install qiskit qiskit-aer

Run: python benchmarks/simulation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from math import pi

from tinyqubit.ir import Circuit
from tinyqubit.simulator import simulate

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


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

    # Bell state
    def make_bell_tq():
        return Circuit(2).h(0).cx(0, 1)
    def make_bell_qk():
        qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
        return qc
    tests.append(("bell_2", 2, make_bell_tq, make_bell_qk))

    # GHZ-10
    def make_ghz10_tq():
        c = Circuit(10).h(0)
        for i in range(9):
            c.cx(i, i+1)
        return c
    def make_ghz10_qk():
        qc = QuantumCircuit(10); qc.h(0)
        for i in range(9):
            qc.cx(i, i+1)
        return qc
    tests.append(("ghz_10", 10, make_ghz10_tq, make_ghz10_qk))

    # GHZ-15
    def make_ghz15_tq():
        c = Circuit(15).h(0)
        for i in range(14):
            c.cx(i, i+1)
        return c
    def make_ghz15_qk():
        qc = QuantumCircuit(15); qc.h(0)
        for i in range(14):
            qc.cx(i, i+1)
        return qc
    tests.append(("ghz_15", 15, make_ghz15_tq, make_ghz15_qk))

    # Random layers
    def make_random12_tq():
        c = Circuit(12)
        for _ in range(5):
            for i in range(12):
                c.h(i)
            for i in range(0, 11, 2):
                c.cx(i, i+1)
            for i in range(1, 11, 2):
                c.cx(i, i+1)
        return c
    def make_random12_qk():
        qc = QuantumCircuit(12)
        for _ in range(5):
            for i in range(12):
                qc.h(i)
            for i in range(0, 11, 2):
                qc.cx(i, i+1)
            for i in range(1, 11, 2):
                qc.cx(i, i+1)
        return qc
    tests.append(("layers_12", 12, make_random12_tq, make_random12_qk))

    # QFT-10
    def make_qft10_tq():
        c = Circuit(10)
        for i in range(10):
            c.h(i)
            for j in range(i+1, 10):
                c.cp(j, i, pi / (2 ** (j - i)))
        return c
    def make_qft10_qk():
        qc = QuantumCircuit(10)
        for i in range(10):
            qc.h(i)
            for j in range(i+1, 10):
                qc.cp(pi / (2 ** (j - i)), j, i)
        return qc
    tests.append(("qft_10", 10, make_qft10_tq, make_qft10_qk))

    # GHZ-20 (2^20 = 1M states)
    def make_ghz20_tq():
        c = Circuit(20).h(0)
        for i in range(19):
            c.cx(i, i+1)
        return c
    def make_ghz20_qk():
        qc = QuantumCircuit(20); qc.h(0)
        for i in range(19):
            qc.cx(i, i+1)
        return qc
    tests.append(("ghz_20", 20, make_ghz20_tq, make_ghz20_qk))

    # GHZ-23 (2^23 = 8M states)
    def make_ghz23_tq():
        c = Circuit(23).h(0)
        for i in range(22):
            c.cx(i, i+1)
        return c
    def make_ghz23_qk():
        qc = QuantumCircuit(23); qc.h(0)
        for i in range(22):
            qc.cx(i, i+1)
        return qc
    tests.append(("ghz_23", 23, make_ghz23_tq, make_ghz23_qk))

    # Layers-18 (deeper circuit, 2^18 = 262K states)
    def make_layers18_tq():
        c = Circuit(18)
        for _ in range(10):
            for i in range(18):
                c.h(i)
            for i in range(0, 17, 2):
                c.cx(i, i+1)
            for i in range(1, 17, 2):
                c.cx(i, i+1)
        return c
    def make_layers18_qk():
        qc = QuantumCircuit(18)
        for _ in range(10):
            for i in range(18):
                qc.h(i)
            for i in range(0, 17, 2):
                qc.cx(i, i+1)
            for i in range(1, 17, 2):
                qc.cx(i, i+1)
        return qc
    tests.append(("layers_18", 18, make_layers18_tq, make_layers18_qk))

    # QFT-16 (more gates)
    def make_qft16_tq():
        c = Circuit(16)
        for i in range(16):
            c.h(i)
            for j in range(i+1, 16):
                c.cp(j, i, pi / (2 ** (j - i)))
        return c
    def make_qft16_qk():
        qc = QuantumCircuit(16)
        for i in range(16):
            qc.h(i)
            for j in range(i+1, 16):
                qc.cp(pi / (2 ** (j - i)), j, i)
        return qc
    tests.append(("qft_16", 16, make_qft16_tq, make_qft16_qk))

    print("Simulation time in milliseconds (lower is better)")
    print("-" * 70)
    print(f"{'Circuit':<12} {'Qubits':>8} {'tinyqubit':>12} {'Qiskit':>12} {'Winner':>12}")
    print("-" * 70)

    for name, n_qubits, make_tq, make_qk in tests:
        tq_c = make_tq()
        tq_time = time_func(lambda: simulate(tq_c))

        if HAS_QISKIT:
            qk_c = make_qk()
            qk_time = time_func(lambda: Statevector(qk_c))
            if tq_time < qk_time: winner = "tinyqubit"
            elif tq_time > qk_time: winner = "Qiskit"
            else: winner = "tie"
        else:
            qk_time, winner = "-", "-"

        print(f"{name:<12} {n_qubits:>8} {tq_time:>10.2f}ms {qk_time:>10.2f}ms {winner:>12}" if HAS_QISKIT else f"{name:<12} {n_qubits:>8} {tq_time:>10.2f}ms {qk_time:>12} {winner:>12}")

    print("-" * 70)


if __name__ == "__main__":
    run_benchmark()
