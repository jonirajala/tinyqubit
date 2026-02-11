"""
Benchmark: tinyqubit vs Qiskit vs PennyLane simulation speed.

Compares statevector simulation time.
Requires: pip install qiskit qiskit-aer pennylane

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

try:
    import pennylane as qml
    import numpy as np
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("PennyLane not installed. Install with: pip install pennylane\n")


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
    def make_bell_pl():
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()
        return circuit
    tests.append(("bell_2", 2, make_bell_tq, make_bell_qk, make_bell_pl))

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
    def make_ghz10_pl():
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(9):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("ghz_10", 10, make_ghz10_tq, make_ghz10_qk, make_ghz10_pl))

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
    def make_ghz15_pl():
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(14):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("ghz_15", 15, make_ghz15_tq, make_ghz15_qk, make_ghz15_pl))

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
    def make_random12_pl():
        def circuit():
            for _ in range(5):
                for i in range(12):
                    qml.Hadamard(wires=i)
                for i in range(0, 11, 2):
                    qml.CNOT(wires=[i, i+1])
                for i in range(1, 11, 2):
                    qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("layers_12", 12, make_random12_tq, make_random12_qk, make_random12_pl))

    # Toffoli chain (3Q gates)
    def make_toffoli8_tq():
        c = Circuit(8)
        for i in range(6):
            c.ccx(i, i+1, i+2)
        return c
    def make_toffoli8_qk():
        qc = QuantumCircuit(8)
        for i in range(6):
            qc.ccx(i, i+1, i+2)
        return qc
    def make_toffoli8_pl():
        def circuit():
            for i in range(6):
                qml.Toffoli(wires=[i, i+1, i+2])
            return qml.state()
        return circuit
    tests.append(("toffoli_8", 8, make_toffoli8_tq, make_toffoli8_qk, make_toffoli8_pl))

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
    def make_qft10_pl():
        def circuit():
            for i in range(10):
                qml.Hadamard(wires=i)
                for j in range(i+1, 10):
                    qml.ControlledPhaseShift(pi / (2 ** (j - i)), wires=[j, i])
            return qml.state()
        return circuit
    tests.append(("qft_10", 10, make_qft10_tq, make_qft10_qk, make_qft10_pl))

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
    def make_ghz20_pl():
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(19):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("ghz_20", 20, make_ghz20_tq, make_ghz20_qk, make_ghz20_pl))

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
    def make_ghz23_pl():
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(22):
                qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("ghz_23", 23, make_ghz23_tq, make_ghz23_qk, make_ghz23_pl))

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
    def make_layers18_pl():
        def circuit():
            for _ in range(10):
                for i in range(18):
                    qml.Hadamard(wires=i)
                for i in range(0, 17, 2):
                    qml.CNOT(wires=[i, i+1])
                for i in range(1, 17, 2):
                    qml.CNOT(wires=[i, i+1])
            return qml.state()
        return circuit
    tests.append(("layers_18", 18, make_layers18_tq, make_layers18_qk, make_layers18_pl))

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
    def make_qft16_pl():
        def circuit():
            for i in range(16):
                qml.Hadamard(wires=i)
                for j in range(i+1, 16):
                    qml.ControlledPhaseShift(pi / (2 ** (j - i)), wires=[j, i])
            return qml.state()
        return circuit
    tests.append(("qft_16", 16, make_qft16_tq, make_qft16_qk, make_qft16_pl))

    print("Simulation time in milliseconds (lower is better)")
    print("-" * 90)
    print(f"{'Circuit':<12} {'Qubits':>8} {'tinyqubit':>12} {'Qiskit':>12} {'PennyLane':>12} {'Winner':>12}")
    print("-" * 90)

    for name, n_qubits, make_tq, make_qk, make_pl in tests:
        tq_c = make_tq()
        tq_time = time_func(lambda: simulate(tq_c))

        qk_time = None
        if HAS_QISKIT:
            qk_c = make_qk()
            qk_time = time_func(lambda: Statevector(qk_c))

        pl_time = None
        if HAS_PENNYLANE:
            pl_circuit = make_pl()
            dev = qml.device('default.qubit', wires=n_qubits)
            qnode = qml.QNode(pl_circuit, dev)
            # Warmup
            _ = qnode()
            pl_time = time_func(lambda: qnode())

        # Determine winner
        times = {"tinyqubit": tq_time}
        if qk_time is not None:
            times["Qiskit"] = qk_time
        if pl_time is not None:
            times["PennyLane"] = pl_time

        winner = min(times, key=times.get)

        # Format output
        qk_str = f"{qk_time:>10.2f}ms" if qk_time is not None else f"{'-':>12}"
        pl_str = f"{pl_time:>10.2f}ms" if pl_time is not None else f"{'-':>12}"

        print(f"{name:<12} {n_qubits:>8} {tq_time:>10.2f}ms {qk_str} {pl_str} {winner:>12}")

    print("-" * 90)


if __name__ == "__main__":
    run_benchmark()
