"""
Shared circuit library for benchmarks.

Provides standard circuits in tinyqubit, Qiskit, and PennyLane formats:
- bernstein_vazirani: Oracle-based algorithm
- hardware_efficient_ansatz: VQE-style variational circuit
- random_clifford_t: Random Clifford+T circuits
- qft: Quantum Fourier Transform
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from math import pi
import random
from typing import Callable

from tinyqubit.ir import Circuit

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def bernstein_vazirani(n: int, secret: int) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Bernstein-Vazirani circuit: finds secret string via single query.

    Args:
        n: Number of qubits (excluding ancilla)
        secret: Secret bit string as integer (must be < 2^n)

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)

    Circuit structure: H^n -> Oracle(secret) -> H^n
    Oracle applies CX from qubit i to ancilla where secret bit i is 1.
    """
    total_qubits = n + 1  # n data qubits + 1 ancilla
    ancilla = n

    # TinyQubit circuit
    tq = Circuit(total_qubits)

    # Initialize ancilla to |-> state
    tq.x(ancilla).h(ancilla)

    # Apply H to all data qubits
    for i in range(n):
        tq.h(i)

    # Oracle: CX for each bit of secret that is 1
    for i in range(n):
        if (secret >> i) & 1:
            tq.cx(i, ancilla)

    # Apply H to all data qubits
    for i in range(n):
        tq.h(i)

    # Qiskit circuit
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(total_qubits)
        qk.x(ancilla)
        qk.h(ancilla)
        for i in range(n):
            qk.h(i)
        for i in range(n):
            if (secret >> i) & 1:
                qk.cx(i, ancilla)
        for i in range(n):
            qk.h(i)

    # PennyLane ops function
    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            qml.PauliX(wires=ancilla)
            qml.Hadamard(wires=ancilla)
            for i in range(n):
                qml.Hadamard(wires=i)
            for i in range(n):
                if (secret >> i) & 1:
                    qml.CNOT(wires=[i, ancilla])
            for i in range(n):
                qml.Hadamard(wires=i)

    return tq, qk, pl_ops


def hardware_efficient_ansatz(n_qubits: int, layers: int, params: list[float] | None = None) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Hardware-efficient ansatz (VQE-style): RY-RZ layer + linear entangling layer.

    Args:
        n_qubits: Number of qubits
        layers: Number of ansatz layers
        params: Optional list of rotation angles. If None, uses pi/4 for all.

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)

    Each layer has:
    - RY(theta) on each qubit
    - RZ(phi) on each qubit
    - CX chain: CX(0,1), CX(1,2), ..., CX(n-2,n-1)
    """
    params_per_layer = 2 * n_qubits  # RY + RZ for each qubit
    total_params = layers * params_per_layer

    if params is None:
        params = [pi / 4] * total_params
    elif len(params) != total_params:
        raise ValueError(f"Expected {total_params} params, got {len(params)}")

    # TinyQubit circuit
    tq = Circuit(n_qubits)
    param_idx = 0

    for _ in range(layers):
        # RY layer
        for q in range(n_qubits):
            tq.ry(q, params[param_idx])
            param_idx += 1
        # RZ layer
        for q in range(n_qubits):
            tq.rz(q, params[param_idx])
            param_idx += 1
        # Entangling layer
        for q in range(n_qubits - 1):
            tq.cx(q, q + 1)

    # Qiskit circuit
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n_qubits)
        param_idx = 0
        for _ in range(layers):
            for q in range(n_qubits):
                qk.ry(params[param_idx], q)
                param_idx += 1
            for q in range(n_qubits):
                qk.rz(params[param_idx], q)
                param_idx += 1
            for q in range(n_qubits - 1):
                qk.cx(q, q + 1)

    # PennyLane ops function
    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            param_idx = 0
            for _ in range(layers):
                for q in range(n_qubits):
                    qml.RY(params[param_idx], wires=q)
                    param_idx += 1
                for q in range(n_qubits):
                    qml.RZ(params[param_idx], wires=q)
                    param_idx += 1
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

    return tq, qk, pl_ops


def random_clifford_t(n_qubits: int, depth: int, seed: int = 42) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Random Clifford+T circuit for stress testing.

    Args:
        n_qubits: Number of qubits
        depth: Number of gate layers
        seed: Random seed for reproducibility

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)

    Each layer randomly applies:
    - Single-qubit: H, S, T with equal probability
    - Two-qubit: CX on random adjacent pairs
    """
    rng = random.Random(seed)

    # TinyQubit circuit
    tq = Circuit(n_qubits)

    # Track gates for Qiskit and PennyLane
    gate_log = []

    for _ in range(depth):
        # Single-qubit layer: random H/S/T on each qubit
        for q in range(n_qubits):
            gate = rng.choice(['h', 's', 't'])
            gate_log.append((gate, q))
            if gate == 'h':
                tq.h(q)
            elif gate == 's':
                tq.s(q)
            else:
                tq.t(q)

        # Two-qubit layer: random CX on some adjacent pairs
        for q in range(0, n_qubits - 1, 2):
            if rng.random() < 0.5:
                tq.cx(q, q + 1)
                gate_log.append(('cx', q, q + 1))

        # Offset CX layer for better coverage
        for q in range(1, n_qubits - 1, 2):
            if rng.random() < 0.5:
                tq.cx(q, q + 1)
                gate_log.append(('cx', q, q + 1))

    # Qiskit circuit
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n_qubits)
        for entry in gate_log:
            if entry[0] == 'h':
                qk.h(entry[1])
            elif entry[0] == 's':
                qk.s(entry[1])
            elif entry[0] == 't':
                qk.t(entry[1])
            elif entry[0] == 'cx':
                qk.cx(entry[1], entry[2])

    # PennyLane ops function
    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            for entry in gate_log:
                if entry[0] == 'h':
                    qml.Hadamard(wires=entry[1])
                elif entry[0] == 's':
                    qml.S(wires=entry[1])
                elif entry[0] == 't':
                    qml.T(wires=entry[1])
                elif entry[0] == 'cx':
                    qml.CNOT(wires=[entry[1], entry[2]])

    return tq, qk, pl_ops


def qft(n: int) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Quantum Fourier Transform on n qubits.

    Args:
        n: Number of qubits

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)

    Standard QFT with controlled phase rotations.
    """
    tq = Circuit(n)

    for i in range(n):
        tq.h(i)
        for j in range(i + 1, n):
            angle = pi / (2 ** (j - i))
            tq.cp(j, i, angle)

    # Qiskit circuit
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n)
        for i in range(n):
            qk.h(i)
            for j in range(i + 1, n):
                angle = pi / (2 ** (j - i))
                qk.cp(angle, j, i)

    # PennyLane ops function
    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            for i in range(n):
                qml.Hadamard(wires=i)
                for j in range(i + 1, n):
                    angle = pi / (2 ** (j - i))
                    qml.ControlledPhaseShift(angle, wires=[j, i])

    return tq, qk, pl_ops


def ghz(n: int) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    GHZ state preparation: |000...0> + |111...1>.

    Args:
        n: Number of qubits

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)
    """
    tq = Circuit(n)
    tq.h(0)
    for i in range(n - 1):
        tq.cx(i, i + 1)

    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n)
        qk.h(0)
        for i in range(n - 1):
            qk.cx(i, i + 1)

    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            qml.Hadamard(wires=0)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])

    return tq, qk, pl_ops


def grover_3qubit() -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    3-qubit Grover search for |111⟩ (one iteration).

    Uses native CCZ gate for oracle and CCX in diffusion.
    This benchmarks 3-qubit gate decomposition and optimization.

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)
    """
    tq = Circuit(3)
    # Superposition
    for i in range(3):
        tq.h(i)
    # Oracle: CCZ marks |111⟩
    tq.ccz(0, 1, 2)
    # Diffusion: H X CCZ X H
    for i in range(3):
        tq.h(i)
    for i in range(3):
        tq.x(i)
    tq.ccz(0, 1, 2)
    for i in range(3):
        tq.x(i)
    for i in range(3):
        tq.h(i)

    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(3)
        for i in range(3):
            qk.h(i)
        qk.ccz(0, 1, 2)
        for i in range(3):
            qk.h(i)
        for i in range(3):
            qk.x(i)
        qk.ccz(0, 1, 2)
        for i in range(3):
            qk.x(i)
        for i in range(3):
            qk.h(i)

    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            for i in range(3):
                qml.Hadamard(wires=i)
            qml.CCZ(wires=[0, 1, 2])
            for i in range(3):
                qml.Hadamard(wires=i)
            for i in range(3):
                qml.PauliX(wires=i)
            qml.CCZ(wires=[0, 1, 2])
            for i in range(3):
                qml.PauliX(wires=i)
            for i in range(3):
                qml.Hadamard(wires=i)

    return tq, qk, pl_ops


def toffoli_chain(n: int) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Chain of Toffoli (CCX) gates: CCX(0,1,2), CCX(1,2,3), ..., CCX(n-3,n-2,n-1).

    Benchmarks 3-qubit gate decomposition scaling.

    Args:
        n: Number of qubits (must be >= 3)

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)
    """
    assert n >= 3
    tq = Circuit(n)
    for i in range(n - 2):
        tq.ccx(i, i + 1, i + 2)

    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n)
        for i in range(n - 2):
            qk.ccx(i, i + 1, i + 2)

    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            for i in range(n - 2):
                qml.Toffoli(wires=[i, i + 1, i + 2])

    return tq, qk, pl_ops


def grover_diffusion(n: int) -> tuple[Circuit, "QuantumCircuit | None", "Callable | None"]:
    """
    Grover diffusion operator (without oracle).

    Args:
        n: Number of qubits

    Returns:
        (tinyqubit Circuit, Qiskit QuantumCircuit or None, PennyLane ops function or None)

    Structure: H^n -> X^n -> MCZ -> X^n -> H^n
    Uses decomposed multi-controlled Z.
    """
    tq = Circuit(n)

    # H^n
    for i in range(n):
        tq.h(i)

    # X^n
    for i in range(n):
        tq.x(i)

    # Multi-controlled Z (decomposed as cascade of CZ)
    # For n qubits, use CZ between pairs
    for i in range(n - 1):
        tq.cz(i, i + 1)

    # X^n
    for i in range(n):
        tq.x(i)

    # H^n
    for i in range(n):
        tq.h(i)

    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n)
        for i in range(n):
            qk.h(i)
        for i in range(n):
            qk.x(i)
        for i in range(n - 1):
            qk.cz(i, i + 1)
        for i in range(n):
            qk.x(i)
        for i in range(n):
            qk.h(i)

    pl_ops = None
    if HAS_PENNYLANE:
        def pl_ops():
            for i in range(n):
                qml.Hadamard(wires=i)
            for i in range(n):
                qml.PauliX(wires=i)
            for i in range(n - 1):
                qml.CZ(wires=[i, i + 1])
            for i in range(n):
                qml.PauliX(wires=i)
            for i in range(n):
                qml.Hadamard(wires=i)

    return tq, qk, pl_ops
