"""Ready-made circuit factories for common quantum algorithms."""
from __future__ import annotations

import math
from ..ir import Circuit, Gate, Parameter
from ..analysis.observable import Observable, Z


def qft(n_qubits: int) -> Circuit:
    """Quantum Fourier Transform on n qubits."""
    c = Circuit(n_qubits)
    for i in range(n_qubits):
        c.h(i)
        for j in range(i + 1, n_qubits):
            c.cp(i, j, math.pi / 2 ** (j - i))
    # Reverse bit order
    for i in range(n_qubits // 2):
        c.swap(i, n_qubits - 1 - i)
    return c


def ghz(n_qubits: int) -> Circuit:
    """GHZ state preparation: |0...0⟩ → (|0...0⟩ + |1...1⟩)/√2."""
    c = Circuit(n_qubits)
    c.h(0)
    for i in range(n_qubits - 1):
        c.cx(i, i + 1)
    return c


def _mcz(c: Circuit, qubits: list[int]) -> None:
    """Multi-controlled Z using native Z/CZ/CCZ. Raises ValueError for n > 3."""
    n = len(qubits)
    if n == 1:
        c.z(qubits[0])
    elif n == 2:
        c.cz(qubits[0], qubits[1])
    elif n == 3:
        c.ccz(qubits[0], qubits[1], qubits[2])
    else:
        raise ValueError(f"grover_oracle supports at most 3 qubits, got {n}")


def grover_oracle(n_qubits: int, marked_states: list[int]) -> Circuit:
    """Phase oracle: flips sign of marked computational basis states (q0=MSB)."""
    if n_qubits > 3:
        raise ValueError(f"grover_oracle supports at most 3 qubits, got {n_qubits}")
    c = Circuit(n_qubits)
    qubits = list(range(n_qubits))
    for state in marked_states:
        # X-flip qubits where bit is 0 (MSB ordering: q0 is highest bit)
        for i in range(n_qubits):
            if not (state >> (n_qubits - 1 - i)) & 1:
                c.x(i)
        _mcz(c, qubits)
        for i in range(n_qubits):
            if not (state >> (n_qubits - 1 - i)) & 1:
                c.x(i)
    return c


def qaoa_mixer(graph: list[tuple[int, ...]], p: int = 1) -> Circuit:
    """QAOA circuit with cost (RZZ) and mixer (RX) layers."""
    n_qubits = max(v for e in graph for v in e[:2]) + 1
    c = Circuit(n_qubits)
    for q in range(n_qubits):
        c.h(q)
    for l in range(p):
        gamma = Parameter(f"gamma_{l}")
        for e in graph:
            i, j = e[0], e[1]
            c.rzz(i, j, gamma)
        beta = Parameter(f"beta_{l}")
        for q in range(n_qubits):
            c.rx(q, beta)
    return c


def maxcut_hamiltonian(edges: list[tuple[int, ...]]) -> Observable:
    """MaxCut QAOA cost Hamiltonian: C = Σ ½(1 - Z_i Z_j) for each edge."""
    H = Observable([])
    for e in edges:
        i, j, w = (e[0], e[1], e[2]) if len(e) == 3 else (e[0], e[1], 1.0)
        H = H + w * 0.5 * (Observable([(1.0, {})]) + -1.0 * (Z(i) @ Z(j)))
    return H
