"""Standard parameterized ansatz templates for variational circuits."""
from __future__ import annotations
from .ir import Circuit, Parameter


def strongly_entangling_layers(circuit: Circuit, n_layers: int, wires: list[int] | None = None, prefix: str = "sel") -> None:
    """RY + RZ per qubit, CX with layer-dependent offset for full connectivity."""
    wires = wires or list(range(circuit.n_qubits))
    n = len(wires)
    for l in range(n_layers):
        for i, w in enumerate(wires):
            circuit.ry(w, Parameter(f"{prefix}_{l}_{i}_y"))
            circuit.rz(w, Parameter(f"{prefix}_{l}_{i}_z"))
        for i in range(n):
            circuit.cx(wires[i], wires[(i + l + 1) % n])


def basic_entangler_layers(circuit: Circuit, n_layers: int, wires: list[int] | None = None, prefix: str = "bel") -> None:
    """RY only, linear CX ladder."""
    wires = wires or list(range(circuit.n_qubits))
    for l in range(n_layers):
        for i, w in enumerate(wires):
            circuit.ry(w, Parameter(f"{prefix}_{l}_{i}"))
        for i in range(len(wires) - 1):
            circuit.cx(wires[i], wires[i + 1])
