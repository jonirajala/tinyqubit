"""
Core IR types - the single representation used throughout.

Contains:
    - Gate: Enum of supported gates (13 primitives)
    - Operation: Dataclass (gate, qubits, params)
    - Circuit: Lazy builder, just appends Operations
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass

class Gate(Enum):
    """12 primitive quantum gates."""
    # Pauli gates
    X = auto()
    Y = auto()
    Z = auto()

    # Single-qubit
    H = auto()
    S = auto()
    T = auto()

    # Rotations (parametric)
    RX = auto()
    RY = auto()
    RZ = auto()

    # Two-qubit
    CX = auto()
    CZ = auto()
    SWAP = auto()

    # Measurement
    MEASURE = auto()

    @property
    def n_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        if self in (Gate.CX, Gate.CZ, Gate.SWAP): return 2
        return 1

    @property
    def n_params(self) -> int:
        """Number of parameters (rotation angles)."""
        if self in (Gate.RX, Gate.RY, Gate.RZ): return 1
        return 0

@dataclass(frozen=True)
class Operation:
    gate: Gate
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


# Circuit - lazy list of operations
class Circuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.ops: list[Operation] = []

    def _add(self, gate: Gate, qubits: tuple, params: tuple = ()) -> Circuit:
        self.ops.append(Operation(gate, qubits, params))
        return self

    def x(self, q: int) -> Circuit: return self._add(Gate.X, (q,))
    def y(self, q: int) -> Circuit: return self._add(Gate.Y, (q,))
    def z(self, q: int) -> Circuit: return self._add(Gate.Z, (q,))
    def h(self, q: int) -> Circuit: return self._add(Gate.H, (q,))
    def s(self, q: int) -> Circuit: return self._add(Gate.S, (q,))
    def t(self, q: int) -> Circuit: return self._add(Gate.T, (q,))
    def rx(self, q: int, theta: float) -> Circuit: return self._add(Gate.RX, (q,), (theta,))
    def ry(self, q: int, theta: float) -> Circuit: return self._add(Gate.RY, (q,), (theta,))
    def rz(self, q: int, theta: float) -> Circuit: return self._add(Gate.RZ, (q,), (theta,))
    def cx(self, c: int, t: int) -> Circuit: return self._add(Gate.CX, (c, t))
    def cz(self, a: int, b: int) -> Circuit: return self._add(Gate.CZ, (a, b))
    def swap(self, a: int, b: int) -> Circuit: return self._add(Gate.SWAP, (a, b))
    def measure(self, q: int) -> Circuit: return self._add(Gate.MEASURE, (q,))


    # TODO: Move these into exports/qasm later when its more complicated and we have more exports
    def to_openqasm(self) -> str:
        """Export to OpenQASM 2.0"""
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{self.n_qubits}];']

        # has measurements, add creg
        if any(op.gate == Gate.MEASURE for op in self.ops):
            lines.append(f'creg c[{self.n_qubits}];')

        for op in self.ops:
            lines.append(self._op_to_qasm(op))

        return '\n'.join(lines)

    def _op_to_qasm(self, op: Operation) -> str:
        if op.gate == Gate.MEASURE:
            return f'measure q[{op.qubits[0]}] -> c[{op.qubits[0]}];'

        name = op.gate.name.lower()
        params = f'({op.params[0]})' if op.params else ''
        qubits = ', '.join(f'q[{q}]' for q in op.qubits)
        return f'{name}{params} {qubits};'

