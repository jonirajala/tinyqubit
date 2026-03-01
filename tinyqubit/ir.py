"""
Core IR types - the single representation used throughout.

Contains:
    - Gate: Enum of supported gates (22 primitives)
    - Operation: Dataclass (gate, qubits, params)
    - Circuit: Lazy builder, just appends Operations
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
import numpy as np


class Parameter:
    """Named symbolic parameter for variational circuits."""
    __slots__ = ('name', 'trainable')
    def __init__(self, name: str, trainable: bool = True):
        self.name = name
        self.trainable = trainable
    def __repr__(self): return f"Parameter({self.name!r})"
    def __eq__(self, other): return isinstance(other, Parameter) and self.name == other.name
    def __hash__(self): return hash(('Parameter', self.name))


def _has_parameter(params: tuple) -> bool:
    """Check if any param is a symbolic Parameter."""
    return any(isinstance(p, Parameter) for p in params)

class Gate(Enum):
    """22 primitive quantum gates."""
    # Pauli gates
    X = auto()
    Y = auto()
    Z = auto()

    # Single-qubit
    H = auto()
    S = auto()
    T = auto()
    SDG = auto()  # S-dagger (S†)
    TDG = auto()  # T-dagger (T†)

    # Rotations (parametric)
    RX = auto()
    RY = auto()
    RZ = auto()

    # Vendor-native single-qubit
    SX = auto()   # √X = RX(π/2) up to global phase

    # Two-qubit
    CX = auto()
    CZ = auto()
    CP = auto()  # Controlled phase
    SWAP = auto()
    ECR = auto()   # Echoed cross-resonance
    RZZ = auto()   # ZZ interaction

    # Three-qubit
    CCX = auto()   # Toffoli
    CCZ = auto()   # Controlled-controlled-Z

    # Measurement and reset
    MEASURE = auto()
    RESET = auto()  # Reset qubit to |0>

    @property
    def n_qubits(self) -> int:
        if self in (Gate.CX, Gate.CZ, Gate.CP, Gate.SWAP, Gate.ECR, Gate.RZZ): return 2
        if self in (Gate.CCX, Gate.CCZ): return 3
        return 1

    @property
    def n_params(self) -> int: return 1 if self in (Gate.RX, Gate.RY, Gate.RZ, Gate.CP, Gate.RZZ) else 0


@dataclass(frozen=True)
class Operation:
    gate: Gate
    qubits: tuple[int, ...]
    params: tuple[float | Parameter, ...] = ()
    classical_bit: int | None = None  # For MEASURE: which classical bit to store result
    condition: tuple[int, int] | None = None  # (classical_bit, expected_value) for conditional


_GATE_ADJOINT = {Gate.S: Gate.SDG, Gate.SDG: Gate.S, Gate.T: Gate.TDG, Gate.TDG: Gate.T}
_PARAM_GATES = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CP, Gate.RZZ})

# Context manager for conditional operations
class _ConditionalContext:
    """Context manager for c_if conditional blocks."""
    def __init__(self, circuit: "Circuit", classical_bit: int, value: int):
        self._circuit = circuit
        self._classical_bit = classical_bit
        self._value = value

    def __enter__(self):
        self._circuit._current_condition = (self._classical_bit, self._value)
        return self

    def __exit__(self, *args):
        self._circuit._current_condition = None


# Circuit - lazy list of operations
class Circuit:
    """Lazy circuit builder. Adds operations to a list."""

    def __init__(self, n_qubits: int, n_classical: int | None = None):
        self.n_qubits = n_qubits
        self.n_classical = n_classical if n_classical is not None else n_qubits
        self.ops: list[Operation] = []
        self._current_condition: tuple[int, int] | None = None  # For c_if context manager
        self._initial_state: np.ndarray | None = None

    def _add(self, gate: Gate, qubits: tuple, params: tuple = (),
             classical_bit: int | None = None) -> "Circuit":
        self.ops.append(Operation(gate, qubits, params, classical_bit, self._current_condition))
        return self

    def x(self, q: int) -> "Circuit": return self._add(Gate.X, (q,))
    def y(self, q: int) -> "Circuit": return self._add(Gate.Y, (q,))
    def z(self, q: int) -> "Circuit": return self._add(Gate.Z, (q,))
    def h(self, q: int) -> "Circuit": return self._add(Gate.H, (q,))
    def s(self, q: int) -> "Circuit": return self._add(Gate.S, (q,))
    def t(self, q: int) -> "Circuit": return self._add(Gate.T, (q,))
    def sdg(self, q: int) -> "Circuit": return self._add(Gate.SDG, (q,))
    def tdg(self, q: int) -> "Circuit": return self._add(Gate.TDG, (q,))
    def rx(self, q: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.RX, (q,), (theta,))
    def ry(self, q: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.RY, (q,), (theta,))
    def rz(self, q: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.RZ, (q,), (theta,))
    def cx(self, c: int, t: int) -> "Circuit": return self._add(Gate.CX, (c, t))
    def cz(self, a: int, b: int) -> "Circuit": return self._add(Gate.CZ, (min(a,b), max(a,b)))  # Canonicalize
    def cp(self, c: int, t: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.CP, (c, t), (theta,))
    def swap(self, a: int, b: int) -> "Circuit": return self._add(Gate.SWAP, (min(a,b), max(a,b)))  # Canonicalize
    def sx(self, q: int) -> "Circuit": return self._add(Gate.SX, (q,))
    def ecr(self, q0: int, q1: int) -> "Circuit": return self._add(Gate.ECR, (q0, q1))
    def rzz(self, q0: int, q1: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.RZZ, (q0, q1), (theta,))
    def ccx(self, c1: int, c2: int, t: int) -> "Circuit": return self._add(Gate.CCX, (c1, c2, t))
    def ccz(self, a: int, b: int, c: int) -> "Circuit": return self._add(Gate.CCZ, tuple(sorted([a, b, c])))  # Symmetric → canonicalize

    def measure(self, q: int, c: int | None = None) -> "Circuit":
        """Measure qubit q, store result in classical bit c (defaults to q)."""
        return self._add(Gate.MEASURE, (q,), (), classical_bit=c if c is not None else q)

    def reset(self, q: int) -> "Circuit":
        """Reset qubit to |0>."""
        return self._add(Gate.RESET, (q,))

    def c_if(self, classical_bit: int, value: int = 1) -> _ConditionalContext:
        """Context manager for conditional operations."""
        return _ConditionalContext(self, classical_bit, value)

    @property
    def parameters(self) -> set[Parameter]:
        """Return set of all unbound Parameters in the circuit."""
        return {p for op in self.ops for p in op.params if isinstance(p, Parameter)}

    @property
    def trainable_parameters(self) -> set[Parameter]:
        """Return set of trainable (non-feature) Parameters."""
        return {p for p in self.parameters if p.trainable}

    @property
    def is_parameterized(self) -> bool:
        """True if any operation has an unbound Parameter."""
        return any(_has_parameter(op.params) for op in self.ops)

    def bind(self, values: dict[str, float]) -> "Circuit":
        """Return new Circuit with Parameters substituted. Missing params left unbound."""
        c = Circuit(self.n_qubits, self.n_classical)
        c._initial_state = self._initial_state
        for op in self.ops:
            if _has_parameter(op.params):
                new_params = tuple(values[p.name] if isinstance(p, Parameter) and p.name in values else p
                                   for p in op.params)
                c.ops.append(Operation(op.gate, op.qubits, new_params, op.classical_bit, op.condition))
            else:
                c.ops.append(op)
        return c

    def bind_params(self, values: dict[str, float]) -> "Circuit":
        """In-place parameter rebinding. Caches param slots for fast repeated calls."""
        if not hasattr(self, '_param_slots'):
            self._param_slots = [(i, op) for i, op in enumerate(self.ops) if _has_parameter(op.params)]
        for i, tmpl in self._param_slots:
            new_params = tuple(values[p.name] if isinstance(p, Parameter) and p.name in values else p for p in tmpl.params)
            self.ops[i] = Operation(tmpl.gate, tmpl.qubits, new_params, tmpl.classical_bit, tmpl.condition)
        return self

    def _structure_key(self) -> tuple:
        """Hashable key capturing circuit structure, ignoring parameter values."""
        return (self.n_qubits, tuple((op.gate, op.qubits) for op in self.ops))

    def inverse(self) -> Circuit:
        """Return the adjoint (inverse) circuit."""
        if self._initial_state is not None:
            raise ValueError("inverse does not support initialized circuits")
        for op in self.ops:
            if op.gate in (Gate.MEASURE, Gate.RESET):
                raise ValueError(f"inverse does not support {op.gate.name}")
            if op.condition is not None:
                raise ValueError("inverse does not support conditional operations")
            if _has_parameter(op.params):
                raise ValueError("inverse does not support unbound Parameters")
        c = Circuit(self.n_qubits, self.n_classical)
        for op in reversed(self.ops):
            gate = _GATE_ADJOINT.get(op.gate, op.gate)
            params = tuple(-p for p in op.params) if op.gate in _PARAM_GATES else op.params
            c.ops.append(Operation(gate, op.qubits, params))
        return c

    def initialize(self, statevector) -> Circuit:
        """Set initial statevector (normalized automatically)."""
        sv = np.asarray(statevector, dtype=complex).ravel()
        if sv.shape[0] != 2 ** self.n_qubits:
            raise ValueError(f"Statevector size {sv.shape[0]} doesn't match {2 ** self.n_qubits}")
        self._initial_state = sv / np.linalg.norm(sv)
        return self

    def to_unitary(self) -> np.ndarray:
        from .simulator import to_unitary
        return to_unitary(self)

    def draw(self) -> None:                                                                                                                                                                                          
        if not self.ops:                                                                                                                                                                                             
            print("\n".join(f"q{q}: ──" for q in range(self.n_qubits))); return                                                                                                                                      
                                                                                                                                                                                                                    
        # Group ops by time step                                                                                                                                                                                     
        free, steps = {}, []                                                                                                                                                                                         
        for op in self.ops:                                                                                                                                                                                          
            t = max((free.get(q, 0) for q in op.qubits), default=0)                                                                                                                                                  
            for q in op.qubits: free[q] = t + 1                                                                                                                                                                      
            while len(steps) <= t: steps.append([])                                                                                                                                                                  
            steps[t].append(op)                                                                                                                                                                                      
                                                                                                                                                                                                                    
        # Build columns with gate symbols
        cols, ranges = [], []
        for ops_t in steps:
            col, rngs = {}, []
            for op in ops_t:
                syms = self._gate_syms(op)
                for q, s in syms.items():
                    if q not in col or col[q] == "│": col[q] = s
                if len(op.qubits) >= 2:
                    rngs.append((min(op.qubits), max(op.qubits)))
            cols.append(col)
            ranges.append(rngs)                                                                                                                                                                                       
                                                                                                                                                                                                                    
        widths = [max(2, max((len(v) for v in c.values() if v not in "│ "), default=2)) for c in cols]                                                                                                               
        pre = len(f"q{self.n_qubits - 1}: ")                                                                                                                                                                         
                                                                                                                                                                                                                    
        out = []                                                                                                                                                                                                     
        for q in range(self.n_qubits):                                                                                                                                                                               
            line = f"q{q}: ".ljust(pre)                                                                                                                                                                              
            for i, c in enumerate(cols):                                                                                                                                                                             
                w, s = widths[i], c.get(q)                                                                                                                                                                           
                if s is None: line += "─" * w                                                                                                                                                                        
                elif s == "│": line += "─" * (w // 2) + "│" + "─" * (w - w // 2 - 1)                                                                                                                                 
                else: line += "─" * ((w - len(s)) // 2) + s + "─" * (w - len(s) - (w - len(s)) // 2)                                                                                                                 
            out.append(line)                                                                                                                                                                                         
            if q < self.n_qubits - 1:
                conn = " " * pre + "".join(
                    " " * ((widths[i] - 1) // 2) + "│" + " " * (widths[i] - (widths[i] - 1) // 2 - 1)
                    if any(mn <= q < mx for mn, mx in ranges[i]) else " " * widths[i]
                    for i in range(len(cols)))
                if "│" in conn: out.append(conn)                                                                                                                                                                     
        print("\n".join(out))                                                                                                                                                                                        
                                                                                                                                                                                                                    
    def _gate_syms(self, op: Operation) -> dict[int, str]:
        g, qs = op.gate, op.qubits
        if g == Gate.MEASURE: return {qs[0]: "M"}
        if g == Gate.RESET: return {qs[0]: "R"}
        if g.n_qubits == 1: return {qs[0]: g.name}
        if g == Gate.CCX:
            col = {qs[0]: "●", qs[1]: "●", qs[2]: "X"}
            for q in range(min(qs) + 1, max(qs)):
                if q not in col: col[q] = "│"
            return col
        if g == Gate.CCZ:
            col = {qs[0]: "●", qs[1]: "●", qs[2]: "●"}
            for q in range(min(qs) + 1, max(qs)):
                if q not in col: col[q] = "│"
            return col
        _2q_syms = {Gate.CX: ("●", "X"), Gate.CZ: ("●", "●"), Gate.SWAP: ("╳", "╳"),
                    Gate.CP: ("●", "P"), Gate.ECR: ("ECR", "ECR"), Gate.RZZ: ("RZZ", "RZZ")}
        s0, s1 = _2q_syms.get(g, (g.name, g.name))
        col = {qs[0]: s0, qs[1]: s1}
        for q in range(min(qs) + 1, max(qs)): col[q] = "│"
        return col             