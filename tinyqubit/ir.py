"""Core IR types: Gate, Operation, Circuit."""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
from math import sqrt, cos, sin, pi as _pi


class Parameter:
    """Named symbolic parameter for variational circuits."""
    __slots__ = ('name', 'trainable')
    def __init__(self, name: str, trainable: bool = True):
        self.name = name
        self.trainable = trainable
    def __repr__(self): return f"Parameter({self.name!r})"
    def __eq__(self, other): return isinstance(other, Parameter) and self.name == other.name
    def __hash__(self): return hash(('Parameter', self.name))
    def __mul__(self, other): return ScaledParam(other, self)
    def __rmul__(self, other): return ScaledParam(other, self)


class ScaledParam:
    """scale * Parameter — resolved at bind time."""
    __slots__ = ('scale', 'param')
    def __init__(self, scale: float, param: Parameter):
        self.scale, self.param = scale, param
    @property
    def name(self): return self.param.name
    @property
    def trainable(self): return self.param.trainable


def _is_param(p) -> bool: return isinstance(p, (Parameter, ScaledParam))
def _has_parameter(params: tuple) -> bool: return any(_is_param(p) for p in params)
def _resolve(p, values: dict):
    if isinstance(p, ScaledParam) and p.param.name in values: return p.scale * values[p.param.name]
    if isinstance(p, Parameter) and p.name in values: return values[p.name]
    return p

class Gate(Enum):
    """24 primitive quantum gates."""
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
    SEXC = auto()  # Single excitation (Givens rotation in |01⟩,|10⟩ subspace)
    DEXC = auto()  # Double excitation (Givens rotation in |0011⟩,|1100⟩ subspace)

    # Three-qubit
    CCX = auto()   # Toffoli
    CCZ = auto()   # Controlled-controlled-Z

    # Measurement and reset
    MEASURE = auto()
    RESET = auto()  # Reset qubit to |0>

    @property
    def n_qubits(self) -> int:
        if self in (Gate.CX, Gate.CZ, Gate.CP, Gate.SWAP, Gate.ECR, Gate.RZZ, Gate.SEXC): return 2
        if self in (Gate.CCX, Gate.CCZ): return 3
        if self == Gate.DEXC: return 4
        return 1

    @property
    def n_params(self) -> int: return 1 if self in (Gate.RX, Gate.RY, Gate.RZ, Gate.CP, Gate.RZZ, Gate.SEXC, Gate.DEXC) else 0


@dataclass(frozen=True)
class Operation:
    gate: Gate
    qubits: tuple[int, ...]
    params: tuple[float | Parameter, ...] = ()
    classical_bit: int | None = None  # For MEASURE: which classical bit to store result
    condition: tuple[int, int] | None = None  # (classical_bit, expected_value) for conditional


_SQRT2_INV = 1 / sqrt(2)
_T_PHASE = np.exp(1j * _pi / 4)
_GATE_1Q_CACHE = {
    Gate.X: np.array([[0, 1], [1, 0]], dtype=complex),
    Gate.Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
    Gate.Z: np.array([[1, 0], [0, -1]], dtype=complex),
    Gate.H: np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT2_INV,
    Gate.S: np.array([[1, 0], [0, 1j]], dtype=complex),
    Gate.SDG: np.array([[1, 0], [0, -1j]], dtype=complex),
    Gate.T: np.array([[1, 0], [0, _T_PHASE]], dtype=complex),
    Gate.TDG: np.array([[1, 0], [0, np.conj(_T_PHASE)]], dtype=complex),
    Gate.SX: 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex),
}
_GATE_1Q_PARAM = {
    Gate.RX: lambda t: np.array([[cos(t/2), -1j*sin(t/2)], [-1j*sin(t/2), cos(t/2)]], dtype=complex),
    Gate.RY: lambda t: np.array([[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]], dtype=complex),
    Gate.RZ: lambda t: np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=complex),
}

def _get_gate_matrix(gate: Gate, params: tuple = ()) -> np.ndarray:
    return _GATE_1Q_CACHE[gate] if gate in _GATE_1Q_CACHE else _GATE_1Q_PARAM[gate](params[0])


_GATE_ADJOINT = {Gate.S: Gate.SDG, Gate.SDG: Gate.S, Gate.T: Gate.TDG, Gate.TDG: Gate.T}
_PARAM_GATES = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CP, Gate.RZZ, Gate.SEXC, Gate.DEXC})
_2Q_DRAW_SYMS = {Gate.CX: ("●", "X"), Gate.CZ: ("●", "●"), Gate.SWAP: ("╳", "╳"),
                 Gate.CP: ("●", "P"), Gate.ECR: ("ECR", "ECR"), Gate.RZZ: ("RZZ", "RZZ")}

class _ConditionalContext:
    def __init__(self, circuit: "Circuit", classical_bit: int, value: int):
        self._circuit = circuit
        self._classical_bit = classical_bit
        self._value = value

    def __enter__(self):
        self._circuit._current_condition = (self._classical_bit, self._value)
        return self

    def __exit__(self, *args):
        self._circuit._current_condition = None


class Circuit:
    """Lazy circuit builder. Adds operations to a list."""

    def __init__(self, n_qubits: int, n_classical: int | None = None):
        self.n_qubits = n_qubits
        self.n_classical = n_classical if n_classical is not None else n_qubits
        self.ops: list[Operation] = []
        self.param_values: dict[str, float] = {}
        self._current_condition: tuple[int, int] | None = None  # For c_if context manager
        self._initial_state: np.ndarray | None = None
        self.backend = None  # Callable[[Circuit, Observable], float] | None

    def _add(self, gate: Gate, qubits: tuple, params: tuple = (),
             classical_bit: int | None = None) -> "Circuit":
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise ValueError(f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit")
        if len(qubits) != len(set(qubits)):
            raise ValueError(f"Gate {gate.name} has duplicate qubits: {qubits}")
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
    def sexc(self, q0: int, q1: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.SEXC, (q0, q1), (theta,))
    def dexc(self, q0: int, q1: int, q2: int, q3: int, theta: "float | Parameter") -> "Circuit": return self._add(Gate.DEXC, (q0, q1, q2, q3), (theta,))
    def ccx(self, c1: int, c2: int, t: int) -> "Circuit": return self._add(Gate.CCX, (c1, c2, t))
    def ccz(self, a: int, b: int, c: int) -> "Circuit": return self._add(Gate.CCZ, tuple(sorted([a, b, c])))  # Symmetric → canonicalize

    def mcry(self, angle, controls: list[int], target: int) -> "Circuit":
        """Multi-controlled RY. Decomposes recursively into CX + RY."""
        n = len(controls)
        if n == 0: return self.ry(target, angle)
        if n == 1:
            self.ry(target, angle / 2)
            self.cx(controls[0], target)
            self.ry(target, -angle / 2)
            return self.cx(controls[0], target)
        self.mcry(angle / 2, controls[:-1], target)
        self.cx(controls[-1], target)
        self.mcry(-angle / 2, controls[:-1], target)
        return self.cx(controls[-1], target)

    def measure(self, q: int, c: int | None = None) -> "Circuit":
        """Measure qubit q, store result in classical bit c (defaults to q)."""
        return self._add(Gate.MEASURE, (q,), (), classical_bit=c if c is not None else q)

    def measure_all(self) -> "Circuit":
        for q in range(self.n_qubits): self.measure(q)
        return self

    def reset(self, q: int) -> "Circuit":
        """Reset qubit to |0>."""
        return self._add(Gate.RESET, (q,))

    def c_if(self, classical_bit: int, value: int = 1) -> _ConditionalContext:
        """Context manager for conditional operations."""
        return _ConditionalContext(self, classical_bit, value)

    @property
    def parameters(self) -> set[Parameter]:
        """Return set of all unbound Parameters in the circuit."""
        return {(p.param if isinstance(p, ScaledParam) else p) for op in self.ops for p in op.params if _is_param(p)}

    @property
    def trainable_parameters(self) -> set[Parameter]:
        """Return set of trainable (non-feature) Parameters."""
        return {p for p in self.parameters if p.trainable}

    @property
    def is_parameterized(self) -> bool:
        """True if any operation has an unbound Parameter."""
        return any(_has_parameter(op.params) for op in self.ops)

    def init_params(self, value: float = 0.0, seed: int | None = None, trainable_only: bool = True, order: str = 'sorted') -> dict[str, float]:
        """Initialize parameter values (stored in circuit)."""
        if order == 'circuit':
            seen = set()
            names = []
            for op in self.ops:
                for p in op.params:
                    if _is_param(p) and (not trainable_only or p.trainable) and p.name not in seen:
                        seen.add(p.name)
                        names.append(p.name)
        elif order == 'sorted':
            names = sorted(p.name for p in (self.trainable_parameters if trainable_only else self.parameters))
        else:
            raise ValueError(f"order must be 'sorted' or 'circuit', got {order!r}")
        if seed is not None:
            rng = np.random.default_rng(seed)
            vals = dict(zip(names, rng.uniform(0, 2 * np.pi, len(names))))
        else:
            vals = {n: value for n in names}
        self.param_values.update(vals)
        return vals

    def bind(self, values: dict[str, float] | None = None) -> "Circuit":
        """Return new Circuit with Parameters substituted. Defaults to stored param_values."""
        values = values if values is not None else self.param_values
        c = Circuit(self.n_qubits, self.n_classical)
        c._initial_state = self._initial_state
        c.backend = self.backend
        for op in self.ops:
            if _has_parameter(op.params):
                new_params = tuple(_resolve(p, values) if _is_param(p) else p for p in op.params)
                c.ops.append(Operation(op.gate, op.qubits, new_params, op.classical_bit, op.condition))
            else:
                c.ops.append(op)
        c._validated = True  # bound circuits have no unbound params and valid qubits
        return c

    def bind_params(self, values: dict[str, float]) -> "Circuit":
        """In-place parameter rebinding. Caches param slots for fast repeated calls."""
        if not hasattr(self, '_param_slots'):
            self._param_slots = [(i, op) for i, op in enumerate(self.ops) if _has_parameter(op.params)]
        for i, tmpl in self._param_slots:
            new_params = tuple(_resolve(p, values) if _is_param(p) else p for p in tmpl.params)
            self.ops[i] = Operation(tmpl.gate, tmpl.qubits, new_params, tmpl.classical_bit, tmpl.condition)
        return self

    def _structure_key(self) -> tuple:
        """Hashable key capturing circuit structure, ignoring parameter values."""
        return (self.n_qubits, tuple((op.gate, op.qubits) for op in self.ops))

    def compose(self, *others: "Circuit", qubit_map: dict[int, int] | None = None) -> "Circuit":
        """Append operations from other circuits onto this one."""
        for other in others:
            if other._initial_state is not None:
                if self._initial_state is not None or self.ops:
                    raise ValueError("cannot compose initialized circuit onto non-empty circuit")
                self._initial_state = other._initial_state
            for op in other.ops:
                qubits = tuple(qubit_map[q] for q in op.qubits) if qubit_map else op.qubits
                self.ops.append(Operation(op.gate, qubits, op.params, op.classical_bit, op.condition))
            self.param_values.update(other.param_values)
            if other.backend is not None:
                self.backend = other.backend
        return self

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

    def to_json(self) -> str:
        from .qasm import circuit_to_json
        return circuit_to_json(self)

    @classmethod
    def from_json(cls, s: str) -> "Circuit":
        from .qasm import circuit_from_json
        return circuit_from_json(s)

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
        s0, s1 = _2Q_DRAW_SYMS.get(g, (g.name, g.name))
        col = {qs[0]: s0, qs[1]: s1}
        for q in range(min(qs) + 1, max(qs)): col[q] = "│"
        return col             