"""
Core IR types - the single representation used throughout.

Contains:
    - Gate: Enum of supported gates (15 primitives)
    - Operation: Dataclass (gate, qubits, params)
    - Circuit: Lazy builder, just appends Operations
"""
from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass

class Gate(Enum):
    """16 primitive quantum gates."""
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

    # Two-qubit
    CX = auto()
    CZ = auto()
    CP = auto()  # Controlled phase
    SWAP = auto()

    # Measurement
    MEASURE = auto()

    @property                                                                                                                                                                                                        
    def n_qubits(self) -> int: return 2 if self in (Gate.CX, Gate.CZ, Gate.CP, Gate.SWAP) else 1                                                                                                                              
                                                                                                                                                                                                                    
    @property                                                                                                                                                                                                        
    def n_params(self) -> int: return 1 if self in (Gate.RX, Gate.RY, Gate.RZ, Gate.CP) else 0   


@dataclass(frozen=True)
class Operation:
    gate: Gate
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


# Circuit - lazy list of operations
class Circuit:
    """Lazy circuit builder. Adds operations to a list."""
    
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
    def sdg(self, q: int) -> Circuit: return self._add(Gate.SDG, (q,))
    def tdg(self, q: int) -> Circuit: return self._add(Gate.TDG, (q,))
    def rx(self, q: int, theta: float) -> Circuit: return self._add(Gate.RX, (q,), (theta,))
    def ry(self, q: int, theta: float) -> Circuit: return self._add(Gate.RY, (q,), (theta,))
    def rz(self, q: int, theta: float) -> Circuit: return self._add(Gate.RZ, (q,), (theta,))
    def cx(self, c: int, t: int) -> Circuit: return self._add(Gate.CX, (c, t))
    def cz(self, a: int, b: int) -> Circuit: return self._add(Gate.CZ, (a, b))
    def cp(self, c: int, t: int, theta: float) -> Circuit: return self._add(Gate.CP, (c, t), (theta,))
    def swap(self, a: int, b: int) -> Circuit: return self._add(Gate.SWAP, (a, b))
    def measure(self, q: int) -> Circuit: return self._add(Gate.MEASURE, (q,))

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
        if g.n_qubits == 1: return {qs[0]: g.name}                                                                                                                                                                   
        s0, s1 = ("●", "X") if g == Gate.CX else ("●", "●") if g == Gate.CZ else ("╳", "╳")                                                                                                                          
        col = {qs[0]: s0, qs[1]: s1}                                                                                                                                                                                 
        for q in range(min(qs) + 1, max(qs)): col[q] = "│"                                                                                                                                                           
        return col             