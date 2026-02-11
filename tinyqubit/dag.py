"""DAG-based circuit IR for compilation passes.

Nodes are operations, edges are data dependencies on qubit/classical-bit wires.
Gates on disjoint qubits are implicitly parallel (no edge between them).

Also contains the centralized commutation rules used by all passes.
"""
from __future__ import annotations

from bisect import insort
from .ir import Circuit, Operation, Gate


# Centralized commutation rules -----------------------------------------------

DIAGONAL_GATES = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ, Gate.CZ, Gate.CCZ}
_DIAG_1Q = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ}


def commutes(op1: Operation, op2: Operation) -> bool:
    """Check if two operations commute (conservative — only proven cases).

    WARNING: Only returns True for proven cases. When adding new gates to the IR,
    you MUST update this function or the optimizer will silently miss opportunities
    (safe) or worse, produce wrong circuits if a case incorrectly returns True.
    """
    if op1.gate in (Gate.MEASURE, Gate.RESET) or op2.gate in (Gate.MEASURE, Gate.RESET):
        return False
    if op1.condition is not None or op2.condition is not None:
        return False
    q1, q2 = set(op1.qubits), set(op2.qubits)
    if not (q1 & q2): return True
    # Single-qubit diagonal gates commute with CX on control qubit
    if op1.gate in _DIAG_1Q and op2.gate == Gate.CX: return op1.qubits[0] == op2.qubits[0]
    if op2.gate in _DIAG_1Q and op1.gate == Gate.CX: return op2.qubits[0] == op1.qubits[0]
    # RX commutes with CX on target qubit
    if op1.gate == Gate.RX and op2.gate == Gate.CX: return op1.qubits[0] == op2.qubits[1]
    if op2.gate == Gate.RX and op1.gate == Gate.CX: return op2.qubits[0] == op1.qubits[1]
    # Diagonal 1Q gates commute with CCX on control qubits
    if op1.gate in _DIAG_1Q and op2.gate == Gate.CCX: return op1.qubits[0] in op2.qubits[:2]
    if op2.gate in _DIAG_1Q and op1.gate == Gate.CCX: return op2.qubits[0] in op1.qubits[:2]
    # RX commutes with CCX on target
    if op1.gate == Gate.RX and op2.gate == Gate.CCX: return op1.qubits[0] == op2.qubits[2]
    if op2.gate == Gate.RX and op1.gate == Gate.CCX: return op2.qubits[0] == op1.qubits[2]
    # Diagonal gates commute with each other
    if op1.gate in DIAGONAL_GATES and op2.gate in DIAGONAL_GATES: return True
    return False


# DAG circuit -----------------------------------------------------------------

class DAGCircuit:
    """Dependency DAG over quantum operations.

    Each node holds an Operation. Edges encode qubit/cbit data dependencies:
    if two gates share a qubit wire, the earlier one is a predecessor of the later.
    """

    def __init__(self, n_qubits: int, n_classical: int = 0):
        self.n_qubits = n_qubits
        self.n_classical = n_classical or n_qubits
        self._ops: dict[int, Operation] = {}
        self._pred: dict[int, list[int]] = {}
        self._succ: dict[int, list[int]] = {}
        self._next_id = 0
        self._qubit_last: dict[int, int] = {}
        self._cbit_last: dict[int, int] = {}
        # Per-qubit wire tracking
        self._qubit_first: dict[int, int] = {}
        self._qubit_pred: dict[int, dict[int, int]] = {q: {} for q in range(n_qubits)}
        self._qubit_succ: dict[int, dict[int, int]] = {q: {} for q in range(n_qubits)}

    def __len__(self) -> int: return len(self._ops)
    def op(self, nid: int) -> Operation: return self._ops[nid]
    def predecessors(self, nid: int) -> list[int]: return self._pred[nid]
    def successors(self, nid: int) -> list[int]: return self._succ[nid]

    def next_on_qubit(self, nid: int, q: int) -> int | None:
        return self._qubit_succ[q].get(nid)
    def prev_on_qubit(self, nid: int, q: int) -> int | None:
        return self._qubit_pred[q].get(nid)
    def first_on_qubit(self, q: int) -> int | None:
        return self._qubit_first.get(q)
    def set_op(self, nid: int, op: Operation):
        """In-place op update (same qubits, different gate/params)."""
        self._ops[nid] = op

    def add_op(self, op: Operation) -> int:
        """Add operation, auto-wiring dependency edges from qubit/cbit usage."""
        nid = self._next_id; self._next_id += 1
        self._ops[nid] = op; self._pred[nid] = []; self._succ[nid] = []
        deps: set[int] = set()
        for q in op.qubits:
            if q in self._qubit_last:
                prev = self._qubit_last[q]
                deps.add(prev)
                self._qubit_succ[q][prev] = nid
                self._qubit_pred[q][nid] = prev
            else:
                self._qubit_first[q] = nid
            self._qubit_last[q] = nid
        if op.condition is not None and op.condition[0] in self._cbit_last:
            deps.add(self._cbit_last[op.condition[0]])
        if op.gate == Gate.MEASURE and op.classical_bit is not None:
            self._cbit_last[op.classical_bit] = nid
        for d in sorted(deps):
            self._succ[d].append(nid); self._pred[nid].append(d)
        return nid

    def remove_node(self, nid: int):
        """Remove node, rewire predecessor->successor edges."""
        # Per-qubit wire cleanup
        op = self._ops[nid]
        for q in op.qubits:
            prev = self._qubit_pred[q].pop(nid, None)
            nxt = self._qubit_succ[q].pop(nid, None)
            if prev is not None and nxt is not None:
                self._qubit_succ[q][prev] = nxt
                self._qubit_pred[q][nxt] = prev
            elif prev is not None:
                del self._qubit_succ[q][prev]
                if self._qubit_last.get(q) == nid: self._qubit_last[q] = prev
            elif nxt is not None:
                del self._qubit_pred[q][nxt]
                if self._qubit_first.get(q) == nid: self._qubit_first[q] = nxt
            else:
                self._qubit_last.pop(q, None)
                self._qubit_first.pop(q, None)
        # DAG edge rewiring
        for p in self._pred[nid]:
            self._succ[p] = [s for s in self._succ[p] if s != nid]
            for s in self._succ[nid]:
                if p not in self._pred[s]:
                    self._succ[p].append(s); self._pred[s].append(p)
        for s in self._succ[nid]:
            self._pred[s] = [p for p in self._pred[s] if p != nid]
        del self._ops[nid], self._pred[nid], self._succ[nid]

    def topological_order(self) -> list[int]:
        """Deterministic topological sort (Kahn's algorithm, smallest-id-first)."""
        in_deg = {nid: len(self._pred[nid]) for nid in self._ops}
        ready = sorted(nid for nid, d in in_deg.items() if d == 0)
        order: list[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for s in self._succ[nid]:
                in_deg[s] -= 1
                if in_deg[s] == 0: insort(ready, s)
        return order

    def topological_ops(self) -> list[Operation]:
        """Operations in deterministic topological order."""
        return [self._ops[nid] for nid in self.topological_order()]

    def depth(self) -> int:
        """Circuit depth (longest path through DAG)."""
        if not self._ops: return 0
        dist: dict[int, int] = {}
        for nid in self.topological_order():
            dist[nid] = max((dist[p] for p in self._pred[nid]), default=0) + 1
        return max(dist.values())

    def layers(self) -> list[list[Operation]]:
        """Group ops into parallel execution layers (ASAP scheduling)."""
        if not self._ops: return []
        layer_of: dict[int, int] = {}
        for nid in self.topological_order():
            layer_of[nid] = max((layer_of[p] + 1 for p in self._pred[nid]), default=0)
        result: list[list[Operation]] = [[] for _ in range(max(layer_of.values()) + 1)]
        for nid in self.topological_order():
            result[layer_of[nid]].append(self._ops[nid])
        return result

    @staticmethod
    def from_circuit(circuit: Circuit) -> DAGCircuit:
        """Build DAG from flat Circuit."""
        dag = DAGCircuit(circuit.n_qubits, circuit.n_classical)
        for op in circuit.ops: dag.add_op(op)
        return dag

    def to_circuit(self) -> Circuit:
        """Linearize DAG back to Circuit."""
        c = Circuit(self.n_qubits, self.n_classical)
        c.ops = self.topological_ops()
        return c
