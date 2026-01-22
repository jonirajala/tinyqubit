"""
Diagonal gate pushing pass.

Moves diagonal gates backward through commuting gates for merge opportunities.
Rules: diag commutes through CX control, CZ either qubit, other diagonals, disjoint qubits.
"""
from ..ir import Circuit, Operation, Gate

DIAGONAL_1Q = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ}


def _can_push(diag: Operation, other: Operation) -> bool:
    """Check if diagonal can commute backward past other."""
    if other.gate in (Gate.MEASURE, Gate.RESET) or diag.condition or other.condition: return False
    q = diag.qubits[0]
    if q not in other.qubits: return True
    if other.gate == Gate.CX and q == other.qubits[0]: return True
    if other.gate == Gate.CZ or other.gate in DIAGONAL_1Q: return True
    return False


def push_diagonals(circuit: Circuit) -> Circuit:
    """Push diagonal gates backward to gather for merging."""
    ops = list(circuit.ops)
    changed = True
    while changed:
        changed = False
        for i in range(len(ops) - 1, -1, -1):
            if i >= len(ops) or ops[i].gate not in DIAGONAL_1Q: continue
            # Find target position, only push if passing non-diagonal
            target, passed_non_diag = i, False
            for j in range(i - 1, -1, -1):
                if not _can_push(ops[i], ops[j]): break
                target = j
                if ops[j].gate not in DIAGONAL_1Q: passed_non_diag = True
            if passed_non_diag and target != i:
                ops.insert(target, ops.pop(i))
                changed = True
    result = Circuit(circuit.n_qubits, circuit.n_classical)
    result.ops = ops
    return result
