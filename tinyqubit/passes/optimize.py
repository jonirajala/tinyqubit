"""
Pattern-based gate optimization.

Contains:
    - optimize(): Apply cancellation and merge rules until fixed point

Rules are declarative and applied in deterministic order:
    - Cancellation: [X,X]→[], [H,H]→[], [CX,CX]→[], [SWAP,SWAP]→[]
    - Merge: [RZ(a),RZ(b)]→[RZ(a+b)], [RX(a),RX(b)]→[RX(a+b)], [RY(a),RY(b)]→[RY(a+b)]
"""

from math import pi
from ..ir import Circuit, Operation, Gate


# Gates that cancel when applied twice (self-inverse)
CANCELLATION_GATES = {Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.CX, Gate.CZ, Gate.SWAP}

# Gates that can be merged (rotations)
MERGE_GATES = {Gate.RX, Gate.RY, Gate.RZ}


def _try_cancel(ops: list[Operation], i: int) -> bool:
    """Try to cancel ops[i] with ops[i+1]. Returns True if cancelled."""
    if i + 1 >= len(ops): return False

    op1, op2 = ops[i], ops[i + 1]
    if op1.gate == op2.gate and op1.gate in CANCELLATION_GATES and op1.qubits == op2.qubits:
        del ops[i:i+2]
        return True
    return False


def _try_merge(ops: list[Operation], i: int) -> bool:
    """Try to merge ops[i] with ops[i+1] (rotations). Returns True if merged."""
    if i + 1 >= len(ops): return False

    op1, op2 = ops[i], ops[i + 1]
    if op1.gate == op2.gate and op1.gate in MERGE_GATES and op1.qubits == op2.qubits:
        angle = (op1.params[0] + op2.params[0] + pi) % (2 * pi) - pi  # normalize to [-π, π]
        if abs(angle) < 1e-10: del ops[i:i+2]
        else:
            ops[i] = Operation(op1.gate, op1.qubits, (angle,))
            del ops[i+1]
        return True
    return False


def _single_pass(ops: list[Operation]) -> bool:
    """Single optimization pass. Returns True if any changes were made."""
    changed = False
    i = 0
    while i < len(ops):
        if _try_cancel(ops, i) or _try_merge(ops, i):
            changed = True
            i = max(0, i - 1)  # Back up to catch new adjacencies
        else:
            i += 1
    return changed


def optimize(circuit: Circuit) -> Circuit:
    """Optimize circuit by applying cancellation and merge rules until fixed point."""
    ops = list(circuit.ops)
    while _single_pass(ops):
        pass
    result = Circuit(circuit.n_qubits)
    result.ops = ops
    return result
