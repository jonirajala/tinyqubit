"""
Pattern-based gate optimization.

Contains:
    - optimize(): Apply cancellation and merge rules until fixed point

Rules applied in deterministic order:
    - Cancellation: [X,X]→[], [H,H]→[], [CX,CX]→[], [SWAP,SWAP]→[]
    - Merge: [RZ(a),RZ(b)]→[RZ(a+b)], [RX(a),RX(b)]→[RX(a+b)], [RY(a),RY(b)]→[RY(a+b)]
    - Clifford: [S,S]→[Z], [T,T]→[S], [S†,S†]→[Z], [T†,T†]→[S†], [S,S†]→[], [T,T†]→[]
    - Hadamard conjugation: [H,X,H]→[Z], [H,Z,H]→[X]
    - Commutation-aware: cancel/merge gates through commuting intermediates
"""

from math import pi
from ..ir import Circuit, Operation, Gate


# Gates that cancel when applied twice (self-inverse)
CANCELLATION_GATES = {Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.CX, Gate.CZ, Gate.SWAP}

# Gates that can be merged (rotations)
MERGE_GATES = {Gate.RX, Gate.RY, Gate.RZ}

# Clifford gate merges: (G1, G2) -> result
CLIFFORD_MERGES = {
    (Gate.S, Gate.S): Gate.Z,
    (Gate.T, Gate.T): Gate.S,
    (Gate.SDG, Gate.SDG): Gate.Z,
    (Gate.TDG, Gate.TDG): Gate.SDG,
}

# Inverse pairs that cancel
INVERSE_PAIRS = {
    (Gate.S, Gate.SDG), (Gate.SDG, Gate.S),
    (Gate.T, Gate.TDG), (Gate.TDG, Gate.T),
}

# Diagonal gates (commute with each other)
DIAGONAL_GATES = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ, Gate.CZ}

# Hadamard conjugation: H·G·H = G'
HADAMARD_CONJUGATES = {
    Gate.X: Gate.Z,
    Gate.Z: Gate.X,
}

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
        if abs(angle) < 1e-9: del ops[i:i+2]
        else:
            ops[i] = Operation(op1.gate, op1.qubits, (angle,))
            del ops[i+1]
        return True
    return False


def _try_inverse_cancel(ops: list[Operation], i: int) -> bool:
    """Try to cancel inverse pairs like S·S† or T·T†. Returns True if cancelled."""
    if i + 1 >= len(ops): return False

    op1, op2 = ops[i], ops[i + 1]
    if op1.qubits == op2.qubits and (op1.gate, op2.gate) in INVERSE_PAIRS:
        del ops[i:i+2]
        return True
    return False


def _try_clifford_merge(ops: list[Operation], i: int) -> bool:
    """Try Clifford merge like S·S→Z or T·T→S. Returns True if merged."""
    if i + 1 >= len(ops): return False

    op1, op2 = ops[i], ops[i + 1]
    if op1.qubits == op2.qubits and (op1.gate, op2.gate) in CLIFFORD_MERGES:
        ops[i] = Operation(CLIFFORD_MERGES[(op1.gate, op2.gate)], op1.qubits)
        del ops[i+1]
        return True
    return False


def commutes(op1: Operation, op2: Operation) -> bool:
    """Check if two operations commute.

    WARNING: This function is conservative - it only returns True for proven cases.
    When adding new gates to the IR, you MUST update this function or the optimizer
    will silently miss optimization opportunities (safe) or worse, if you add a case
    that incorrectly returns True, the optimizer can produce semantically wrong circuits.
    """
    # Never commute across barriers: MEASURE, RESET, conditional ops
    if op1.gate in (Gate.MEASURE, Gate.RESET) or op2.gate in (Gate.MEASURE, Gate.RESET):
        return False
    if op1.condition is not None or op2.condition is not None:
        return False

    q1, q2 = set(op1.qubits), set(op2.qubits)

    # Disjoint qubits always commute
    if not (q1 & q2): return True

    # Single-qubit diagonal gates (Z, S, T, SDG, TDG, RZ) commute with CX on control qubit
    diag_1q = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ}
    if op1.gate in diag_1q and op2.gate == Gate.CX: return op1.qubits[0] == op2.qubits[0]
    if op2.gate in diag_1q and op1.gate == Gate.CX: return op2.qubits[0] == op1.qubits[0]

    # RX commutes with CX on target qubit
    if op1.gate == Gate.RX and op2.gate == Gate.CX: return op1.qubits[0] == op2.qubits[1]
    if op2.gate == Gate.RX and op1.gate == Gate.CX: return op2.qubits[0] == op1.qubits[1]

    # Diagonal gates commute with each other
    if op1.gate in DIAGONAL_GATES and op2.gate in DIAGONAL_GATES: return True

    return False


def _can_commute_to(ops: list[Operation], i: int, j: int) -> bool:
    """Check if ops[i] can commute past all ops between i and j."""
    for k in range(i + 1, j):
        if not commutes(ops[i], ops[k]): return False
    return True


def _try_cancel_through_commutation(ops: list[Operation], i: int, window: int = 5) -> bool:
    """Try to cancel ops[i] with a gate within window by commuting."""
    op1 = ops[i]
    if op1.gate not in CANCELLATION_GATES: return False

    for j in range(i + 1, min(i + window + 1, len(ops))):
        op2 = ops[j]
        # Same gate, same qubits, can cancel
        if op1.gate == op2.gate and op1.qubits == op2.qubits:
            if _can_commute_to(ops, i, j):
                del ops[j]
                del ops[i]
                return True
    return False


def _try_merge_through_commutation(ops: list[Operation], i: int, window: int = 5) -> bool:
    """Try to merge ops[i] with a rotation gate within window by commuting."""
    op1 = ops[i]
    if op1.gate not in MERGE_GATES: return False

    for j in range(i + 1, min(i + window + 1, len(ops))):
        op2 = ops[j]
        # Same rotation gate, same qubits
        if op1.gate == op2.gate and op1.qubits == op2.qubits:
            if _can_commute_to(ops, i, j):
                # Merge angles
                angle = (op1.params[0] + op2.params[0] + pi) % (2 * pi) - pi
                del ops[j]
                if abs(angle) < 1e-9:
                    del ops[i]  # Angle is ~0, remove entirely
                else:
                    ops[i] = Operation(op1.gate, op1.qubits, (angle,))
                return True
    return False


def _try_hadamard_conjugate(ops: list[Operation], i: int) -> bool:
    """Try to apply H·G·H = G' transformation. Returns True if changed."""
    if i + 2 >= len(ops): return False

    op1, op2, op3 = ops[i], ops[i + 1], ops[i + 2]

    # Check pattern: H(q) · G(q) · H(q) where G in {X, Z}
    if (op1.gate == Gate.H and op3.gate == Gate.H and
        op1.qubits == op2.qubits == op3.qubits and
        op2.gate in HADAMARD_CONJUGATES):
        # Replace H·G·H with conjugate
        ops[i] = Operation(HADAMARD_CONJUGATES[op2.gate], op1.qubits)
        del ops[i + 1:i + 3]
        return True

    return False


def _is_pauli_like(op: Operation, gate: Gate, rot_gate: Gate) -> bool:
    """Check if op is gate or rot_gate(π + 2πk) (equivalent up to global phase)."""
    return op.gate == gate or (op.gate == rot_gate and op.params and
                               abs(op.params[0] % (2 * pi) - pi) < 1e-9)


def _try_cx_conjugation(ops: list[Operation], i: int, window: int = 10) -> bool:
    """CX·P·CX patterns: Z(t)→Z both, X(c)→X both, Z(c)→Z(c), X(t)→X(t)."""
    if ops[i].gate != Gate.CX:
        return False
    c, t = ops[i].qubits

    for j in range(i + 2, min(i + window + 1, len(ops))):
        if ops[j].gate != Gate.CX or ops[j].qubits != (c, t):
            continue

        # Find Pauli-like gate on c or t
        pauli_idx, is_z, on_target = None, None, None
        for k in range(i + 1, j):
            op = ops[k]
            if len(op.qubits) == 1 and op.qubits[0] in (c, t):
                if _is_pauli_like(op, Gate.Z, Gate.RZ):
                    pauli_idx, is_z, on_target = k, True, op.qubits[0] == t
                    break
                if _is_pauli_like(op, Gate.X, Gate.RX):
                    pauli_idx, is_z, on_target = k, False, op.qubits[0] == t
                    break

        if pauli_idx is None:
            continue
        if not all(k == pauli_idx or commutes(ops[k], ops[i]) for k in range(i + 1, j)):
            continue

        # Build replacement gates (use rotation form if input was rotation)
        use_rot = ops[pauli_idx].gate in (Gate.RZ, Gate.RX)
        g = (Gate.RZ if use_rot else Gate.Z) if is_z else (Gate.RX if use_rot else Gate.X)
        make = lambda q: Operation(g, (q,), (pi,)) if use_rot else Operation(g, (q,))

        # Rules: Z(t)/X(c) → both qubits, Z(c)/X(t) → single qubit
        if is_z == on_target:  # Z on target or X on control → propagates to both
            new = [make(c), make(t)]
        else:  # Z on control or X on target → unchanged, removes 2 CX
            new = [make(t if on_target else c)]

        intermediates = [ops[k] for k in range(i + 1, j) if k != pauli_idx]
        ops[i:j + 1] = intermediates + new
        return True
    return False


def _try_hadamard_cx_to_cz(ops: list[Operation], i: int) -> bool:
    """H(t)·CX(c,t)·H(t) → CZ(c,t). CZ is symmetric so qubit order is preserved from CX."""
    if i + 2 >= len(ops):
        return False
    op1, op2, op3 = ops[i], ops[i + 1], ops[i + 2]
    if (op1.gate == Gate.H and op2.gate == Gate.CX and op3.gate == Gate.H and
            op1.qubits[0] == op2.qubits[1] == op3.qubits[0]):
        ops[i:i + 3] = [Operation(Gate.CZ, op2.qubits)]
        return True
    return False


def _single_pass(ops: list[Operation]) -> bool:
    """Single optimization pass. Returns True if any changes were made."""
    changed = False
    i = 0
    while i < len(ops):
        if (_try_cancel(ops, i) or _try_merge(ops, i) or
            _try_inverse_cancel(ops, i) or _try_clifford_merge(ops, i) or
            _try_hadamard_conjugate(ops, i) or _try_cx_conjugation(ops, i) or
            _try_hadamard_cx_to_cz(ops, i) or
            _try_cancel_through_commutation(ops, i) or _try_merge_through_commutation(ops, i)):
            changed = True
            i = max(0, i - 1)  # Back up to catch new adjacencies
        else:
            i += 1
    return changed


def optimize(circuit: Circuit, max_iterations: int = 1000) -> Circuit:
    """Optimize circuit by applying cancellation and merge rules until fixed point.

    Args:
        circuit: Circuit to optimize.
        max_iterations: Safety cap to prevent infinite loops (should never be hit).
    """
    ops = list(circuit.ops)
    for _ in range(max_iterations):
        if not _single_pass(ops):
            break
    result = Circuit(circuit.n_qubits)
    result.ops = ops
    return result
