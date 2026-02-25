"""
Pattern-based gate optimization (DAG-native).

Contains:
    - optimize(): Apply cancellation and merge rules until fixed point

Rules applied in deterministic order:
    - Cancellation: [X,X]->[], [H,H]->[], [CX,CX]->[], [SWAP,SWAP]->[]
    - Merge: [RZ(a),RZ(b)]->[RZ(a+b)], [RX(a),RX(b)]->[RX(a+b)], [RY(a),RY(b)]->[RY(a+b)]
    - Clifford: [S,S]->[Z], [T,T]->[S], [S†,S†]->[Z], [T†,T†]->[S†], [S,S†]->[], [T,T†]->[]
    - Hadamard conjugation: [H,X,H]->[Z], [H,Z,H]->[X]
    - Commutation-aware: cancel/merge gates through commuting intermediates
"""

from math import pi
from ..ir import Circuit, Operation, Gate, _has_parameter, Parameter
from ..dag import DAGCircuit, commutes, DIAGONAL_GATES


# Gates that cancel when applied twice (self-inverse)
CANCELLATION_GATES = {Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.CX, Gate.CZ, Gate.SWAP, Gate.CCX, Gate.CCZ}

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

# Hadamard conjugation: H·G·H = G'
HADAMARD_CONJUGATES = {
    Gate.X: Gate.Z,
    Gate.Z: Gate.X,
}


def _find_partner(dag: DAGCircuit, nid: int, predicate, max_steps: int = 50) -> int | None:
    """Walk forward on the first qubit's wire from nid, checking commutation.

    When a node matching predicate is found, verify commutation on ALL shared
    qubit wires between nid and the match. Returns match nid or None.
    """
    op = dag.op(nid)
    q0 = op.qubits[0]

    cur = dag.next_on_qubit(nid, q0)
    for _ in range(max_steps):
        if cur is None:
            break
        if cur not in dag._ops:
            cur = dag.next_on_qubit(cur, q0)
            continue
        candidate = dag.op(cur)
        if predicate(candidate):
            # Verify commutation on ALL shared qubit wires
            ok = True
            for q in op.qubits:
                mid = dag.next_on_qubit(nid, q)
                while mid is not None and mid != cur:
                    if mid in dag._ops and not commutes(op, dag.op(mid)):
                        ok = False
                        break
                    mid = dag.next_on_qubit(mid, q)
                if not ok:
                    break
            if ok:
                return cur
        # Check if we can commute past this node
        if not commutes(op, candidate):
            return None
        cur = dag.next_on_qubit(cur, q0)
    return None


def _try_cancel(dag: DAGCircuit, nid: int) -> bool:
    """Try to cancel nid with a matching gate on the qubit wire."""
    op = dag.op(nid)
    if op.gate not in CANCELLATION_GATES:
        return False

    match = _find_partner(dag, nid, lambda c: c.gate == op.gate and c.qubits == op.qubits)
    if match is not None:
        dag.remove_node(match)
        dag.remove_node(nid)
        return True
    return False


def _try_merge(dag: DAGCircuit, nid: int) -> bool:
    """Try to merge nid with a matching rotation on the qubit wire."""
    op = dag.op(nid)
    if op.gate not in MERGE_GATES:
        return False
    if _has_parameter(op.params):
        return False

    match = _find_partner(dag, nid, lambda c: c.gate == op.gate and c.qubits == op.qubits
                          and not _has_parameter(c.params))
    if match is not None:
        op2 = dag.op(match)
        angle = (op.params[0] + op2.params[0] + pi) % (2 * pi) - pi
        if abs(angle) < 1e-9:
            dag.remove_node(match)
            dag.remove_node(nid)
        else:
            dag.set_op(nid, Operation(op.gate, op.qubits, (angle,)))
            dag.remove_node(match)
        return True
    return False


def _try_inverse_cancel(dag: DAGCircuit, nid: int) -> bool:
    """Try to cancel inverse pairs like S·S† or T·T†."""
    op = dag.op(nid)
    inv_gates = {g2 for (g1, g2) in INVERSE_PAIRS if g1 == op.gate}
    if not inv_gates:
        return False

    match = _find_partner(dag, nid, lambda c: c.gate in inv_gates and c.qubits == op.qubits)
    if match is not None:
        dag.remove_node(match)
        dag.remove_node(nid)
        return True
    return False


def _try_clifford_merge(dag: DAGCircuit, nid: int) -> bool:
    """Try Clifford merge like S·S→Z or T·T→S."""
    op = dag.op(nid)
    merge_gates = {g2: result for (g1, g2), result in CLIFFORD_MERGES.items() if g1 == op.gate}
    if not merge_gates:
        return False

    match = _find_partner(dag, nid, lambda c: c.gate in merge_gates and c.qubits == op.qubits)
    if match is not None:
        result_gate = merge_gates[dag.op(match).gate]
        dag.set_op(nid, Operation(result_gate, op.qubits))
        dag.remove_node(match)
        return True
    return False


def _try_hadamard_conjugate(dag: DAGCircuit, nid: int) -> bool:
    """Try to apply H·G·H = G' transformation."""
    op = dag.op(nid)
    if op.gate != Gate.H:
        return False

    q = op.qubits[0]
    mid = dag.next_on_qubit(nid, q)
    if mid is None or mid not in dag._ops:
        return False
    mid_op = dag.op(mid)
    if mid_op.gate not in HADAMARD_CONJUGATES or mid_op.qubits != op.qubits:
        return False

    end = dag.next_on_qubit(mid, q)
    if end is None or end not in dag._ops:
        return False
    end_op = dag.op(end)
    if end_op.gate != Gate.H or end_op.qubits != op.qubits:
        return False

    # Replace H·G·H with conjugate
    dag.set_op(nid, Operation(HADAMARD_CONJUGATES[mid_op.gate], op.qubits))
    dag.remove_node(end)
    dag.remove_node(mid)
    return True


def _is_pauli_like(op: Operation, gate: Gate, rot_gate: Gate) -> bool:
    """Check if op is gate or rot_gate(π + 2πk) (equivalent up to global phase)."""
    return op.gate == gate or (op.gate == rot_gate and op.params and
                               not isinstance(op.params[0], Parameter) and
                               abs(op.params[0] % (2 * pi) - pi) < 1e-9)


def _try_cx_conjugation(dag: DAGCircuit, nid: int) -> bool:
    """CX·P·CX patterns: Z(t)→Z both, X(c)→X both, Z(c)→Z(c), X(t)→X(t)."""
    op = dag.op(nid)
    if op.gate != Gate.CX:
        return False
    c, t = op.qubits

    # Walk control qubit wire to find matching CX (don't stop at non-commuting gates)
    cur = dag.next_on_qubit(nid, c)
    steps = 0
    while cur is not None and steps < 50:
        if cur not in dag._ops:
            cur = dag.next_on_qubit(cur, c)
            steps += 1
            continue
        cur_op = dag.op(cur)
        if cur_op.gate == Gate.CX and cur_op.qubits == (c, t):
            # Found matching CX — look for Pauli-like gate between them
            pauli_nid, is_z, on_target = None, None, None

            # Search on both control and target wires
            for q in (c, t):
                mid = dag.next_on_qubit(nid, q)
                while mid is not None and mid != cur:
                    if mid in dag._ops:
                        mid_op = dag.op(mid)
                        if len(mid_op.qubits) == 1 and mid_op.qubits[0] in (c, t):
                            if _is_pauli_like(mid_op, Gate.Z, Gate.RZ):
                                pauli_nid, is_z, on_target = mid, True, mid_op.qubits[0] == t
                                break
                            if _is_pauli_like(mid_op, Gate.X, Gate.RX):
                                pauli_nid, is_z, on_target = mid, False, mid_op.qubits[0] == t
                                break
                    mid = dag.next_on_qubit(mid, q)
                if pauli_nid is not None:
                    break

            if pauli_nid is None:
                cur = dag.next_on_qubit(cur, c)
                steps += 1
                continue

            # Check all intermediates (except pauli) commute with CX
            ok = True
            for q in (c, t):
                mid = dag.next_on_qubit(nid, q)
                while mid is not None and mid != cur:
                    if mid in dag._ops and mid != pauli_nid:
                        if not commutes(dag.op(mid), op):
                            ok = False
                            break
                    mid = dag.next_on_qubit(mid, q)
                if not ok:
                    break
            if not ok:
                cur = dag.next_on_qubit(cur, c)
                steps += 1
                continue

            # Build replacement
            pauli_op = dag.op(pauli_nid)
            use_rot = pauli_op.gate in (Gate.RZ, Gate.RX)
            g = (Gate.RZ if use_rot else Gate.Z) if is_z else (Gate.RX if use_rot else Gate.X)
            make = lambda q: Operation(g, (q,), (pi,)) if use_rot else Operation(g, (q,))

            # Remove the 2 CX gates and the pauli gate
            dag.remove_node(cur)       # second CX
            dag.remove_node(pauli_nid) # pauli

            if is_z == on_target:  # Z on target or X on control → propagates to both
                dag.set_op(nid, make(c))
                dag.add_op(make(t))
            else:  # Z on control or X on target → single qubit
                dag.set_op(nid, make(t if on_target else c))

            return True

        cur = dag.next_on_qubit(cur, c)
        steps += 1
    return False


def _try_hadamard_cx_to_cz(dag: DAGCircuit, nid: int) -> bool:
    """H(t)·CX(c,t)·H(t) → CZ(c,t)."""
    op = dag.op(nid)
    if op.gate != Gate.H:
        return False

    q = op.qubits[0]
    mid = dag.next_on_qubit(nid, q)
    if mid is None or mid not in dag._ops:
        return False
    mid_op = dag.op(mid)
    if mid_op.gate != Gate.CX or mid_op.qubits[1] != q:
        return False

    end = dag.next_on_qubit(mid, q)
    if end is None or end not in dag._ops:
        return False
    end_op = dag.op(end)
    if end_op.gate != Gate.H or end_op.qubits[0] != q:
        return False

    # Replace H·CX·H with CZ
    dag.set_op(mid, Operation(Gate.CZ, mid_op.qubits))
    dag.remove_node(end)
    dag.remove_node(nid)
    return True


def _dag_pass(dag: DAGCircuit) -> bool:
    """Single DAG-native optimization pass. Returns True if any changes made."""
    changed = False
    order = dag.topological_order()  # snapshot
    for nid in order:
        if nid not in dag._ops:
            continue
        if (_try_cancel(dag, nid) or _try_merge(dag, nid) or
            _try_inverse_cancel(dag, nid) or _try_clifford_merge(dag, nid) or
            _try_hadamard_conjugate(dag, nid) or _try_cx_conjugation(dag, nid) or
            _try_hadamard_cx_to_cz(dag, nid)):
            changed = True
    return changed


def optimize(inp, max_iterations: int = 1000):
    """Optimize circuit by applying cancellation and merge rules. Accepts Circuit or DAGCircuit."""
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp
    for _ in range(max_iterations):
        if not _dag_pass(dag):
            break
        # Rebuild to fix wire tracking after add_op in CX conjugation
        new = DAGCircuit(dag.n_qubits, dag.n_classical)
        for op in dag.topological_ops():
            new.add_op(op)
        dag = new
    return dag.to_circuit() if from_circuit else dag
