"""Pattern-based gate cancellation, merging, and conjugation until fixed point."""

import sys, os
from math import pi
from ..ir import Circuit, Operation, Gate, _has_parameter, Parameter
from ..dag import DAGCircuit, commutes, DIAGONAL_GATES

_DEBUG = int(os.environ.get("TINYQUBIT_DEBUG", "0"))


# Partner rules: (gate1, gate2, result) — find gate2 through commuting intermediates
#   result = None   → cancel both
#   result = Gate   → replace gate1 with result, remove gate2
#   result = "merge" → sum rotation angles (remove both if ≈ 0)
# Order matters: cancellation before merge for same gate pair
PARTNER_RULES: list[tuple[Gate, Gate, Gate | str | None]] = [
    # Self-inverse (A·A = I)
    (Gate.X,    Gate.X,    None),
    (Gate.Y,    Gate.Y,    None),
    (Gate.Z,    Gate.Z,    None),
    (Gate.H,    Gate.H,    None),
    (Gate.CX,   Gate.CX,   None),
    (Gate.CZ,   Gate.CZ,   None),
    (Gate.SWAP, Gate.SWAP,  None),
    (Gate.CCX,  Gate.CCX,  None),
    (Gate.CCZ,  Gate.CCZ,  None),
    # Inverse pairs (A·A† = I)
    (Gate.S,    Gate.SDG,  None),
    (Gate.SDG,  Gate.S,    None),
    (Gate.T,    Gate.TDG,  None),
    (Gate.TDG,  Gate.T,    None),
    # Clifford merges (A·A = B)
    (Gate.S,    Gate.S,    Gate.Z),
    (Gate.T,    Gate.T,    Gate.S),
    (Gate.SDG,  Gate.SDG,  Gate.Z),
    (Gate.TDG,  Gate.TDG,  Gate.SDG),
    # Rotation merges
    (Gate.RX,   Gate.RX,   "merge"),
    (Gate.RY,   Gate.RY,   "merge"),
    (Gate.RZ,   Gate.RZ,   "merge"),
    (Gate.CP,   Gate.CP,   "merge"),
    # Cross-type diagonal merges: different named diagonal gates → RZ(sum)
    *((a, b, "cross_diag") for a in (Gate.S, Gate.SDG, Gate.T, Gate.TDG, Gate.Z)
      for b in (Gate.S, Gate.SDG, Gate.T, Gate.TDG, Gate.Z) if a != b),
]

_DIAG_ANGLE = {Gate.T: pi/4, Gate.TDG: -pi/4, Gate.S: pi/2, Gate.SDG: -pi/2, Gate.Z: pi}

# Build index: gate -> [(partner_gate, result), ...]
_PARTNER_INDEX: dict[Gate, list[tuple[Gate, Gate | str | None]]] = {}
for _g1, _g2, _res in PARTNER_RULES:
    _PARTNER_INDEX.setdefault(_g1, []).append((_g2, _res))

# Conjugation: bookend·inner·bookend → result (strict adjacency, no commutation walk)
# 1Q: all on same qubit — replace nid with result, remove mid+end
CONJUGATE_1Q: list[tuple[Gate, Gate, Gate]] = [
    (Gate.H, Gate.X, Gate.Z),
    (Gate.H, Gate.Z, Gate.X),
]

# 2Q: bookend(q)·inner_2q·bookend(q), q at position pos in inner's qubits
# (bookend, inner, bookend_pos_in_inner, result, bookend_pos_in_result)
# Replace mid with result, remove nid+end; swap qubits when pos != result_pos
CONJUGATE_2Q: list[tuple[Gate, Gate, int, Gate, int]] = [
    (Gate.H, Gate.CX, 1, Gate.CZ, 1),    # H·CX·H = CZ (H on target)
    (Gate.H, Gate.CZ, 0, Gate.CX, 1),    # H(q0)·CZ·H(q0) = CX (H qubit → CX target)
    (Gate.H, Gate.CZ, 1, Gate.CX, 1),    # H(q1)·CZ·H(q1) = CX (H qubit → CX target)
]

_CONJUGATE_BOOKENDS = frozenset({b for b, _, _ in CONJUGATE_1Q} | {b for b, _, _, _, _ in CONJUGATE_2Q})


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


def _try_partner_rule(dag: DAGCircuit, nid: int) -> bool:
    """Try partner-based rules: cancel, inverse cancel, clifford merge, rotation merge."""
    op = dag.op(nid)
    rules = _PARTNER_INDEX.get(op.gate)
    if not rules:
        return False
    for partner_gate, result in rules:
        is_merge = result == "merge"
        is_cross = result == "cross_diag"
        if is_merge and _has_parameter(op.params):
            continue
        def _match(c, pg=partner_gate):
            return c.gate == pg and c.qubits == op.qubits and c.condition == op.condition
        pred = (lambda c: _match(c) and not _has_parameter(c.params)) if is_merge else _match
        match = _find_partner(dag, nid, pred)
        if match is None:
            continue
        if _DEBUG >= 2:
            op2 = dag.op(match)
            if result is None: print(f"    opt: cancel {op.gate.name}+{op2.gate.name} on {op.qubits}", file=sys.stderr)
            elif is_merge or is_cross: print(f"    opt: merge {op.gate.name}+{op2.gate.name} on {op.qubits}", file=sys.stderr)
            else: print(f"    opt: {op.gate.name}+{partner_gate.name} -> {result.name} on {op.qubits}", file=sys.stderr)
        if result is None:
            dag.remove_node(match)
            dag.remove_node(nid)
        elif is_merge:
            op2 = dag.op(match)
            angle = (op.params[0] + op2.params[0] + pi) % (2 * pi) - pi
            if abs(angle) < 1e-9:
                dag.remove_node(match)
                dag.remove_node(nid)
            else:
                dag.set_op(nid, Operation(op.gate, op.qubits, (angle,), condition=op.condition))
                dag.remove_node(match)
        elif is_cross:
            angle = (_DIAG_ANGLE[op.gate] + _DIAG_ANGLE[dag.op(match).gate] + pi) % (2 * pi) - pi
            if abs(angle) < 1e-9:
                dag.remove_node(match)
                dag.remove_node(nid)
            else:
                dag.set_op(nid, Operation(Gate.RZ, op.qubits, (angle,), condition=op.condition))
                dag.remove_node(match)
        else:
            dag.set_op(nid, Operation(result, op.qubits, condition=op.condition))
            dag.remove_node(match)
        return True
    return False


def _try_conjugate(dag: DAGCircuit, nid: int, basis: frozenset[Gate] | None = None) -> bool:
    """Try bookend·inner·bookend conjugation patterns (strict adjacency)."""
    op = dag.op(nid)
    if op.gate not in _CONJUGATE_BOOKENDS:
        return False
    q = op.qubits[0]
    mid = dag.next_on_qubit(nid, q)
    if mid is None or mid not in dag._ops:
        return False
    mid_op = dag.op(mid)
    end = dag.next_on_qubit(mid, q)
    if end is None or end not in dag._ops:
        return False
    end_op = dag.op(end)
    if end_op.gate != op.gate or end_op.qubits != op.qubits:
        return False
    if not (op.condition == mid_op.condition == end_op.condition):
        return False
    cond = op.condition
    # 1Q rules
    for bookend, inner, result in CONJUGATE_1Q:
        if op.gate == bookend and mid_op.gate == inner and mid_op.qubits == op.qubits:
            if _DEBUG >= 2:
                print(f"    opt: conjugate {bookend.name}*{inner.name}*{bookend.name} -> {result.name} on {op.qubits}", file=sys.stderr)
            dag.set_op(nid, Operation(result, op.qubits, condition=cond))
            dag.remove_node(mid)
            dag.remove_node(end)
            return True
    # 2Q rules — skip if result gate not in target basis
    for bookend, inner, pos, result, result_pos in CONJUGATE_2Q:
        if basis is not None and result not in basis:
            continue
        if op.gate == bookend and mid_op.gate == inner and mid_op.qubits[pos] == q:
            if _DEBUG >= 2:
                print(f"    opt: conjugate {bookend.name}*{inner.name}*{bookend.name} -> {result.name} on q{q}", file=sys.stderr)
            rq = mid_op.qubits if pos == result_pos else mid_op.qubits[::-1]
            dag.set_op(mid, Operation(result, rq, condition=cond))
            dag.remove_node(nid)
            dag.remove_node(end)
            return True
    return False


def _is_pauli_like(op: Operation, gate: Gate, rot_gate: Gate) -> bool:
    """Check if op is gate or rot_gate(π + 2πk) (equivalent up to global phase)."""
    return op.gate == gate or (op.gate == rot_gate and op.params and
                               not isinstance(op.params[0], Parameter) and
                               abs(op.params[0] % (2 * pi) - pi) < 1e-9)


def _try_cx_conjugation(dag: DAGCircuit, nid: int) -> bool:
    """CX·P·CX patterns: Z(t)→Z both, X(c)→X both, Z(c)→Z(c), X(t)→X(t)."""
    _ops = dag._ops  # local refs for hot loop
    op = _ops.get(nid)
    if op is None or op.gate != Gate.CX or op.condition is not None:
        return False
    c, t = op.qubits
    _qsucc = dag._qubit_succ  # inline next_on_qubit
    _qsucc_c = _qsucc[c].get

    # Walk control qubit wire to find matching CX
    cur = _qsucc_c(nid)
    steps = 0
    while cur is not None and steps < 50:
        cur_op = _ops.get(cur)
        if cur_op is None:
            cur = _qsucc_c(cur); steps += 1; continue
        if cur_op.gate == Gate.CX and cur_op.qubits == (c, t):
            pauli_nid, is_z, on_target = None, None, None

            _PI2 = 2 * pi
            for q in (c, t):
                _qsucc_q = _qsucc[q].get
                mid = _qsucc_q(nid)
                while mid is not None and mid != cur:
                    mid_op = _ops.get(mid)
                    if mid_op is not None:
                        if len(mid_op.qubits) == 1 and mid_op.qubits[0] in (c, t):
                            mg = mid_op.gate
                            if mg == Gate.Z:
                                pauli_nid, is_z, on_target = mid, True, mid_op.qubits[0] == t; break
                            if mg == Gate.X:
                                pauli_nid, is_z, on_target = mid, False, mid_op.qubits[0] == t; break
                            if (mg == Gate.RZ or mg == Gate.RX) and mid_op.params and not isinstance(mid_op.params[0], Parameter) and abs(mid_op.params[0] % _PI2 - pi) < 1e-9:
                                pauli_nid, is_z, on_target = mid, mg == Gate.RZ, mid_op.qubits[0] == t; break
                    mid = _qsucc_q(mid)
                if pauli_nid is not None:
                    break

            if pauli_nid is None:
                cur = _qsucc_c(cur); steps += 1; continue

            # Check all intermediates (except pauli) commute with CX
            ok = True
            for q in (c, t):
                _qs = _qsucc[q].get
                mid = _qs(nid)
                while mid is not None and mid != cur:
                    mid_op = _ops.get(mid)
                    if mid_op is not None and mid != pauli_nid:
                        if not commutes(mid_op, op):
                            ok = False; break
                    mid = _qs(mid)
                if not ok: break
            if not ok:
                cur = _qsucc_c(cur); steps += 1; continue

            pauli_op = _ops[pauli_nid]
            use_rot = pauli_op.gate in (Gate.RZ, Gate.RX)
            g = (Gate.RZ if use_rot else Gate.Z) if is_z else (Gate.RX if use_rot else Gate.X)
            make = lambda q: Operation(g, (q,), (pi,)) if use_rot else Operation(g, (q,))

            if _DEBUG >= 2:
                side = 'target' if on_target else 'control'
                print(f"    opt: CX sandwich {pauli_op.gate.name} on {side} -> eliminate CX pair on ({c},{t})", file=sys.stderr)
            # Remove pauli; replace CX gates with single-qubit equivalents
            dag.remove_node(pauli_nid)

            if is_z == on_target:  # Z on target or X on control → propagates to both
                dag.replace_op(nid, make(c))
                dag.replace_op(cur, make(t))
            else:  # Z on control or X on target → single qubit
                dag.replace_op(nid, make(t if on_target else c))
                dag.remove_node(cur)

            return True

        cur = _qsucc_c(cur)
        steps += 1
    return False


def _dag_pass(dag: DAGCircuit, basis: frozenset[Gate] | None = None) -> bool:
    """Single DAG-native optimization pass. Returns True if any changes made."""
    changed = False
    order = dag.topological_order()  # snapshot
    for nid in order:
        if nid not in dag._ops:
            continue
        if _try_partner_rule(dag, nid) or _try_conjugate(dag, nid, basis) or _try_cx_conjugation(dag, nid):
            changed = True
    return changed


def optimize(inp, max_iterations: int = 1000, basis: frozenset[Gate] | None = None):
    """Optimize circuit by applying cancellation and merge rules. Accepts Circuit or DAGCircuit."""
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp
    for _ in range(max_iterations):
        if not _dag_pass(dag, basis):
            break
    return dag.to_circuit() if from_circuit else dag
