"""Qubit routing using SABRE algorithm."""

from ..ir import Circuit, Operation, Gate
from ..dag import DAGCircuit
from ..target import Target
from ..tracker import QubitTracker, PendingSwap


def _score_swap(swap: tuple[int, int], front: list[int], extended: list[int],
                dag: DAGCircuit, l2p: list[int], p2l: list[int],
                dist: list[list[int | float]], decay: list[list[float]],
                qubit_depth: list[int] | None = None) -> float:
    """Score a SWAP candidate. Lower is better."""
    p0, p1 = swap
    new_l2p = l2p.copy()
    if p2l[p0] != -1: new_l2p[p2l[p0]] = p1
    if p2l[p1] != -1: new_l2p[p2l[p1]] = p0

    max_depth = (max(qubit_depth) or 1) if qubit_depth else 0
    score = decay[min(p0, p1)][max(p0, p1)]
    for nid, weight in [(g, 1.0) for g in front] + [(g, 0.5) for g in extended]:
        op = dag.op(nid)
        if op.gate.n_qubits == 2:
            pa, pb = new_l2p[op.qubits[0]], new_l2p[op.qubits[1]]
            d = dist[pa][pb]
            if 0 <= d < float('inf'):
                if qubit_depth:
                    weight *= 0.5 + 0.5 * (max(qubit_depth[pa], qubit_depth[pb]) / max_depth)
                score += weight * d
    return score


def route(inp, target: Target, initial_layout: list[int] | None = None, objective: str | None = None, lookahead_depth: int = 20):
    """Route circuit using SABRE algorithm. Accepts Circuit or DAGCircuit.

    initial_layout: logical->physical mapping. layout[i] = physical qubit for logical i.
    """
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp

    if target.n_qubits < dag.n_qubits:
        raise ValueError(f"Target has {target.n_qubits} qubits but circuit needs {dag.n_qubits}")

    tracker = QubitTracker(target.n_qubits, initial_layout=initial_layout)
    result = DAGCircuit(target.n_qubits, dag.n_classical)

    def finalize():
        result._tracker = tracker
        if from_circuit:
            c = result.to_circuit(); c._tracker = tracker; return c
        return result

    if target.is_all_to_all() or not dag._ops:
        for op in dag.topological_ops(): result.add_op(op)
        return finalize()

    # Track in-degrees separately (we read from dag without modifying it)
    in_deg = {nid: len(dag.predecessors(nid)) for nid in dag._ops}
    executed: set[int] = set()
    dist = target.all_pairs_error_costs() if objective == "error" else target.all_pairs_distances()
    n = target.n_qubits
    decay = [[0.0] * n for _ in range(n)]
    depth_aware = objective == "depth"
    qubit_depth = [0] * n if depth_aware else None
    l2p = list(initial_layout) if initial_layout is not None else list(range(dag.n_qubits))
    p2l = [-1] * n
    for lq, pq in enumerate(l2p): p2l[pq] = lq

    def mark_done(nid: int):
        executed.add(nid)
        for s in dag.successors(nid): in_deg[s] -= 1

    def flush():
        for p in tracker.flush():
            if isinstance(p, PendingSwap): result.add_op(Operation(Gate.SWAP, (p.phys_a, p.phys_b)))
            else: result.add_op(Operation(p.gate, p.phys_qubits, p.params))

    swap_budget = len(dag._ops) * n  # scale with both ops and qubits
    swaps_inserted = 0
    stall = 0  # consecutive SWAPs without executing a 2Q gate
    while True:
        # Front layer: gates with all dependencies satisfied
        front = sorted(nid for nid in dag._ops if in_deg.get(nid, 0) == 0 and nid not in executed)
        if not front: break

        # Barriers (MEASURE/RESET/conditional) must flush pending ops first
        barrier = next((nid for nid in front if dag.op(nid).gate in (Gate.MEASURE, Gate.RESET) or dag.op(nid).condition), None)
        if barrier is not None:
            flush()
            op = dag.op(barrier)
            phys_q = tuple(l2p[q] for q in op.qubits)
            result.add_op(Operation(op.gate, phys_q, op.params, op.classical_bit, op.condition))
            mark_done(barrier)
            continue

        # Execute gates that are ready (1Q always, 2Q if qubits adjacent)
        progress = False
        for nid in front:
            op = dag.op(nid)
            if op.gate.n_qubits == 1:
                pa = l2p[op.qubits[0]]
                tracker.add_gate(op.gate, (pa,), op.params)
                if depth_aware: qubit_depth[pa] += 1
                mark_done(nid)
                progress = True
            elif op.gate.n_qubits == 2:
                pa, pb = l2p[op.qubits[0]], l2p[op.qubits[1]]
                if target.are_connected(pa, pb):
                    tracker.add_gate(op.gate, (pa, pb), op.params)
                    if depth_aware:
                        d = max(qubit_depth[pa], qubit_depth[pb]) + 1
                        qubit_depth[pa] = qubit_depth[pb] = d
                    mark_done(nid)
                    progress = True
                    stall = 0
        if progress: continue

        # Blocked: collect candidate SWAPs and check connectivity
        candidates: set[tuple[int, int]] = set()
        for nid in front:
            op = dag.op(nid)
            if op.gate.n_qubits == 2:
                pa, pb = l2p[op.qubits[0]], l2p[op.qubits[1]]
                if dist[pa][pb] < 0 or dist[pa][pb] == float('inf'):
                    raise ValueError(f"No path between qubits {pa} and {pb} - disconnected topology")
                for pq in (pa, pb):
                    for nb in target.neighbors(pq):
                        candidates.add((min(pq, nb), max(pq, nb)))

        # Extended set: future gates for lookahead scoring
        # NOTE: disable lookahead when stalled to prevent oscillation
        if stall < n:
            extended, layer = set(), set(front)
            for _ in range(lookahead_depth):
                nxt = {s for nid in layer for s in dag.successors(nid) if s not in executed and s not in extended}
                if not nxt: break
                extended |= nxt
                layer = nxt
        else:
            extended = set()

        # Pick SWAP with lowest score (sorted for determinism on ties)
        best = min(sorted(candidates), key=lambda sw: _score_swap(sw, front, list(extended), dag, l2p, p2l, dist, decay, qubit_depth))
        p0, p1 = best
        tracker.record_swap(p0, p1, -1)
        swaps_inserted += 1
        stall += 1
        if swaps_inserted > swap_budget:
            raise RuntimeError("SABRE routing exceeded swap budget")

        # Update logical <-> physical mappings after SWAP
        l0, l1 = p2l[p0], p2l[p1]
        p2l[p0], p2l[p1] = l1, l0
        if l0 != -1: l2p[l0] = p1
        if l1 != -1: l2p[l1] = p0
        if depth_aware:
            sd = max(qubit_depth[p0], qubit_depth[p1]) + 3
            qubit_depth[p0] = qubit_depth[p1] = sd
        decay[min(p0, p1)][max(p0, p1)] += 0.001

    flush()
    return finalize()
