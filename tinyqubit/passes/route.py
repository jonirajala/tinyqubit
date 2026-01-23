"""Qubit routing using SABRE algorithm."""

from ..ir import Circuit, Operation, Gate
from ..target import Target
from ..tracker import QubitTracker, PendingSwap


def _build_dag(circuit: Circuit) -> tuple[list[list[int]], list[int]]:
    """Build dependency DAG. Returns (successors, in_degree)."""
    n = len(circuit.ops)
    successors: list[list[int]] = [[] for _ in range(n)]
    in_degree = [0] * n
    last_qubit: dict[int, int] = {}
    last_cbit: dict[int, int] = {}

    for i, op in enumerate(circuit.ops):
        for q in op.qubits:
            if q in last_qubit:
                successors[last_qubit[q]].append(i)
                in_degree[i] += 1
            last_qubit[q] = i
        if op.condition and op.condition[0] in last_cbit:
            successors[last_cbit[op.condition[0]]].append(i)
            in_degree[i] += 1
        if op.gate == Gate.MEASURE and op.classical_bit is not None:
            last_cbit[op.classical_bit] = i

    return successors, in_degree


def _score_swap(swap: tuple[int, int], front: list[int], extended: list[int],
                circuit: Circuit, l2p: list[int], p2l: list[int],
                dist: list[list[int]], decay: list[list[float]]) -> float:
    """Score a SWAP candidate. Lower is better."""
    p0, p1 = swap
    new_l2p = l2p.copy()
    if p2l[p0] != -1: new_l2p[p2l[p0]] = p1
    if p2l[p1] != -1: new_l2p[p2l[p1]] = p0

    score = decay[min(p0, p1)][max(p0, p1)]
    for gate_idx, weight in [(g, 1.0) for g in front] + [(g, 0.5) for g in extended]:
        op = circuit.ops[gate_idx]
        if op.gate.n_qubits == 2:
            d = dist[new_l2p[op.qubits[0]]][new_l2p[op.qubits[1]]]
            if d >= 0: score += weight * d
    return score


def route(circuit: Circuit, target: Target) -> Circuit:
    """Route circuit using SABRE algorithm.

    SABRE processes gates in dependency order, inserting SWAPs when 2Q gates
    need non-adjacent qubits. Uses lookahead scoring to pick SWAPs that help
    both current and future gates.
    """
    if target.n_qubits < circuit.n_qubits:
        raise ValueError(f"Target has {target.n_qubits} qubits but circuit needs {circuit.n_qubits}")

    tracker = QubitTracker(target.n_qubits)
    result = Circuit(target.n_qubits)

    if target.is_all_to_all() or not circuit.ops:
        result.ops = circuit.ops.copy()
        result._tracker = tracker
        return result

    successors, in_degree = _build_dag(circuit)
    executed = [False] * len(circuit.ops)
    dist = target.all_pairs_distances()
    n = target.n_qubits
    decay = [[0.0] * n for _ in range(n)]  # prevents repeated swaps on same edge
    l2p = list(range(circuit.n_qubits))    # logical -> physical qubit mapping
    p2l = [-1] * n                          # physical -> logical (-1 = unmapped)
    for i in range(circuit.n_qubits): p2l[i] = i

    def mark_done(idx: int):
        executed[idx] = True
        for s in successors[idx]: in_degree[s] -= 1

    def flush():
        for p in tracker.flush():
            if isinstance(p, PendingSwap): result.ops.append(Operation(Gate.SWAP, (p.phys_a, p.phys_b)))
            else: result.ops.append(Operation(p.gate, p.phys_qubits, p.params))

    while True:
        # Front layer: gates with all dependencies satisfied
        front = [i for i, d in enumerate(in_degree) if d == 0 and not executed[i]]
        if not front: break

        # Barriers (MEASURE/RESET/conditional) must flush pending ops first
        barrier = next((i for i in front if circuit.ops[i].gate in (Gate.MEASURE, Gate.RESET) or circuit.ops[i].condition), None)
        if barrier is not None:
            flush()
            op = circuit.ops[barrier]
            phys_q = tuple(l2p[q] for q in op.qubits)
            result.ops.append(Operation(op.gate, phys_q, op.params, op.classical_bit, op.condition))
            mark_done(barrier)
            continue

        # Execute gates that are ready (1Q always, 2Q if qubits adjacent)
        progress = False
        for i in front:
            op = circuit.ops[i]
            if op.gate.n_qubits == 1:
                tracker.add_gate(op.gate, (l2p[op.qubits[0]],), op.params)
                mark_done(i)
                progress = True
            elif op.gate.n_qubits == 2:
                pa, pb = l2p[op.qubits[0]], l2p[op.qubits[1]]
                if target.are_connected(pa, pb):
                    tracker.add_gate(op.gate, (pa, pb), op.params)
                    mark_done(i)
                    progress = True
        if progress: continue

        # Blocked: need SWAP. First check topology is connected.
        for i in front:
            op = circuit.ops[i]
            if op.gate.n_qubits == 2:
                pa, pb = l2p[op.qubits[0]], l2p[op.qubits[1]]
                if dist[pa][pb] < 0:
                    raise ValueError(f"No path between qubits {pa} and {pb} - disconnected topology")

        # Extended set: future gates for lookahead scoring
        extended, layer = set(), set(front)
        for _ in range(20):
            nxt = {s for i in layer for s in successors[i] if not executed[s] and s not in extended}
            if not nxt: break
            extended |= nxt
            layer = nxt

        # Candidate SWAPs: edges touching qubits involved in blocked 2Q gates
        candidates = set()
        for i in front:
            op = circuit.ops[i]
            if op.gate.n_qubits == 2:
                for q in op.qubits:
                    pq = l2p[q]
                    for nb in target.neighbors(pq):
                        candidates.add((min(pq, nb), max(pq, nb)))

        # Pick SWAP with lowest score (sorted for determinism on ties)
        best = min(sorted(candidates), key=lambda sw: _score_swap(sw, front, list(extended), circuit, l2p, p2l, dist, decay))
        p0, p1 = best
        tracker.record_swap(p0, p1, -1)

        # Update logical <-> physical mappings after SWAP
        l0, l1 = p2l[p0], p2l[p1]
        p2l[p0], p2l[p1] = l1, l0
        if l0 != -1: l2p[l0] = p1
        if l1 != -1: l2p[l1] = p0
        decay[min(p0, p1)][max(p0, p1)] += 0.001

    flush()
    result._tracker = tracker
    return result
