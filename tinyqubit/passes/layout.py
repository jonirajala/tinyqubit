"""Initial qubit layout selection: VF2 subgraph isomorphism + SabreLayout fallback."""

from ..dag import DAGCircuit
from ..target import Target


def _interaction_graph(dag: DAGCircuit) -> set[tuple[int, int]]:
    """Extract canonical pairs of logical qubits that interact via 2Q+ gates."""
    edges: set[tuple[int, int]] = set()
    for op in dag._ops.values():
        qs = op.qubits
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                edges.add((min(qs[i], qs[j]), max(qs[i], qs[j])))
    return edges


def _vf2_layout(edges: set[tuple[int, int]], n_logical: int,
                target: Target) -> list[int] | None:
    """VF2 backtracking subgraph isomorphism. Returns layout or None."""
    if not edges:
        return list(range(n_logical))

    ig_adj: dict[int, set[int]] = {i: set() for i in range(n_logical)}
    for a, b in edges:
        ig_adj[a].add(b); ig_adj[b].add(a)

    active = sorted((q for q in range(n_logical) if ig_adj[q]),
                    key=lambda q: (-len(ig_adj[q]), q))
    free = [q for q in range(n_logical) if not ig_adj[q]]
    tgt_adj = {i: set(target.neighbors(i)) for i in range(target.n_qubits)}
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(active):
            return True
        lq = active[idx]
        mapped = [mapping[n] for n in ig_adj[lq] if n in mapping]
        cands = (set.intersection(*(tgt_adj[p] for p in mapped)) - used
                 if mapped else set(range(target.n_qubits)) - used)
        for pq in sorted(cands):
            mapping[lq] = pq; used.add(pq)
            if backtrack(idx + 1): return True
            del mapping[lq]; used.remove(pq)
        return False

    if not backtrack(0):
        return None

    remaining = sorted(set(range(target.n_qubits)) - used)
    layout = [0] * n_logical
    for lq, pq in mapping.items(): layout[lq] = pq
    for i, lq in enumerate(free): layout[lq] = remaining[i]
    return layout


def _sabre_layout(dag: DAGCircuit, target: Target) -> list[int]:
    """Forward-backward routing to find a good initial layout."""
    from .route import route

    fwd_layout = route(dag, target)._tracker.logical_to_physical[:dag.n_qubits]
    rev_dag = DAGCircuit(dag.n_qubits, dag.n_classical)
    for op in reversed(dag.topological_ops()): rev_dag.add_op(op)
    return route(rev_dag, target, initial_layout=fwd_layout)._tracker.logical_to_physical[:dag.n_qubits]


def select_layout(dag: DAGCircuit, target: Target) -> list[int] | None:
    """Select initial qubit layout. Returns layout or None for identity."""
    if target.is_all_to_all() or not dag._ops:
        return None
    edges = _interaction_graph(dag)
    if not edges:
        return None

    identity = list(range(dag.n_qubits))
    layout = _vf2_layout(edges, dag.n_qubits, target)
    if layout is not None:
        return None if layout == identity else layout

    # Skip SabreLayout if interaction graph denser than coupling map
    if len(edges) <= len(target.edges):
        layout = _sabre_layout(dag, target)
        if layout != identity: return layout
    return None
