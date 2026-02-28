"""Problem-specific Hamiltonians."""
from __future__ import annotations
from ..observable import Observable, Z


def maxcut_hamiltonian(edges: list[tuple[int, ...]]) -> Observable:
    """MaxCut QAOA cost Hamiltonian: C = Σ ½(1 - Z_i Z_j) for each edge."""
    H = Observable([])
    for e in edges:
        i, j, w = (e[0], e[1], e[2]) if len(e) == 3 else (e[0], e[1], 1.0)
        H = H + w * 0.5 * (Observable([(1.0, {})]) + -1.0 * (Z(i) @ Z(j)))
    return H
