"""
Hardware target definitions.

Contains:
    - Target: Hardware description (n_qubits, edges, basis_gates)
    - Helper methods: are_connected, shortest_path, is_all_to_all
"""

from dataclasses import dataclass, field
from collections import deque
from .ir import Gate


@dataclass
class Target:
    """Describes quantum hardware constraints."""
    n_qubits: int
    edges: frozenset[tuple[int, int]]
    basis_gates: frozenset[Gate]
    name: str = ""
    _adj: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Build adjacency list once, pre-sorted for determinism
        adj = {i: [] for i in range(self.n_qubits)}
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        self._adj = {k: sorted(v) for k, v in adj.items()}

    def are_connected(self, q0: int, q1: int) -> bool:
        """Check if two qubits can do a 2Q gate directly."""
        return (q0, q1) in self.edges or (q1, q0) in self.edges

    def shortest_path(self, q0: int, q1: int) -> list[int]:
        """Find shortest path between qubits using BFS. Returns [q0, ..., q1]."""
        if q0 == q1: return [q0]
        if self.are_connected(q0, q1): return [q0, q1]

        # BFS with predecessor tracking
        prev = {q0: None}
        queue = deque([q0])

        while queue:
            current = queue.popleft()
            for neighbor in self._adj[current]:
                if neighbor in prev:
                    continue
                prev[neighbor] = current
                if neighbor == q1:
                    # Reconstruct path
                    path = []
                    node = q1
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return path[::-1]
                queue.append(neighbor)

        return []

    def is_all_to_all(self) -> bool:
        """Check if every qubit pair is connected (no routing needed)."""
        expected = self.n_qubits * (self.n_qubits - 1) // 2
        return len(self.edges) >= expected
