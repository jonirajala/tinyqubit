"""
Hardware target definitions.

Contains:
    - Target: Hardware description (n_qubits, edges, basis_gates)
    - Helper methods: are_connected, shortest_path, is_all_to_all, distance

Note: Edges are treated as UNDIRECTED. Edge (a,b) allows 2Q gates in both
directions. For hardware with directional CX, additional handling would be
needed in the compiler (not currently implemented).
"""

from dataclasses import dataclass, field
from collections import deque
from .ir import Gate


@dataclass
class Target:
    """Describes quantum hardware constraints. Edges are undirected."""
    n_qubits: int
    edges: frozenset[tuple[int, int]]
    basis_gates: frozenset[Gate]
    name: str = ""
    _adj: dict[int, list[int]] = field(default_factory=dict, repr=False, compare=False)
    _dist: dict[tuple[int, int], int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        # Validate edges
        for a, b in self.edges:
            if not (0 <= a < self.n_qubits and 0 <= b < self.n_qubits):
                raise ValueError(f"Edge ({a},{b}) has invalid qubit index for {self.n_qubits}-qubit target")
            if a == b:
                raise ValueError(f"Self-loop ({a},{a}) not allowed")

        # Build adjacency list, pre-sorted for determinism
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_qubits)}
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        object.__setattr__(self, '_adj', {k: sorted(set(v)) for k, v in adj.items()})
        object.__setattr__(self, '_dist', {})

    def are_connected(self, q0: int, q1: int) -> bool:
        """Check if two qubits can do a 2Q gate directly."""
        return (q0, q1) in self.edges or (q1, q0) in self.edges

    def distance(self, q0: int, q1: int) -> int:
        """SWAP distance between qubits (cached). Returns 0 if same, -1 if unreachable."""
        if q0 == q1: return 0
        key = (min(q0, q1), max(q0, q1))
        if key not in self._dist:
            path = self.shortest_path(q0, q1)
            self._dist[key] = len(path) - 1 if path else -1
        return self._dist[key]

    def shortest_path(self, q0: int, q1: int) -> list[int]:
        """Find shortest path between qubits using BFS. Returns [q0, ..., q1] or [] if unreachable."""
        if q0 == q1: return [q0]
        if self.are_connected(q0, q1): return [q0, q1]

        prev: dict[int, int | None] = {q0: None}
        queue = deque([q0])

        while queue:
            current = queue.popleft()
            for neighbor in self._adj[current]:
                if neighbor in prev: continue
                prev[neighbor] = current
                if neighbor == q1:
                    path, node = [], q1
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return path[::-1]
                queue.append(neighbor)
        return []

    def is_all_to_all(self) -> bool:
        """Check if every qubit pair is connected (no routing needed)."""
        # Normalize to undirected pairs to handle duplicate edges
        pairs = {(min(a, b), max(a, b)) for a, b in self.edges if a != b}
        expected = self.n_qubits * (self.n_qubits - 1) // 2
        return len(pairs) >= expected
