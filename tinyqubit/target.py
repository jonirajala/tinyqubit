"""Hardware target definitions."""

from dataclasses import dataclass, field
from .ir import Gate


@dataclass
class Target:
    """Describes quantum hardware constraints. Edges are undirected for routing;
    when directed=True, edges define allowed CX directions for the direction pass."""
    n_qubits: int
    edges: frozenset[tuple[int, int]]
    basis_gates: frozenset[Gate]
    name: str = ""
    directed: bool = False
    edge_error: dict[tuple[int, int], float] | None = None
    duration: dict[Gate, int] | None = None
    _adj: dict[int, list[int]] = field(default_factory=dict, repr=False, compare=False)
    _all_pairs: list[list[int]] | None = field(default=None, repr=False, compare=False)
    _all_pairs_error: list[list[float]] | None = field(default=None, repr=False, compare=False)

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
        if self.edge_error is not None:
            normalized = {(min(a, b), max(a, b)) for a, b in self.edges}
            provided = {(min(a, b), max(a, b)) for a, b in self.edge_error}
            if provided != normalized:
                raise ValueError(f"edge_error must match edges exactly (missing={normalized - provided}, extra={provided - normalized})")

    def are_connected(self, q0: int, q1: int) -> bool:
        """Check if two qubits can do a 2Q gate directly."""
        return (q0, q1) in self.edges or (q1, q0) in self.edges

    def neighbors(self, qubit: int) -> list[int]:
        """Return sorted list of qubits adjacent to given qubit."""
        return self._adj[qubit]

    def is_all_to_all(self) -> bool:
        """Check if every qubit pair is connected (no routing needed)."""
        # Normalize to undirected pairs to handle duplicate edges
        pairs = {(min(a, b), max(a, b)) for a, b in self.edges if a != b}
        expected = self.n_qubits * (self.n_qubits - 1) // 2
        return len(pairs) >= expected

    def all_pairs_distances(self) -> list[list[int]]:
        """Compute all-pairs shortest path distances using Floyd-Warshall. Cached."""
        if self._all_pairs is not None:
            return self._all_pairs

        n = self.n_qubits
        INF = n + 1
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for a, b in self.edges:
            dist[a][b] = dist[b][a] = 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        for i in range(n):
            for j in range(n):
                if dist[i][j] == INF:
                    dist[i][j] = -1

        object.__setattr__(self, '_all_pairs', dist)
        return dist

    def all_pairs_error_costs(self) -> list[list[float]]:
        """All-pairs shortest paths weighted by edge_error. Cached."""
        if self._all_pairs_error is not None:
            return self._all_pairs_error
        if self.edge_error is None:
            raise ValueError("edge_error not set on this target")

        n = self.n_qubits
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0.0
        for (a, b), err in self.edge_error.items():
            dist[a][b] = dist[b][a] = err

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        object.__setattr__(self, '_all_pairs_error', dist)
        return dist

    def validate(self, circuit) -> list[str]:
        """Check circuit against target constraints. Returns list of error strings (empty = valid)."""
        errors = []
        if circuit.n_qubits > self.n_qubits:
            errors.append(f"Circuit has {circuit.n_qubits} qubits but target has {self.n_qubits}")
        allowed = self.basis_gates | {Gate.MEASURE, Gate.RESET}
        basis_names = ", ".join(sorted(g.name for g in self.basis_gates))
        for op in circuit.ops:
            if op.gate not in allowed:
                errors.append(f"Gate {op.gate.name} on qubits {op.qubits} not in target basis {{{basis_names}}}")
            if op.gate.n_qubits >= 2 and not self.are_connected(*op.qubits[:2]):
                errors.append(f"Gate {op.gate.name}{op.qubits} not connected in target topology")
            if self.directed and op.gate == Gate.CX and op.qubits not in self.edges:
                errors.append(f"CX{op.qubits} wrong direction for directed target")
        return errors
