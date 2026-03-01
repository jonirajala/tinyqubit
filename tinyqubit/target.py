"""
Hardware target definitions.

Contains:
    - Target: Hardware description (n_qubits, edges, basis_gates)
    - Helper methods: are_connected, shortest_path, is_all_to_all, distance

Note: Routing always treats edges as undirected (both directions valid for
adjacency). When directed=True, edges define allowed CX directions and the
fix_direction pass inserts H-sandwiches for reversed CX gates.
"""

from dataclasses import dataclass, field
from collections import deque
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
    virtual_gates: frozenset[Gate] = frozenset()
    _adj: dict[int, list[int]] = field(default_factory=dict, repr=False, compare=False)
    _dist: dict[tuple[int, int], int] = field(default_factory=dict, repr=False, compare=False)
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
        object.__setattr__(self, '_dist', {})

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

    def expected_error(self, circuit) -> float:
        """Sum of edge_error for 2Q gates not in virtual_gates."""
        if self.edge_error is None:
            raise ValueError("edge_error not set on this target")
        return sum(self.edge_error.get((min(op.qubits), max(op.qubits)), 0.0)
                   for op in circuit.ops if op.gate.n_qubits == 2 and op.gate not in self.virtual_gates)


def validate(circuit, target: Target) -> list[str]:
    """Check circuit against target constraints. Returns list of error strings (empty = valid)."""
    errors = []
    if circuit.n_qubits > target.n_qubits:
        errors.append(f"Circuit has {circuit.n_qubits} qubits but target has {target.n_qubits}")
    allowed = target.basis_gates | {Gate.MEASURE, Gate.RESET}
    basis_names = ", ".join(sorted(g.name for g in target.basis_gates))
    for op in circuit.ops:
        if op.gate not in allowed:
            errors.append(f"Gate {op.gate.name} on qubits {op.qubits} not in target basis {{{basis_names}}}")
        if op.gate.n_qubits >= 2 and not target.are_connected(*op.qubits[:2]):
            errors.append(f"Gate {op.gate.name}{op.qubits} not connected in target topology")
        if target.directed and op.gate == Gate.CX and op.qubits not in target.edges:
            errors.append(f"CX{op.qubits} wrong direction for directed target")
    return errors


# Built-in hardware targets -------

_IBM_BASIS = frozenset({Gate.SX, Gate.RZ, Gate.CX})
_IBM_HERON_BASIS = frozenset({Gate.SX, Gate.RZ, Gate.ECR})
_IONQ_BASIS = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX})
_CZ_BASIS = frozenset({Gate.RX, Gate.RZ, Gate.CZ})

def _all_to_all(n):
    return frozenset((i, j) for i in range(n) for j in range(i + 1, n))

# IBM Eagle r3 heavy-hex coupling map (127Q, directed CX)
# Source: qiskit-ibm-runtime FakeBrisbane conf_brisbane.json
_IBM_EAGLE_EDGES = frozenset({
    (1,0), (2,1), (3,2), (4,3), (4,5), (4,15), (6,5), (6,7),
    (7,8), (8,9), (10,9), (10,11), (11,12), (12,17), (13,12), (14,0),
    (14,18), (15,22), (16,8), (16,26), (17,30), (18,19), (20,19), (20,33),
    (21,20), (21,22), (22,23), (24,23), (24,34), (25,24), (26,25), (27,26),
    (28,27), (28,29), (28,35), (30,29), (30,31), (31,32), (32,36), (33,39),
    (34,43), (35,47), (36,51), (37,38), (39,38), (40,39), (40,41), (41,53),
    (42,41), (42,43), (43,44), (44,45), (46,45), (46,47), (48,47), (48,49),
    (50,49), (50,51), (52,37), (52,56), (53,60), (54,45), (54,64), (55,49),
    (55,68), (56,57), (57,58), (58,59), (58,71), (59,60), (60,61), (62,61),
    (62,63), (62,72), (63,64), (65,64), (65,66), (67,66), (67,68), (69,68),
    (69,70), (73,66), (74,70), (74,89), (75,90), (76,75), (77,71), (77,76),
    (77,78), (79,78), (79,80), (80,81), (81,72), (81,82), (82,83), (83,92),
    (84,83), (85,73), (85,84), (85,86), (86,87), (87,88), (88,89), (91,79),
    (92,102), (93,87), (93,106), (94,90), (94,95), (95,96), (97,96), (97,98),
    (98,91), (99,98), (100,99), (100,110), (101,100), (101,102), (102,103), (104,103),
    (105,104), (105,106), (107,106), (108,107), (108,112), (109,96), (110,118), (111,104),
    (112,126), (113,114), (114,109), (114,115), (116,115), (116,117), (117,118), (118,119),
    (120,119), (121,120), (122,111), (122,121), (122,123), (124,123), (125,124), (125,126),
})

# Rigetti Ankaa-2 octagonal lattice (84Q, undirected CZ)
# NOTE: qubits 42 and 48 are absent from the topology
_RIGETTI_ANKAA_EDGES = frozenset({
    (0,1), (0,7), (1,2), (1,8), (2,3), (2,9), (3,4), (3,10),
    (4,5), (4,11), (5,6), (5,12), (6,13), (7,8), (7,14), (8,9),
    (8,15), (9,10), (9,16), (10,11), (11,12), (11,18), (12,13), (12,19),
    (13,20), (14,15), (14,21), (15,16), (15,22), (16,17), (16,23), (17,18),
    (17,24), (18,19), (18,25), (19,20), (19,26), (20,27), (21,22), (21,28),
    (22,23), (22,29), (23,24), (23,30), (24,25), (24,31), (25,26), (25,32),
    (26,27), (26,33), (28,29), (28,35), (29,30), (29,36), (30,31), (30,37),
    (31,38), (32,33), (32,39), (33,34), (33,40), (34,41), (35,36), (36,37),
    (36,43), (37,38), (37,44), (38,39), (38,45), (39,40), (39,46), (40,41),
    (40,47), (43,50), (44,45), (44,51), (45,46), (45,52), (46,47), (46,53),
    (47,54), (49,50), (49,56), (50,51), (50,57), (51,52), (51,58), (52,53),
    (52,59), (53,54), (53,60), (54,55), (54,61), (55,62), (56,57), (56,63),
    (57,58), (57,64), (58,59), (58,65), (59,60), (59,66), (60,61), (60,67),
    (61,62), (61,68), (62,69), (63,64), (63,70), (64,65), (64,71), (65,66),
    (65,72), (66,73), (67,68), (67,74), (68,69), (68,75), (69,76), (70,71),
    (70,77), (71,72), (71,78), (72,73), (72,79), (73,74), (73,80), (74,75),
    (74,81), (75,76), (75,82), (76,83), (77,78), (78,79), (79,80), (80,81),
    (81,82), (82,83),
})

# NOTE: These are offline reference topologies for testing. Brisbane/Osaka/Kyoto are retired.
# For live backends, use ibm_target() from tinyqubit.export.backends.ibm_native.
IBM_EAGLE_R3 = Target(n_qubits=127, edges=_IBM_EAGLE_EDGES, basis_gates=_IBM_BASIS, name="ibm_eagle_r3", directed=True)
IBM_BRISBANE = IBM_EAGLE_R3  # retired 2025-11-03
IBM_OSAKA = IBM_EAGLE_R3     # retired 2024-08-13
IBM_KYOTO = IBM_EAGLE_R3     # retired 2024-09-05
IBM_TORINO = IBM_EAGLE_R3    # live but topology is stale — use ibm_target("ibm_torino") instead
IONQ_HARMONY = Target(n_qubits=11, edges=_all_to_all(11), basis_gates=_IONQ_BASIS, name="ionq_harmony")
IONQ_ARIA = Target(n_qubits=25, edges=_all_to_all(25), basis_gates=_IONQ_BASIS, name="ionq_aria")
RIGETTI_ANKAA = Target(n_qubits=84, edges=_RIGETTI_ANKAA_EDGES, basis_gates=_CZ_BASIS, name="rigetti_ankaa")

# IQM Garnet square lattice (20Q, undirected CZ)
# Source: qiskit-on-iqm fake_garnet.py
_IQM_GARNET_EDGES = frozenset({
    (0,1), (0,3), (1,4), (2,3), (2,7), (3,4), (3,8), (4,5),
    (4,9), (5,6), (5,10), (6,11), (7,8), (7,12), (8,9), (8,13),
    (9,10), (9,14), (10,11), (10,15), (11,16), (12,13), (13,14),
    (13,17), (14,15), (14,18), (15,16), (15,19), (17,18), (18,19),
})

# IQM Spark star topology (5Q, undirected CZ)
_IQM_SPARK_EDGES = frozenset({(0,2), (1,2), (2,3), (2,4)})

IQM_GARNET = Target(n_qubits=20, edges=_IQM_GARNET_EDGES, basis_gates=_CZ_BASIS, name="iqm_garnet")
IQM_SPARK = Target(n_qubits=5, edges=_IQM_SPARK_EDGES, basis_gates=_CZ_BASIS, name="iqm_spark")
