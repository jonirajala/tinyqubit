"""
Topology generators for routing benchmarks.

Provides common quantum hardware coupling maps:
- line_topology: Linear chain (1D)
- grid_topology: 2D mesh/grid
- heavy_hex_topology: IBM's heavy-hexagon topology
- all_to_all_topology: Fully connected (IonQ-style)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Gate
from tinyqubit.target import Target

try:
    from qiskit.transpiler import CouplingMap
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def line_topology(n: int) -> frozenset[tuple[int, int]]:
    """
    Linear chain topology: 0-1-2-...-n-1

    Args:
        n: Number of qubits

    Returns:
        Frozenset of edges (bidirectional)
    """
    edges = set()
    for i in range(n - 1):
        edges.add((i, i + 1))
        edges.add((i + 1, i))
    return frozenset(edges)


def grid_topology(rows: int, cols: int) -> frozenset[tuple[int, int]]:
    """
    2D grid/mesh topology.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        Frozenset of edges (bidirectional)

    Qubit numbering: row-major, i.e., qubit(r,c) = r * cols + c
    """
    edges = set()

    def qubit(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            # Right neighbor
            if c < cols - 1:
                edges.add((qubit(r, c), qubit(r, c + 1)))
                edges.add((qubit(r, c + 1), qubit(r, c)))
            # Down neighbor
            if r < rows - 1:
                edges.add((qubit(r, c), qubit(r + 1, c)))
                edges.add((qubit(r + 1, c), qubit(r, c)))

    return frozenset(edges)


def heavy_hex_topology(n: int) -> tuple[frozenset[tuple[int, int]], int]:
    """
    IBM heavy-hexagon-like topology.

    This is a simplified version that captures the key characteristic:
    alternating rows of data qubits and ancilla qubits, with limited
    connectivity (each data qubit connected to ~2 neighbors).

    Args:
        n: Minimum number of qubits (actual count returned as second value)

    Returns:
        (edges, actual_n_qubits) - frozenset of edges and the actual qubit count

    Pattern for n=8:
        0 - 1 - 2
            |   |
            3   4
            |   |
        5 - 6 - 7

    For larger n, extends this pattern.
    """
    if n < 5:
        # Fall back to line for small n
        return line_topology(n), n

    edges = set()

    # For simplicity, construct a heavy-hex-like pattern
    # with rows: top data, middle ancilla, bottom data

    # Calculate layout that fits at least n qubits
    cols = max(2, (n + 2) // 3)
    n_top = cols
    n_mid = cols - 1  # One fewer ancilla
    n_bot = cols
    actual_n = n_top + n_mid + n_bot

    # Qubit indices:
    # top row: 0 to n_top-1
    # mid row: n_top to n_top + n_mid - 1
    # bot row: n_top + n_mid to end

    top_start = 0
    mid_start = n_top
    bot_start = n_top + n_mid

    # Top row horizontal connections
    for i in range(n_top - 1):
        edges.add((top_start + i, top_start + i + 1))
        edges.add((top_start + i + 1, top_start + i))

    # Bottom row horizontal connections
    for i in range(n_bot - 1):
        edges.add((bot_start + i, bot_start + i + 1))
        edges.add((bot_start + i + 1, bot_start + i))

    # Vertical connections through ancilla
    # Each ancilla connects to one top and one bottom qubit
    for i in range(n_mid):
        # Connect top[i+1] to mid[i] and mid[i] to bot[i+1]
        mid_q = mid_start + i
        top_q = top_start + i + 1
        bot_q = bot_start + i + 1

        edges.add((top_q, mid_q))
        edges.add((mid_q, top_q))
        edges.add((bot_q, mid_q))
        edges.add((mid_q, bot_q))

    return frozenset(edges), actual_n


def all_to_all_topology(n: int) -> frozenset[tuple[int, int]]:
    """
    Fully connected topology (IonQ-style trapped ion).

    Args:
        n: Number of qubits

    Returns:
        Frozenset of edges (bidirectional) - all pairs connected
    """
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            edges.add((i, j))
            edges.add((j, i))
    return frozenset(edges)


def create_target(name: str, n_qubits: int, edges: frozenset[tuple[int, int]]) -> Target:
    """
    Create a Target from topology edges.

    Args:
        name: Name for the target
        n_qubits: Number of qubits
        edges: Topology edges

    Returns:
        Target with standard basis gates
    """
    return Target(
        n_qubits=n_qubits,
        edges=edges,
        basis_gates=frozenset({Gate.CX, Gate.H, Gate.SWAP, Gate.RZ, Gate.RX, Gate.RY}),
        name=name
    )


def create_coupling_map(edges: frozenset[tuple[int, int]]) -> "CouplingMap | None":
    """
    Create a Qiskit CouplingMap from topology edges.

    Args:
        edges: Topology edges

    Returns:
        CouplingMap or None if Qiskit not available
    """
    if not HAS_QISKIT:
        return None

    # CouplingMap expects list of [control, target] pairs
    # Use unique directed edges
    edge_list = [(a, b) for a, b in edges if a < b]
    return CouplingMap(couplinglist=edge_list)


# Pre-defined topologies for benchmarks
# Format: name -> (n_qubits or None, edge_generator, is_heavy_hex)
# For heavy_hex, n_qubits is None as it's determined by the generator
TOPOLOGIES = {
    "line_5": (5, lambda: line_topology(5), False),
    "line_10": (10, lambda: line_topology(10), False),
    "grid_2x3": (6, lambda: grid_topology(2, 3), False),
    "grid_3x3": (9, lambda: grid_topology(3, 3), False),
    "grid_4x4": (16, lambda: grid_topology(4, 4), False),
    "heavy_hex_8": (None, lambda: heavy_hex_topology(8), True),
    "heavy_hex_11": (None, lambda: heavy_hex_topology(11), True),
    "all_to_all_5": (5, lambda: all_to_all_topology(5), False),
    "all_to_all_8": (8, lambda: all_to_all_topology(8), False),
}


def get_topology(name: str) -> tuple[int, frozenset[tuple[int, int]]]:
    """
    Get a pre-defined topology by name.

    Args:
        name: Topology name (e.g., "line_5", "grid_3x3", "heavy_hex_8")

    Returns:
        (n_qubits, edges)
    """
    if name not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {name}. Available: {list(TOPOLOGIES.keys())}")
    n_qubits, edge_fn, is_heavy_hex = TOPOLOGIES[name]
    if is_heavy_hex:
        edges, actual_n = edge_fn()
        return actual_n, edges
    else:
        return n_qubits, edge_fn()
