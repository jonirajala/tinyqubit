"""
Pytest fixtures for test targets and circuits.
"""
import pytest
from tinyqubit.ir import Circuit, Gate
from tinyqubit.target import Target


# =============================================================================
# Topology Factories (test utilities)
# =============================================================================

def line_topology(n: int) -> frozenset[tuple[int, int]]:
    """Linear chain: 0-1-2-3-..."""
    return frozenset((i, i + 1) for i in range(n - 1))


def grid_topology(rows: int, cols: int) -> frozenset[tuple[int, int]]:
    """2D grid topology."""
    edges = set()
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c < cols - 1: edges.add((i, i + 1))
            if r < rows - 1: edges.add((i, i + cols))
    return frozenset(edges)


def all_to_all_topology(n: int) -> frozenset[tuple[int, int]]:
    """Every qubit connected to every other."""
    return frozenset((i, j) for i in range(n) for j in range(i + 1, n))


# =============================================================================
# Test Targets
# =============================================================================

@pytest.fixture
def ibm_line_5():
    """IBM-style: 5 qubits, line topology, {RZ, X, CX} basis."""
    return Target(
        n_qubits=5,
        edges=line_topology(5),
        basis_gates=frozenset({Gate.RZ, Gate.X, Gate.CX}),
        name="ibm_line_5"
    )

@pytest.fixture
def ibm_grid_4():
    """IBM-style: 4 qubits, 2x2 grid, {RZ, X, CX} basis."""
    return Target(
        n_qubits=4,
        edges=grid_topology(2, 2),
        basis_gates=frozenset({Gate.RZ, Gate.X, Gate.CX}),
        name="ibm_grid_4"
    )

@pytest.fixture
def rigetti_line_5():
    """Rigetti-style: 5 qubits, line topology, {RX, RZ, CZ} basis."""
    return Target(
        n_qubits=5,
        edges=line_topology(5),
        basis_gates=frozenset({Gate.RX, Gate.RZ, Gate.CZ}),
        name="rigetti_line_5"
    )

@pytest.fixture
def ionq_4():
    """IonQ-style: 4 qubits, all-to-all, {RX, RY, RZ, CX} basis."""
    return Target(
        n_qubits=4,
        edges=all_to_all_topology(4),
        basis_gates=frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX}),
        name="ionq_4"
    )

@pytest.fixture
def ionq_11():
    """IonQ-style: 11 qubits, all-to-all, {RX, RY, RZ, CX} basis."""
    return Target(
        n_qubits=11,
        edges=all_to_all_topology(11),
        basis_gates=frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX}),
        name="ionq_11"
    )


# =============================================================================
# Test Circuits
# =============================================================================

@pytest.fixture
def bell_circuit():
    """Bell state: H(0), CX(0,1)."""
    return Circuit(2).h(0).cx(0, 1)

@pytest.fixture
def ghz_circuit():
    """GHZ state: H(0), CX(0,1), CX(1,2)."""
    return Circuit(3).h(0).cx(0, 1).cx(1, 2)
