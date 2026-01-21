"""
Metrics tests for routing and optimization passes.

Tests:
    - Routing only adds SWAPs (preserves original 2Q gates)
    - Optimization never increases gate count
    - Depth never increases after optimization
"""
import pytest
from math import pi

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.target import Target
from tinyqubit.passes.route import route
from tinyqubit.passes.optimize import optimize
from tinyqubit.passes.decompose import decompose
from tinyqubit.compile import transpile


# =============================================================================
# Helpers
# =============================================================================

def line_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, i + 1) for i in range(n - 1))


def count_gates(circuit: Circuit) -> int:
    """Count total gates (excluding MEASURE)."""
    return sum(1 for op in circuit.ops if op.gate != Gate.MEASURE)


def count_2q_gates(circuit: Circuit) -> int:
    """Count 2-qubit gates."""
    return sum(1 for op in circuit.ops if op.gate.n_qubits == 2)


def count_swaps(circuit: Circuit) -> int:
    """Count SWAP gates."""
    return sum(1 for op in circuit.ops if op.gate == Gate.SWAP)


def circuit_depth(circuit: Circuit) -> int:
    """Calculate circuit depth (longest path)."""
    if not circuit.ops:
        return 0
    free = {}
    for op in circuit.ops:
        t = max((free.get(q, 0) for q in op.qubits), default=0) + 1
        for q in op.qubits:
            free[q] = t
    return max(free.values()) if free else 0


# =============================================================================
# Routing metrics tests
# =============================================================================

class TestRoutingMetrics:
    """Tests that routing preserves logical gates and only adds SWAPs."""

    @pytest.fixture
    def line_target(self):
        return Target(
            n_qubits=5,
            edges=line_topology(5),
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX, Gate.SWAP}),
            name="line_5"
        )

    def test_routing_preserves_gate_types(self, line_target):
        """Routing preserves all original gate types (H, CX, etc.)."""
        c = Circuit(5).h(0).cx(0, 1).cx(1, 2).h(2)

        routed = route(c, line_target)

        original_gates = {op.gate for op in c.ops}
        routed_gates = {op.gate for op in routed.ops if op.gate != Gate.SWAP}

        assert original_gates == routed_gates, (
            f"Gate types changed: {original_gates} -> {routed_gates}"
        )

    def test_routing_only_adds_swaps(self, line_target):
        """Routing only adds SWAP gates, no other gates."""
        c = Circuit(5).h(0).cx(0, 4)  # Non-adjacent, needs routing

        original_gates = list(c.ops)
        routed = route(c, line_target)

        # Count non-SWAP gates - should equal original
        routed_non_swaps = [op for op in routed.ops if op.gate != Gate.SWAP]
        assert len(routed_non_swaps) == len(original_gates), (
            f"Non-SWAP count changed: {len(original_gates)} -> {len(routed_non_swaps)}"
        )

    def test_routing_increases_2q_gates(self, line_target):
        """Routing may add SWAPs, increasing 2Q count."""
        c = Circuit(5).cx(0, 4)  # Far apart, needs SWAPs

        original_2q = count_2q_gates(c)
        routed = route(c, line_target)
        routed_2q = count_2q_gates(routed)

        assert routed_2q >= original_2q, "Routing should not remove 2Q gates"
        assert count_swaps(routed) > 0, "Expected SWAPs for non-adjacent qubits"

    def test_no_swaps_for_adjacent(self, line_target):
        """No SWAPs needed for adjacent qubits."""
        c = Circuit(5).cx(0, 1).cx(1, 2).cx(2, 3)

        routed = route(c, line_target)

        assert count_swaps(routed) == 0, "Adjacent qubits should not need SWAPs"

    def test_all_to_all_no_swaps(self):
        """All-to-all connectivity never needs SWAPs."""
        edges = frozenset((i, j) for i in range(4) for j in range(i + 1, 4))
        target = Target(
            n_qubits=4,
            edges=edges,
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
            name="all_to_all"
        )

        c = Circuit(4).cx(0, 3).cx(1, 2).cx(0, 2)

        routed = route(c, target)

        assert count_swaps(routed) == 0, "All-to-all should never need SWAPs"


# =============================================================================
# Optimization metrics tests
# =============================================================================

class TestOptimizationMetrics:
    """Tests that optimization reduces or maintains gate counts."""

    def test_optimization_cancels_redundant(self):
        """Optimization cancels X·X, H·H, CX·CX patterns."""
        c = Circuit(2)
        c.ops = [
            Operation(Gate.X, (0,)),
            Operation(Gate.X, (0,)),  # Cancels with above
            Operation(Gate.H, (1,)),
            Operation(Gate.H, (1,)),  # Cancels with above
        ]

        optimized = optimize(c)

        assert count_gates(optimized) == 0, "Should cancel all redundant gates"

    def test_optimization_never_increases_gates(self):
        """Optimization never increases total gate count."""
        # Various circuits
        circuits = [
            Circuit(2).h(0).cx(0, 1),
            Circuit(3).h(0).cx(0, 1).cx(1, 2).h(2),
            Circuit(2).rz(0, pi/4).rz(0, pi/4),  # Should merge
            Circuit(2).s(0).s(0),  # Should become Z
        ]

        for c in circuits:
            before = count_gates(c)
            after = count_gates(optimize(c))
            assert after <= before, f"Optimization increased gates: {before} -> {after}"

    def test_optimization_merges_rotations(self):
        """Optimization merges consecutive rotations."""
        c = Circuit(1)
        c.ops = [
            Operation(Gate.RZ, (0,), (pi/4,)),
            Operation(Gate.RZ, (0,), (pi/4,)),
        ]

        optimized = optimize(c)

        assert count_gates(optimized) == 1, "Should merge into single RZ"
        assert optimized.ops[0].gate == Gate.RZ
        assert abs(optimized.ops[0].params[0] - pi/2) < 1e-10

    def test_optimization_clifford_merge(self):
        """Optimization merges Clifford gates (S·S → Z)."""
        c = Circuit(1)
        c.ops = [Operation(Gate.S, (0,)), Operation(Gate.S, (0,))]

        optimized = optimize(c)

        assert count_gates(optimized) == 1
        assert optimized.ops[0].gate == Gate.Z

    def test_optimization_preserves_needed_gates(self):
        """Optimization preserves gates that cannot be reduced."""
        c = Circuit(2).h(0).cx(0, 1).h(1)

        before = count_gates(c)
        after = count_gates(optimize(c))

        assert after == before, "Should preserve non-redundant gates"


# =============================================================================
# Depth metrics tests
# =============================================================================

class TestDepthMetrics:
    """Tests for circuit depth calculations and optimization."""

    def test_depth_single_ops(self):
        """Single operations have depth 1."""
        c = Circuit(2).h(0)
        assert circuit_depth(c) == 1

    def test_depth_parallel_ops(self):
        """Parallel operations don't increase depth."""
        c = Circuit(3).h(0).h(1).h(2)  # All parallel
        assert circuit_depth(c) == 1

    def test_depth_serial_ops(self):
        """Serial operations increase depth."""
        c = Circuit(1).h(0).x(0).z(0)  # All on same qubit
        assert circuit_depth(c) == 3

    def test_depth_2q_gates(self):
        """2Q gates affect depth of both qubits."""
        c = Circuit(3).h(0).cx(0, 1).h(1)
        # H(0): depth 1
        # CX(0,1): depth 2 (after H)
        # H(1): depth 3 (after CX)
        assert circuit_depth(c) == 3

    def test_optimization_maintains_or_reduces_depth(self):
        """Optimization doesn't increase depth."""
        c = Circuit(2)
        c.ops = [
            Operation(Gate.H, (0,)),
            Operation(Gate.H, (0,)),  # Cancels
            Operation(Gate.X, (1,)),
        ]

        before = circuit_depth(c)
        after = circuit_depth(optimize(c))

        assert after <= before, f"Depth increased: {before} -> {after}"


# =============================================================================
# Full transpile metrics tests
# =============================================================================

class TestTranspileMetrics:
    """End-to-end metrics tests for full transpilation."""

    @pytest.fixture
    def line_target(self):
        return Target(
            n_qubits=5,
            edges=line_topology(5),
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
            name="line_5"
        )

    def test_transpile_respects_basis(self, line_target):
        """Transpiled circuit only uses basis gates."""
        c = Circuit(5).h(0).s(1).t(2).cx(0, 1)

        result = transpile(c, line_target)

        for op in result.ops:
            assert op.gate in line_target.basis_gates or op.gate == Gate.MEASURE, (
                f"Gate {op.gate.name} not in basis"
            )

    def test_transpile_respects_connectivity(self, line_target):
        """Transpiled circuit 2Q gates only on connected qubits."""
        c = Circuit(5).cx(0, 4)  # Non-adjacent

        result = transpile(c, line_target)

        for op in result.ops:
            if op.gate.n_qubits == 2:
                q0, q1 = op.qubits
                is_connected = (q0, q1) in line_target.edges or (q1, q0) in line_target.edges
                assert is_connected, f"2Q gate on non-connected qubits: {q0}, {q1}"
