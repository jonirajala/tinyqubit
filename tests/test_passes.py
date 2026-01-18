"""
Tests for transpiler passes: routing and optimization.
"""
import pytest
from math import pi

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.target import Target
from tinyqubit.tracker import QubitTracker
from tinyqubit.passes.route import route
from tinyqubit.passes.optimize import optimize

from conftest import line_topology, all_to_all_topology


# =============================================================================
# QubitTracker Tests
# =============================================================================

def test_tracker_initial_mapping():
    """Initial mapping is identity."""
    tracker = QubitTracker(5)
    for i in range(5):
        assert tracker.logical_to_phys(i) == i
        assert tracker.phys_to_logical(i) == i


def test_tracker_single_swap():
    """Single swap updates mapping correctly."""
    tracker = QubitTracker(3)
    tracker.record_swap(0, 1)

    # Logical 0 now at physical 1, logical 1 now at physical 0
    assert tracker.logical_to_phys(0) == 1
    assert tracker.logical_to_phys(1) == 0
    assert tracker.logical_to_phys(2) == 2

    assert tracker.phys_to_logical(0) == 1
    assert tracker.phys_to_logical(1) == 0


def test_tracker_double_swap_cancels():
    """Two swaps of same qubits returns to identity."""
    tracker = QubitTracker(3)
    tracker.record_swap(0, 1)
    tracker.record_swap(0, 1)

    for i in range(3):
        assert tracker.logical_to_phys(i) == i


def test_tracker_swap_log():
    """Swap log records operations."""
    tracker = QubitTracker(3)
    tracker.record_swap(0, 1, triggered_by=5)
    tracker.record_swap(1, 2, triggered_by=7)

    assert len(tracker.swap_log) == 2
    assert tracker.swap_log[0] == (0, 1, 5)
    assert tracker.swap_log[1] == (1, 2, 7)


def test_tracker_materialize_cancels_swaps():
    """Materialize cancels consecutive SWAP-SWAP pairs."""
    from tinyqubit.tracker import PendingSwap, PendingGate

    tracker = QubitTracker(3)
    tracker.record_swap(0, 1)
    tracker.record_swap(0, 1)  # Should cancel with previous

    materialized = tracker.materialize()
    assert len(materialized) == 0  # Both cancelled


def test_tracker_materialize_no_cancel_with_gate_between():
    """SWAPs with gate between don't cancel."""
    from tinyqubit.tracker import PendingSwap, PendingGate

    tracker = QubitTracker(3)
    tracker.record_swap(0, 1)
    tracker.add_gate(Gate.X, (0,))
    tracker.record_swap(0, 1)

    materialized = tracker.materialize()
    assert len(materialized) == 3  # SWAP, X, SWAP - no cancellation


# =============================================================================
# Routing Tests
# =============================================================================

@pytest.fixture
def line_5():
    return Target(
        n_qubits=5,
        edges=line_topology(5),
        basis_gates=frozenset({Gate.RZ, Gate.X, Gate.CX}),
        name="line_5"
    )


@pytest.fixture
def all_to_all_4():
    return Target(
        n_qubits=4,
        edges=all_to_all_topology(4),
        basis_gates=frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX}),
        name="all_to_all_4"
    )


def test_route_no_swaps_needed(line_5):
    """Adjacent qubits need no swaps."""
    c = Circuit(2).cx(0, 1)
    routed = route(c, line_5)

    assert len(routed.ops) == 1
    assert routed.ops[0].gate == Gate.CX


def test_route_inserts_swaps(line_5):
    """Non-adjacent qubits get SWAPs inserted."""
    c = Circuit(5).cx(0, 4)  # 0 and 4 are not adjacent on line
    routed = route(c, line_5)

    # Should have SWAPs + the CX
    swap_count = sum(1 for op in routed.ops if op.gate == Gate.SWAP)
    assert swap_count > 0

    # Should end with CX on adjacent qubits
    cx_ops = [op for op in routed.ops if op.gate == Gate.CX]
    assert len(cx_ops) == 1


def test_route_all_to_all_no_swaps(all_to_all_4):
    """All-to-all connectivity needs no routing."""
    c = Circuit(4).cx(0, 3).cx(1, 2)
    routed = route(c, all_to_all_4)

    # No SWAPs should be inserted
    swap_count = sum(1 for op in routed.ops if op.gate == Gate.SWAP)
    assert swap_count == 0
    assert len(routed.ops) == 2


def test_route_single_qubit_gates(line_5):
    """Single-qubit gates just get remapped."""
    c = Circuit(3).h(0).x(1).rz(2, 1.5)
    routed = route(c, line_5)

    assert len(routed.ops) == 3
    assert routed.ops[0].gate == Gate.H
    assert routed.ops[1].gate == Gate.X
    assert routed.ops[2].gate == Gate.RZ


def test_route_deterministic(line_5):
    """Same circuit always routes the same way."""
    c = Circuit(5).cx(0, 4).cx(1, 3)

    routed1 = route(c, line_5)
    routed2 = route(c, line_5)

    assert len(routed1.ops) == len(routed2.ops)
    for op1, op2 in zip(routed1.ops, routed2.ops):
        assert op1 == op2


def test_route_tracks_swaps(line_5):
    """Routed circuit has tracker with swap log."""
    c = Circuit(5).cx(0, 4)
    routed = route(c, line_5)

    assert hasattr(routed, '_tracker')
    assert len(routed._tracker.swap_log) > 0


# =============================================================================
# Optimization Tests
# =============================================================================

def test_optimize_cancel_xx():
    """X X cancels to nothing."""
    c = Circuit(1).x(0).x(0)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_cancel_hh():
    """H H cancels to nothing."""
    c = Circuit(1).h(0).h(0)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_cancel_cxcx():
    """CX CX cancels to nothing."""
    c = Circuit(2).cx(0, 1).cx(0, 1)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_cancel_swap_swap():
    """SWAP SWAP cancels to nothing."""
    c = Circuit(2).swap(0, 1).swap(0, 1)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_no_cancel_different_qubits():
    """X on different qubits doesn't cancel."""
    c = Circuit(2).x(0).x(1)
    opt = optimize(c)
    assert len(opt.ops) == 2


def test_optimize_merge_rz():
    """RZ(a) RZ(b) merges to RZ(a+b)."""
    c = Circuit(1).rz(0, 0.5).rz(0, 0.3)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.RZ
    assert abs(opt.ops[0].params[0] - 0.8) < 1e-10


def test_optimize_merge_rx():
    """RX(a) RX(b) merges to RX(a+b)."""
    c = Circuit(1).rx(0, pi/4).rx(0, pi/4)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.RX
    assert abs(opt.ops[0].params[0] - pi/2) < 1e-10


def test_optimize_merge_to_zero():
    """RZ(a) RZ(-a) cancels to nothing."""
    c = Circuit(1).rz(0, 1.5).rz(0, -1.5)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_merge_full_rotation():
    """RZ(π) RZ(π) = RZ(2π) cancels to nothing (2π ≡ 0)."""
    c = Circuit(1).rz(0, pi).rz(0, pi)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_chain_cancellation():
    """X H H X cancels to nothing (via chain)."""
    c = Circuit(1).x(0).h(0).h(0).x(0)
    opt = optimize(c)
    assert len(opt.ops) == 0


def test_optimize_preserves_order():
    """Non-cancellable gates preserve order."""
    c = Circuit(2).h(0).cx(0, 1).x(1)
    opt = optimize(c)

    assert len(opt.ops) == 3
    assert opt.ops[0].gate == Gate.H
    assert opt.ops[1].gate == Gate.CX
    assert opt.ops[2].gate == Gate.X


def test_optimize_deterministic():
    """Optimization is deterministic."""
    c = Circuit(2).x(0).h(0).h(0).cx(0, 1).cx(0, 1).rz(1, 0.5).rz(1, 0.5)

    opt1 = optimize(c)
    opt2 = optimize(c)

    assert len(opt1.ops) == len(opt2.ops)
    for op1, op2 in zip(opt1.ops, opt2.ops):
        assert op1 == op2


# =============================================================================
# Integration Tests
# =============================================================================

def test_routing_preserves_semantics_adjacent():
    """Routing adjacent qubits preserves statevector (no SWAPs needed)."""
    from tinyqubit.simulator import simulate, states_equal

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CX}),
        name="line_2"
    )

    # Bell state on adjacent qubits - no routing needed
    c = Circuit(2).h(0).cx(0, 1)
    routed = route(c, target)

    assert states_equal(simulate(c), simulate(routed))


def test_routing_preserves_semantics_with_swaps():
    """Routing with SWAPs preserves semantics (accounting for final permutation)."""
    from tinyqubit.simulator import simulate, states_equal
    import numpy as np

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX}),
        name="line_3"
    )

    # CX on non-adjacent qubits - needs routing
    c = Circuit(3).h(0).cx(0, 2)
    routed = route(c, target)

    # Get final qubit mapping
    tracker = routed._tracker
    perm = tracker.logical_to_physical

    # Simulate both
    original_state = simulate(c)
    routed_state = simulate(routed)

    # Permute original state to match physical ordering
    n = c.n_qubits
    original_reshaped = original_state.reshape([2] * n)
    permuted = np.transpose(original_reshaped, perm).reshape(-1)

    assert states_equal(permuted, routed_state)


def test_route_then_optimize(line_5):
    """Route then optimize removes some inserted SWAPs."""
    # Circuit that needs routing
    c = Circuit(3).cx(0, 2)
    routed = route(c, line_5)

    # Count operations before optimization
    before = len(routed.ops)

    # This specific case might not have cancellable SWAPs,
    # but the pipeline should work
    opt = optimize(routed)
    after = len(opt.ops)

    # At minimum, should not increase
    assert after <= before
