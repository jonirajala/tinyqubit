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
from tinyqubit.passes.decompose import decompose

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


# =============================================================================
# Decomposition Tests
# =============================================================================

def test_decompose_swap_to_cx():
    """SWAP decomposes to 3 CX gates."""
    c = Circuit(2).swap(0, 1)
    basis = frozenset({Gate.CX, Gate.RZ, Gate.RX})
    dec = decompose(c, basis)

    assert len(dec.ops) == 3
    assert all(op.gate == Gate.CX for op in dec.ops)


def test_decompose_h_to_rotations():
    """H decomposes to RZ RX RZ."""
    c = Circuit(1).h(0)
    basis = frozenset({Gate.RZ, Gate.RX})
    dec = decompose(c, basis)

    assert len(dec.ops) == 3
    assert dec.ops[0].gate == Gate.RZ
    assert dec.ops[1].gate == Gate.RX
    assert dec.ops[2].gate == Gate.RZ


def test_decompose_s_to_rz():
    """S decomposes to RZ(π/2)."""
    c = Circuit(1).s(0)
    basis = frozenset({Gate.RZ})
    dec = decompose(c, basis)

    assert len(dec.ops) == 1
    assert dec.ops[0].gate == Gate.RZ
    assert abs(dec.ops[0].params[0] - pi/2) < 1e-10


def test_decompose_t_to_rz():
    """T decomposes to RZ(π/4)."""
    c = Circuit(1).t(0)
    basis = frozenset({Gate.RZ})
    dec = decompose(c, basis)

    assert len(dec.ops) == 1
    assert dec.ops[0].gate == Gate.RZ
    assert abs(dec.ops[0].params[0] - pi/4) < 1e-10


def test_decompose_cz_to_h_cx():
    """CZ decomposes to H CX H."""
    c = Circuit(2).cz(0, 1)
    basis = frozenset({Gate.H, Gate.CX})
    dec = decompose(c, basis)

    assert len(dec.ops) == 3
    assert dec.ops[0].gate == Gate.H
    assert dec.ops[1].gate == Gate.CX
    assert dec.ops[2].gate == Gate.H


def test_decompose_keeps_basis_gates():
    """Gates already in basis are not decomposed."""
    c = Circuit(2).cx(0, 1).rz(0, 0.5)
    basis = frozenset({Gate.CX, Gate.RZ})
    dec = decompose(c, basis)

    assert len(dec.ops) == 2
    assert dec.ops[0].gate == Gate.CX
    assert dec.ops[1].gate == Gate.RZ


def test_decompose_chained():
    """CZ decomposes to H CX H, then H decomposes further if not in basis."""
    c = Circuit(2).cz(0, 1)
    basis = frozenset({Gate.CX, Gate.RZ, Gate.RX})  # No H in basis
    dec = decompose(c, basis)

    # CZ → H CX H → (RZ RX RZ) CX (RZ RX RZ)
    assert len(dec.ops) == 7
    assert dec.ops[3].gate == Gate.CX  # CX in middle


def test_decompose_preserves_semantics():
    """Decomposition preserves circuit semantics."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(2).h(0).cx(0, 1)  # Bell state
    basis = frozenset({Gate.CX, Gate.RZ, Gate.RX})
    dec = decompose(c, basis)

    assert states_equal(simulate(c), simulate(dec))


def test_decompose_ry_to_rotations():
    """RY decomposes to RX RZ RX."""
    c = Circuit(1).ry(0, 1.0)
    basis = frozenset({Gate.RZ, Gate.RX})
    dec = decompose(c, basis)

    assert len(dec.ops) == 3
    assert dec.ops[0].gate == Gate.RX
    assert dec.ops[1].gate == Gate.RZ
    assert dec.ops[2].gate == Gate.RX
    assert abs(dec.ops[1].params[0] - 1.0) < 1e-10  # Angle preserved


def test_decompose_ry_preserves_semantics():
    """RY decomposition preserves circuit semantics."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(1).ry(0, pi/3)
    basis = frozenset({Gate.RZ, Gate.RX})
    dec = decompose(c, basis)

    assert states_equal(simulate(c), simulate(dec))


# =============================================================================
# Full Pipeline Tests
# =============================================================================

def test_transpile_full_pipeline():
    """Full transpile pipeline: route → decompose → optimize."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_3"
    )

    # Circuit needing routing (CX on non-adjacent qubits) and decomposition (H)
    c = Circuit(3).h(0).cx(0, 2)
    result = transpile(c, target)

    # All gates should be in basis
    for op in result.ops:
        assert op.gate in target.basis_gates


def test_transpile_preserves_semantics():
    """Full transpile preserves circuit semantics."""
    from tinyqubit.compile import transpile
    from tinyqubit.simulator import simulate, states_equal

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_2"
    )

    c = Circuit(2).h(0).cx(0, 1)  # Bell state
    result = transpile(c, target)

    assert states_equal(simulate(c), simulate(result))


def test_transpile_with_routing_semantics():
    """Transpile with routing preserves semantics (accounting for permutation)."""
    from tinyqubit.compile import transpile
    from tinyqubit.simulator import simulate, states_equal
    import numpy as np

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_3"
    )

    c = Circuit(3).h(0).cx(0, 2)  # Non-adjacent qubits
    result = transpile(c, target)

    # Get permutation from tracker (attached during routing)
    perm = result._tracker.logical_to_physical

    # Simulate and compare with permutation
    original = simulate(c)
    transpiled = simulate(result)

    # Permute original to match physical ordering
    original_reshaped = original.reshape([2] * 3)
    permuted = np.transpose(original_reshaped, perm).reshape(-1)

    assert states_equal(permuted, transpiled)


def test_transpile_various_gates():
    """Transpile handles S, T, Y, Z gates."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_2"
    )

    c = Circuit(2).s(0).t(1).y(0).z(1)
    result = transpile(c, target)

    # All gates should be in basis
    for op in result.ops:
        assert op.gate in target.basis_gates


def test_transpile_swap_becomes_3cx():
    """SWAP gate decomposes to 3 CX gates."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_2"
    )

    c = Circuit(2).swap(0, 1)
    result = transpile(c, target)

    cx_count = sum(1 for op in result.ops if op.gate == Gate.CX)
    assert cx_count == 3


def test_transpile_empty_circuit():
    """Transpile handles empty circuit."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_2"
    )

    c = Circuit(2)
    result = transpile(c, target)

    assert len(result.ops) == 0


def test_transpile_single_gate():
    """Transpile handles single gate."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=1,
        edges=frozenset(),
        basis_gates=frozenset({Gate.RZ, Gate.RX}),
        name="single"
    )

    c = Circuit(1).h(0)
    result = transpile(c, target)

    # H → RZ RX RZ
    assert len(result.ops) == 3
    for op in result.ops:
        assert op.gate in target.basis_gates


def test_transpile_optimization_reduces_gates():
    """Optimizer removes redundant gates after decomposition."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=1,
        edges=frozenset(),
        basis_gates=frozenset({Gate.RZ, Gate.RX}),
        name="single"
    )

    # H H = I, should cancel
    c = Circuit(1).h(0).h(0)
    result = transpile(c, target)

    # After decomposition: (RZ RX RZ)(RZ RX RZ)
    # RZ RZ in middle should merge, potentially more cancellation
    # At minimum, should be less than 6 gates
    assert len(result.ops) < 6


def test_transpile_multiple_2q_gates():
    """Transpile handles multiple 2Q gates needing routing."""
    from tinyqubit.compile import transpile
    from tinyqubit.simulator import simulate, states_equal
    import numpy as np

    target = Target(
        n_qubits=4,
        edges=line_topology(4),  # 0-1-2-3
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_4"
    )

    # Multiple non-adjacent CX gates
    c = Circuit(4).h(0).cx(0, 3).cx(1, 3)
    result = transpile(c, target)

    # All gates in basis
    for op in result.ops:
        assert op.gate in target.basis_gates


def test_transpile_all_to_all_no_routing():
    """All-to-all connectivity skips routing."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=3,
        edges=all_to_all_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="all_to_all_3"
    )

    c = Circuit(3).h(0).cx(0, 2)  # Would need routing on line
    result = transpile(c, target)

    # Should have no extra CX from SWAP decomposition
    # H → 3 gates, CX → 1 gate = 4 total
    assert len(result.ops) == 4


def test_transpile_cz_basis():
    """Transpile to CZ-based basis (Rigetti-style)."""
    from tinyqubit.compile import transpile
    from tinyqubit.simulator import simulate, states_equal

    target = Target(
        n_qubits=2,
        edges=frozenset({(0, 1)}),
        basis_gates=frozenset({Gate.CZ, Gate.RZ, Gate.RX, Gate.H}),
        name="rigetti_style"
    )

    c = Circuit(2).cx(0, 1)
    result = transpile(c, target)

    # CX should become H CZ H or similar
    for op in result.ops:
        assert op.gate in target.basis_gates

    # Verify semantics
    assert states_equal(simulate(c), simulate(result))


def test_transpile_deterministic():
    """Same input always produces same output."""
    from tinyqubit.compile import transpile

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.RZ, Gate.RX}),
        name="line_3"
    )

    c = Circuit(3).h(0).cx(0, 2).rz(1, 0.5)

    result1 = transpile(c, target)
    result2 = transpile(c, target)

    assert len(result1.ops) == len(result2.ops)
    for op1, op2 in zip(result1.ops, result2.ops):
        assert op1 == op2
