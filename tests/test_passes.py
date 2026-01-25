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
# Phase 4.5 Optimization Tests (Clifford, Inverse Pairs, Commutation, Hadamard)
# =============================================================================

def test_optimize_clifford_ss_to_z():
    """S·S merges to Z."""
    c = Circuit(1).s(0).s(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z


def test_optimize_clifford_tt_to_s():
    """T·T merges to S."""
    c = Circuit(1).t(0).t(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.S


def test_optimize_inverse_s_sdg():
    """S·S† cancels to identity."""
    c = Circuit(1).s(0).sdg(0)
    opt = optimize(c)

    assert len(opt.ops) == 0


def test_optimize_inverse_sdg_s():
    """S†·S cancels to identity."""
    c = Circuit(1).sdg(0).s(0)
    opt = optimize(c)

    assert len(opt.ops) == 0


def test_optimize_inverse_t_tdg():
    """T·T† cancels to identity."""
    c = Circuit(1).t(0).tdg(0)
    opt = optimize(c)

    assert len(opt.ops) == 0


def test_optimize_inverse_tdg_t():
    """T†·T cancels to identity."""
    c = Circuit(1).tdg(0).t(0)
    opt = optimize(c)

    assert len(opt.ops) == 0


def test_optimize_hadamard_conjugate_x():
    """H·X·H = Z."""
    c = Circuit(1).h(0).x(0).h(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z


def test_optimize_hadamard_conjugate_z():
    """H·Z·H = X."""
    c = Circuit(1).h(0).z(0).h(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.X


def test_optimize_hadamard_conjugate_different_qubits():
    """H(0)·X(1)·H(0) doesn't simplify (different qubits)."""
    c = Circuit(2).h(0).x(1).h(0)
    opt = optimize(c)

    # H·H on q0 cancels, X on q1 remains
    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.X
    assert opt.ops[0].qubits == (1,)


def test_optimize_commutation_diagonal():
    """Diagonal gates commute and cancel: Z·S·Z cancels to S."""
    c = Circuit(1).z(0).s(0).z(0)
    opt = optimize(c)

    # Z gates cancel through S (diagonal gates commute)
    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.S


def test_optimize_commutation_cancel_z_through_cx_control():
    """Z cancels through CX on control qubit: Z(0)·CX(0,1)·Z(0) = CX(0,1)."""
    c = Circuit(2).z(0).cx(0, 1).z(0)
    opt = optimize(c)

    # Z gates can cancel through CX (Z=RZ(π) commutes on control, and Z is self-inverse)
    z_ops = [op for op in opt.ops if op.gate == Gate.Z]
    assert len(z_ops) == 0
    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.CX


def test_optimize_no_cancel_z_through_cx_target():
    """Z doesn't cancel through CX on target qubit."""
    c = Circuit(2).z(1).cx(0, 1).z(1)
    opt = optimize(c)

    # Z doesn't commute through CX on target, so no cancellation
    z_ops = [op for op in opt.ops if op.gate == Gate.Z]
    assert len(z_ops) == 2


def test_optimize_commutation_cancel_hh():
    """H gates cancel through commuting intermediate: H(0)·X(1)·H(0)."""
    c = Circuit(2).h(0).x(1).h(0)
    opt = optimize(c)

    # H·H on q0 can cancel (X on different qubit commutes)
    h_ops = [op for op in opt.ops if op.gate == Gate.H]
    assert len(h_ops) == 0


def test_optimize_chain_clifford():
    """Chain: T·T·T·T = Z (via T·T=S, S·S=Z)."""
    c = Circuit(1).t(0).t(0).t(0).t(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z


def test_optimize_clifford_tdgtdg_to_sdg():
    """TDG·TDG merges to SDG."""
    c = Circuit(1).tdg(0).tdg(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.SDG


def test_optimize_clifford_sdgsdg_to_z():
    """SDG·SDG merges to Z."""
    c = Circuit(1).sdg(0).sdg(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z


def test_optimize_merge_rz_through_cx_control():
    """RZ merges through CX on control qubit."""
    c = Circuit(2).rz(0, 0.5).cx(0, 1).rz(0, 0.3)
    opt = optimize(c)

    # RZ gates merge through CX (RZ commutes on control)
    rz_ops = [op for op in opt.ops if op.gate == Gate.RZ]
    assert len(rz_ops) == 1
    assert abs(rz_ops[0].params[0] - 0.8) < 1e-10


def test_optimize_no_merge_rz_through_cx_target():
    """RZ doesn't merge through CX on target qubit."""
    c = Circuit(2).rz(1, 0.5).cx(0, 1).rz(1, 0.3)
    opt = optimize(c)

    # RZ doesn't commute through CX on target, no merge
    rz_ops = [op for op in opt.ops if op.gate == Gate.RZ]
    assert len(rz_ops) == 2


def test_optimize_merge_rz_through_diagonal():
    """RZ merges through other diagonal gates."""
    c = Circuit(1).rz(0, 0.5).s(0).rz(0, 0.3)
    opt = optimize(c)

    # RZ gates merge through S (both diagonal)
    rz_ops = [op for op in opt.ops if op.gate == Gate.RZ]
    assert len(rz_ops) == 1
    assert abs(rz_ops[0].params[0] - 0.8) < 1e-10


def test_optimize_merge_rx_through_cx_target():
    """RX merges through CX on target qubit."""
    c = Circuit(2).rx(1, 0.5).cx(0, 1).rx(1, 0.3)
    opt = optimize(c)

    # RX gates merge through CX (RX commutes on target)
    rx_ops = [op for op in opt.ops if op.gate == Gate.RX]
    assert len(rx_ops) == 1
    assert abs(rx_ops[0].params[0] - 0.8) < 1e-10


def test_optimize_merge_to_zero_through_commutation():
    """RZ(a)·CX·RZ(-a) cancels to just CX."""
    c = Circuit(2).rz(0, 1.5).cx(0, 1).rz(0, -1.5)
    opt = optimize(c)

    # RZ gates merge to 0 and are removed
    rz_ops = [op for op in opt.ops if op.gate == Gate.RZ]
    assert len(rz_ops) == 0
    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.CX


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

    state_orig, _ = simulate(c)
    state_routed, _ = simulate(routed)
    assert states_equal(state_orig, state_routed)


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
    original_state, _ = simulate(c)
    routed_state, _ = simulate(routed)

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


def test_decompose_sdg_to_rz():
    """S† decomposes to RZ(-π/2)."""
    c = Circuit(1).sdg(0)
    basis = frozenset({Gate.RZ})
    dec = decompose(c, basis)

    assert len(dec.ops) == 1
    assert dec.ops[0].gate == Gate.RZ
    assert abs(dec.ops[0].params[0] - (-pi/2)) < 1e-10


def test_decompose_tdg_to_rz():
    """T† decomposes to RZ(-π/4)."""
    c = Circuit(1).tdg(0)
    basis = frozenset({Gate.RZ})
    dec = decompose(c, basis)

    assert len(dec.ops) == 1
    assert dec.ops[0].gate == Gate.RZ
    assert abs(dec.ops[0].params[0] - (-pi/4)) < 1e-10


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

    state_orig, _ = simulate(c)
    state_dec, _ = simulate(dec)
    assert states_equal(state_orig, state_dec)


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

    state_orig, _ = simulate(c)
    state_dec, _ = simulate(dec)
    assert states_equal(state_orig, state_dec)


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

    state_orig, _ = simulate(c)
    state_result, _ = simulate(result)
    assert states_equal(state_orig, state_result)


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
    original, _ = simulate(c)
    transpiled, _ = simulate(result)

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
    state_orig, _ = simulate(c)
    state_result, _ = simulate(result)
    assert states_equal(state_orig, state_result)


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


# =============================================================================
# Regression tests for feature interactions
# =============================================================================

def test_decompose_preserves_conditions():
    """Decomposing conditional gates must preserve the condition."""
    from tinyqubit.passes.decompose import decompose

    c = Circuit(2)
    c.x(0).measure(0, 0)
    with c.c_if(0, 1):
        c.h(1)  # H needs decomposition to RZ/RX basis

    basis = frozenset({Gate.RZ, Gate.RX, Gate.CX})
    result = decompose(c, basis)

    # All decomposed H gates should have the condition
    h_decomposed = [op for op in result.ops if op.gate in (Gate.RZ, Gate.RX) and op.qubits == (1,)]
    assert len(h_decomposed) == 3  # H = RZ RX RZ
    for op in h_decomposed:
        assert op.condition == (0, 1), "Condition lost during decomposition"


def test_optimizer_respects_measurement_barrier():
    """Optimizer must not commute gates across measurements."""
    from tinyqubit.passes.optimize import optimize

    # X; MEASURE; X should NOT cancel (measurement is barrier)
    c = Circuit(1)
    c.x(0).measure(0, 0).x(0)
    result = optimize(c)

    # Both X gates should remain
    x_count = sum(1 for op in result.ops if op.gate == Gate.X)
    assert x_count == 2, "Optimizer incorrectly cancelled across measurement"


def test_optimizer_respects_conditional_barrier():
    """Optimizer must not commute gates across conditional ops."""
    from tinyqubit.passes.optimize import optimize

    c = Circuit(1)
    c.x(0).measure(0, 0)
    with c.c_if(0, 1):
        c.z(0)  # Conditional Z
    c.x(0)

    result = optimize(c)

    # X gates should NOT cancel (conditional Z is barrier)
    x_count = sum(1 for op in result.ops if op.gate == Gate.X)
    assert x_count == 2, "Optimizer incorrectly cancelled across conditional"


def test_cz_canonicalization_enables_cancellation():
    """CZ(a,b) and CZ(b,a) should cancel after canonicalization."""
    from tinyqubit.passes.optimize import optimize

    c = Circuit(2).cz(0, 1).cz(1, 0)
    result = optimize(c)

    # Should cancel to empty
    assert len(result.ops) == 0, "CZ(0,1); CZ(1,0) should cancel"


def test_swap_canonicalization_enables_cancellation():
    """SWAP(a,b) and SWAP(b,a) should cancel after canonicalization."""
    from tinyqubit.passes.optimize import optimize

    c = Circuit(2).swap(0, 1).swap(1, 0)
    result = optimize(c)

    # Should cancel to empty
    assert len(result.ops) == 0, "SWAP(0,1); SWAP(1,0) should cancel"


# =============================================================================
# Target validation and utility tests
# =============================================================================

def test_target_validates_edge_endpoints():
    """Target rejects edges with invalid qubit indices."""
    with pytest.raises(ValueError, match="invalid qubit index"):
        Target(n_qubits=3, edges=frozenset({(0, 5)}), basis_gates=frozenset())


def test_target_rejects_self_loops():
    """Target rejects self-loop edges."""
    with pytest.raises(ValueError, match="Self-loop"):
        Target(n_qubits=3, edges=frozenset({(1, 1)}), basis_gates=frozenset())


def test_target_is_all_to_all_with_duplicate_edges():
    """is_all_to_all handles both (a,b) and (b,a) correctly."""
    # 3 qubits needs 3 pairs: (0,1), (0,2), (1,2)
    # Include duplicates - should still detect missing (1,2)
    edges = frozenset({(0, 1), (1, 0), (0, 2), (2, 0)})  # Missing (1,2)
    target = Target(n_qubits=3, edges=edges, basis_gates=frozenset())
    assert not target.is_all_to_all(), "Should detect missing (1,2) pair"

    # Now with all pairs (including duplicates)
    edges_full = frozenset({(0, 1), (1, 0), (0, 2), (1, 2)})
    target_full = Target(n_qubits=3, edges=edges_full, basis_gates=frozenset())
    assert target_full.is_all_to_all()


def test_target_distance_cached():
    """distance() method returns correct values and caches results."""
    target = Target(
        n_qubits=4,
        edges=frozenset({(0, 1), (1, 2), (2, 3)}),  # Line: 0-1-2-3
        basis_gates=frozenset()
    )
    assert target.distance(0, 0) == 0  # Same qubit
    assert target.distance(0, 1) == 1  # Adjacent
    assert target.distance(0, 2) == 2  # Two hops
    assert target.distance(0, 3) == 3  # Three hops
    assert target.distance(3, 0) == 3  # Symmetric


def test_target_distance_disconnected():
    """distance() returns -1 for disconnected qubits."""
    target = Target(
        n_qubits=4,
        edges=frozenset({(0, 1), (2, 3)}),  # Two disconnected pairs
        basis_gates=frozenset()
    )
    assert target.distance(0, 1) == 1
    assert target.distance(0, 2) == -1  # Unreachable
    assert target.distance(1, 3) == -1  # Unreachable


# =============================================================================
# QubitTracker validation tests
# =============================================================================

def test_tracker_validates_swap_indices():
    """Tracker rejects swaps with invalid qubit indices."""
    tracker = QubitTracker(3)
    with pytest.raises(ValueError, match="Invalid physical qubit"):
        tracker.record_swap(0, 5)


def test_tracker_ignores_self_swap():
    """SWAP(a, a) is a no-op."""
    tracker = QubitTracker(3)
    tracker.record_swap(1, 1)
    assert tracker.pending == []  # No swap recorded
    assert tracker.logical_to_phys(1) == 1  # Mapping unchanged


def test_tracker_add_logical_gate():
    """add_logical_gate auto-translates qubit indices."""
    tracker = QubitTracker(3)
    tracker.record_swap(0, 1, triggered_by=-1)  # Now logical 0 is at physical 1
    tracker.add_logical_gate(Gate.X, (0,))

    # Should have pending SWAP and X on physical qubit 1
    materialized = tracker.materialize()
    assert len(materialized) == 2
    assert materialized[1].phys_qubits == (1,)


def test_tracker_flush_clears_pending():
    """flush() returns and clears pending ops."""
    tracker = QubitTracker(3)
    tracker.record_swap(0, 1, triggered_by=0)
    tracker.add_gate(Gate.X, (0,))

    ops1 = tracker.flush()
    assert len(ops1) == 2

    ops2 = tracker.flush()
    assert len(ops2) == 0  # Cleared


def test_route_flushes_before_measure():
    """Routing flushes pending SWAPs before MEASURE barrier."""
    # Circuit: CX(0,2) on line 0-1-2, then MEASURE(0)
    # SWAPs needed for CX must appear BEFORE the measure
    c = Circuit(3).cx(0, 2).measure(0)

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.SWAP})
    )

    routed = route(c, target)

    # Find positions of SWAP and MEASURE
    swap_indices = [i for i, op in enumerate(routed.ops) if op.gate == Gate.SWAP]
    measure_idx = next(i for i, op in enumerate(routed.ops) if op.gate == Gate.MEASURE)

    # All SWAPs must come before MEASURE
    assert all(si < measure_idx for si in swap_indices), "SWAPs must be flushed before MEASURE"


def test_route_flushes_before_conditional():
    """Routing flushes pending SWAPs before conditional ops."""
    c = Circuit(3)
    c.cx(0, 2)  # Needs routing
    c.measure(1, 0)
    with c.c_if(0, 1):
        c.x(0)  # Conditional - barrier

    target = Target(
        n_qubits=3,
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX, Gate.SWAP, Gate.X})
    )

    routed = route(c, target)

    # Find conditional X
    cond_idx = next(i for i, op in enumerate(routed.ops) if op.condition is not None)

    # All SWAPs before conditional
    swap_indices = [i for i, op in enumerate(routed.ops) if op.gate == Gate.SWAP]
    assert all(si < cond_idx for si in swap_indices)


def test_route_validates_target_has_enough_qubits():
    """Routing fails if target has fewer qubits than circuit."""
    c = Circuit(5).cx(0, 4)

    target = Target(
        n_qubits=3,  # Not enough!
        edges=line_topology(3),
        basis_gates=frozenset({Gate.CX})
    )

    with pytest.raises(ValueError, match="Target has 3 qubits but circuit needs 5"):
        route(c, target)


def test_route_disconnected_topology_raises():
    """Routing fails for disconnected topology when path needed."""
    c = Circuit(4).cx(0, 3)  # Needs path between 0 and 3

    # Disconnected: 0-1 and 2-3 are separate components
    target = Target(
        n_qubits=4,
        edges=frozenset({(0, 1), (2, 3)}),
        basis_gates=frozenset({Gate.CX, Gate.SWAP})
    )

    with pytest.raises(ValueError, match="No path between"):
        route(c, target)


def test_route_allows_larger_target():
    """Routing works when target has more qubits than circuit (ancillas)."""
    c = Circuit(2).cx(0, 1)

    target = Target(
        n_qubits=5,  # More than needed
        edges=line_topology(5),
        basis_gates=frozenset({Gate.CX})
    )

    # Should work - uses first 2 qubits of larger target
    routed = route(c, target)
    assert routed.n_qubits == 5


# =============================================================================
# Push Diagonals Tests
# =============================================================================

from tinyqubit.passes.push_diagonals import push_diagonals


def test_push_diagonals_rz_through_cx_control():
    """RZ on control pushes backward through CX."""
    c = Circuit(2).cx(0, 1).rz(0, 0.5)
    pushed = push_diagonals(c)

    # RZ should now be before CX
    assert pushed.ops[0].gate == Gate.RZ
    assert pushed.ops[0].qubits == (0,)
    assert pushed.ops[1].gate == Gate.CX


def test_push_diagonals_rz_not_through_cx_target():
    """RZ on target does NOT push through CX."""
    c = Circuit(2).cx(0, 1).rz(1, 0.5)
    pushed = push_diagonals(c)

    # RZ stays after CX (doesn't commute on target)
    assert pushed.ops[0].gate == Gate.CX
    assert pushed.ops[1].gate == Gate.RZ
    assert pushed.ops[1].qubits == (1,)


def test_push_diagonals_through_cz():
    """RZ pushes through CZ on either qubit."""
    c = Circuit(2).cz(0, 1).rz(0, 0.5).rz(1, 0.3)
    pushed = push_diagonals(c)

    # Both RZ should be before CZ
    assert pushed.ops[0].gate == Gate.RZ
    assert pushed.ops[1].gate == Gate.RZ
    assert pushed.ops[2].gate == Gate.CZ


def test_push_diagonals_through_disjoint():
    """Diagonal pushes through gates on disjoint qubits."""
    c = Circuit(3).h(0).cx(0, 1).rz(2, 0.5)
    pushed = push_diagonals(c)

    # RZ(2) should push all the way to the front
    assert pushed.ops[0].gate == Gate.RZ
    assert pushed.ops[0].qubits == (2,)


def test_push_diagonals_stops_at_h():
    """Diagonal stops at H gate (doesn't commute)."""
    c = Circuit(1).h(0).rz(0, 0.5)
    pushed = push_diagonals(c)

    # RZ stays after H
    assert pushed.ops[0].gate == Gate.H
    assert pushed.ops[1].gate == Gate.RZ


def test_push_diagonals_stops_at_measure():
    """Diagonal stops at MEASURE barrier."""
    c = Circuit(1).rz(0, 0.3)
    c.measure(0, 0)
    c.rz(0, 0.5)
    pushed = push_diagonals(c)

    # RZ after measure should not push past measure
    assert pushed.ops[1].gate == Gate.MEASURE
    assert pushed.ops[2].gate == Gate.RZ


def test_push_diagonals_cnot_conj_pattern():
    """CNOT conjugation pattern: Z gates on control should gather."""
    c = Circuit(2).z(0).cx(0, 1).z(0).z(1).cx(0, 1).z(1)
    pushed = push_diagonals(c)
    opt = optimize(pushed)

    # Full optimization: push_diagonals + CX conjugation should reduce to Z(0)
    # Original: Z(0) CX Z(0) Z(1) CX Z(1) → push → Z(0) Z(0) CX Z(1) CX Z(1)
    # → cancel Z(0)Z(0) → CX Z(1) CX Z(1) → CX conjugation → Z(0) Z(1) Z(1) → Z(0)
    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z
    assert opt.ops[0].qubits == (0,)


# === CX Conjugation Tests ===

def test_cx_conjugation_z_on_target():
    """CX · Z(target) · CX → Z(control) Z(target)."""
    c = Circuit(2).cx(0, 1).z(1).cx(0, 1)
    opt = optimize(c)

    assert len(opt.ops) == 2
    gates = sorted([(op.gate, op.qubits) for op in opt.ops])
    assert gates == [(Gate.Z, (0,)), (Gate.Z, (1,))]


def test_cx_conjugation_x_on_control():
    """CX · X(control) · CX → X(control) X(target)."""
    c = Circuit(2).cx(0, 1).x(0).cx(0, 1)
    opt = optimize(c)

    assert len(opt.ops) == 2
    gates = sorted([(op.gate, op.qubits) for op in opt.ops])
    assert gates == [(Gate.X, (0,)), (Gate.X, (1,))]


def test_cx_conjugation_z_on_control():
    """CX · Z(control) · CX → Z(control)."""
    c = Circuit(2).cx(0, 1).z(0).cx(0, 1)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z
    assert opt.ops[0].qubits == (0,)


def test_cx_conjugation_x_on_target():
    """CX · X(target) · CX → X(target)."""
    c = Circuit(2).cx(0, 1).x(1).cx(0, 1)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.X
    assert opt.ops[0].qubits == (1,)


def test_cx_conjugation_with_commuting_intermediate():
    """CX conjugation works with commuting gates in between."""
    # CX(0,1) Z(2) Z(1) CX(0,1) - Z(2) commutes with CX, should still apply conjugation
    c = Circuit(3).cx(0, 1).z(2).z(1).cx(0, 1)
    opt = optimize(c)

    # Z(1) conjugation: CX Z(1) CX → Z(0) Z(1), Z(2) preserved
    assert len(opt.ops) == 3
    gates = sorted([(op.gate, op.qubits) for op in opt.ops])
    assert gates == [(Gate.Z, (0,)), (Gate.Z, (1,)), (Gate.Z, (2,))]


def test_cx_conjugation_chain():
    """Multiple CX conjugations combine with other optimizations."""
    # CX Z(1) CX Z(1) → Z(0) Z(1) Z(1) → Z(0)
    c = Circuit(2).cx(0, 1).z(1).cx(0, 1).z(1)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.Z
    assert opt.ops[0].qubits == (0,)


def test_cx_conjugation_preserves_semantics():
    """CX conjugation preserves circuit semantics."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(2).cx(0, 1).z(1).cx(0, 1)
    opt = optimize(c)

    state_orig, _ = simulate(c)
    state_opt, _ = simulate(opt)
    assert states_equal(state_orig, state_opt)


# === 2-Qubit Template Tests ===

def test_hadamard_cx_to_cz():
    """H(t)·CX(c,t)·H(t) → CZ(c,t)."""
    c = Circuit(2).h(1).cx(0, 1).h(1)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.CZ
    assert opt.ops[0].qubits == (0, 1)


def test_hadamard_cx_to_cz_reversed_control():
    """H(t)·CX(c,t)·H(t) → CZ(c,t) with different control."""
    c = Circuit(2).h(0).cx(1, 0).h(0)
    opt = optimize(c)

    assert len(opt.ops) == 1
    assert opt.ops[0].gate == Gate.CZ
    assert opt.ops[0].qubits == (1, 0)  # Preserves original CX qubit order (CZ is symmetric)


def test_hadamard_cx_to_cz_preserves_semantics():
    """H·CX·H → CZ preserves circuit semantics."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(2).h(1).cx(0, 1).h(1)
    opt = optimize(c)

    state_orig, _ = simulate(c)
    state_opt, _ = simulate(opt)
    assert states_equal(state_orig, state_opt)


def test_push_diagonals_rz_merge_after_push():
    """RZ gates merge after being pushed together."""
    c = Circuit(2).rz(0, 0.5).cx(0, 1).rz(0, 0.3)
    pushed = push_diagonals(c)
    opt = optimize(pushed)

    # Should have only 1 RZ on qubit 0 (merged)
    rz_q0 = [op for op in opt.ops if op.gate == Gate.RZ and op.qubits == (0,)]
    assert len(rz_q0) == 1
    assert abs(rz_q0[0].params[0] - 0.8) < 1e-9


def test_push_diagonals_preserves_semantics():
    """Pushing preserves circuit semantics."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(2).h(0).cx(0, 1).rz(0, 0.5).rz(1, 0.3)
    pushed = push_diagonals(c)

    state_orig, _ = simulate(c)
    state_pushed, _ = simulate(pushed)
    assert states_equal(state_orig, state_pushed)


def test_push_diagonals_preserves_semantics_complex():
    """Pushing preserves semantics on complex circuit."""
    from tinyqubit.simulator import simulate, states_equal

    c = Circuit(3)
    c.h(0).cx(0, 1).rz(0, 0.5).cz(1, 2).rz(1, 0.3).rz(2, 0.7)
    pushed = push_diagonals(c)

    state_orig, _ = simulate(c)
    state_pushed, _ = simulate(pushed)
    assert states_equal(state_orig, state_pushed)


def test_push_diagonals_empty_circuit():
    """Push handles empty circuit."""
    c = Circuit(2)
    pushed = push_diagonals(c)
    assert len(pushed.ops) == 0


def test_push_diagonals_no_diagonals():
    """Push handles circuit with no diagonal gates."""
    c = Circuit(2).h(0).cx(0, 1).x(1)
    pushed = push_diagonals(c)
    assert len(pushed.ops) == 3
    assert pushed.ops[0].gate == Gate.H
    assert pushed.ops[1].gate == Gate.CX
    assert pushed.ops[2].gate == Gate.X


def test_push_diagonals_all_clifford_diagonals():
    """Push handles S, T, SDG, TDG gates."""
    c = Circuit(2).cx(0, 1).s(0).t(0).sdg(0).tdg(0)
    pushed = push_diagonals(c)

    # All 4 diagonal gates should push before CX
    assert pushed.ops[4].gate == Gate.CX
    for i in range(4):
        assert pushed.ops[i].gate in {Gate.S, Gate.T, Gate.SDG, Gate.TDG}


# =============================================================================
# Target all_pairs_distances Tests
# =============================================================================

def test_target_all_pairs_distances_correctness():
    """all_pairs_distances returns correct distance matrix."""
    target = Target(
        n_qubits=4,
        edges=frozenset({(0, 1), (1, 2), (2, 3)}),  # Line: 0-1-2-3
        basis_gates=frozenset()
    )

    dist = target.all_pairs_distances()

    # Check specific distances
    assert dist[0][0] == 0
    assert dist[0][1] == 1
    assert dist[0][2] == 2
    assert dist[0][3] == 3
    assert dist[1][2] == 1
    assert dist[1][3] == 2
    assert dist[2][3] == 1

    # Symmetry
    for i in range(4):
        for j in range(4):
            assert dist[i][j] == dist[j][i]


def test_target_all_pairs_distances_cached():
    """all_pairs_distances caches and returns same object."""
    target = Target(
        n_qubits=3,
        edges=frozenset({(0, 1), (1, 2)}),
        basis_gates=frozenset()
    )

    dist1 = target.all_pairs_distances()
    dist2 = target.all_pairs_distances()

    assert dist1 is dist2, "all_pairs_distances should return cached result"


def test_target_all_pairs_distances_disconnected():
    """all_pairs_distances returns -1 for disconnected qubits."""
    target = Target(
        n_qubits=4,
        edges=frozenset({(0, 1), (2, 3)}),  # Two disconnected pairs
        basis_gates=frozenset()
    )

    dist = target.all_pairs_distances()

    assert dist[0][1] == 1
    assert dist[2][3] == 1
    assert dist[0][2] == -1
    assert dist[0][3] == -1
    assert dist[1][2] == -1
    assert dist[1][3] == -1
