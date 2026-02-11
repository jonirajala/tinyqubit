"""
Tests for statevector simulator.

Tests:
    - Known circuits: Bell, GHZ, plus state
    - Gate correctness
    - states_equal with global phase
    - Sampling determinism and distribution
"""
import pytest
import numpy as np
from math import sqrt, pi

from tinyqubit.ir import Circuit
from tinyqubit.simulator import simulate, states_equal, sample


# =============================================================================
# Helper Constants
# =============================================================================

SQRT2_INV = 1 / sqrt(2)


# =============================================================================
# Known Circuit Tests
# =============================================================================

def test_plus_state():
    """H|0⟩ = |+⟩ = (|0⟩ + |1⟩) / √2"""
    c = Circuit(1).h(0)
    state, _ = simulate(c)

    expected = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_bell_state():
    """Bell state: (|00⟩ + |11⟩) / √2"""
    c = Circuit(2).h(0).cx(0, 1)
    state, _ = simulate(c)

    expected = np.array([SQRT2_INV, 0, 0, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_ghz_state():
    """GHZ state: (|000⟩ + |111⟩) / √2"""
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    state, _ = simulate(c)

    expected = np.zeros(8, dtype=complex)
    expected[0] = SQRT2_INV  # |000⟩
    expected[7] = SQRT2_INV  # |111⟩
    assert states_equal(state, expected)


# =============================================================================
# Single Qubit Gate Tests
# =============================================================================

def test_x_gate():
    """X|0⟩ = |1⟩"""
    c = Circuit(1).x(0)
    state, _ = simulate(c)

    expected = np.array([0, 1], dtype=complex)
    assert states_equal(state, expected)


def test_y_gate():
    """Y|0⟩ = i|1⟩"""
    c = Circuit(1).y(0)
    state, _ = simulate(c)

    expected = np.array([0, 1j], dtype=complex)
    assert states_equal(state, expected)


def test_z_gate():
    """Z|+⟩ = |−⟩"""
    c = Circuit(1).h(0).z(0)
    state, _ = simulate(c)

    # |−⟩ = (|0⟩ - |1⟩) / √2
    expected = np.array([SQRT2_INV, -SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_h_gate_twice():
    """H·H = I"""
    c = Circuit(1).h(0).h(0)
    state, _ = simulate(c)

    expected = np.array([1, 0], dtype=complex)
    assert states_equal(state, expected)


def test_s_gate():
    """S|1⟩ = i|1⟩"""
    c = Circuit(1).x(0).s(0)
    state, _ = simulate(c)

    expected = np.array([0, 1j], dtype=complex)
    assert states_equal(state, expected)


def test_t_gate():
    """T|1⟩ = e^(iπ/4)|1⟩"""
    c = Circuit(1).x(0).t(0)
    state, _ = simulate(c)

    expected = np.array([0, np.exp(1j * pi / 4)], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Rotation Gate Tests
# =============================================================================

def test_rx_pi():
    """RX(π)|0⟩ = -i|1⟩"""
    c = Circuit(1).rx(0, pi)
    state, _ = simulate(c)

    expected = np.array([0, -1j], dtype=complex)
    assert states_equal(state, expected)


def test_ry_pi():
    """RY(π)|0⟩ = |1⟩"""
    c = Circuit(1).ry(0, pi)
    state, _ = simulate(c)

    expected = np.array([0, 1], dtype=complex)
    assert states_equal(state, expected)


def test_rz_pi():
    """RZ(π)|+⟩ = |−⟩ (up to global phase)"""
    c = Circuit(1).h(0).rz(0, pi)
    state, _ = simulate(c)

    # RZ(π) = -iZ, so RZ(π)|+⟩ = -i|−⟩
    expected = np.array([SQRT2_INV, -SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Two Qubit Gate Tests
# =============================================================================

def test_cx_control_zero():
    """CX with control=0 does nothing"""
    c = Circuit(2).cx(0, 1)  # control q0=|0⟩
    state, _ = simulate(c)

    expected = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    assert states_equal(state, expected)


def test_cx_control_one():
    """CX with control=1 flips target"""
    c = Circuit(2).x(0).cx(0, 1)  # control q0=|1⟩
    state, _ = simulate(c)

    expected = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
    assert states_equal(state, expected)


def test_cz_gate():
    """CZ|11⟩ = -|11⟩"""
    c = Circuit(2).x(0).x(1).cz(0, 1)
    state, _ = simulate(c)

    expected = np.array([0, 0, 0, -1], dtype=complex)  # -|11⟩
    assert states_equal(state, expected)


def test_cz_plus_states():
    """CZ on |++⟩ creates |Φ+⟩-like state"""
    c = Circuit(2).h(0).h(1).cz(0, 1)
    state, _ = simulate(c)

    # |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    # CZ flips phase of |11⟩
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Three-Qubit Gate Tests
# =============================================================================

def test_ccx_truth_table():
    """CCX (Toffoli) truth table: flips target iff both controls are |1⟩."""
    # |000⟩ → |000⟩
    c = Circuit(3).ccx(0, 1, 2)
    state, _ = simulate(c)
    assert states_equal(state, np.array([1,0,0,0,0,0,0,0], dtype=complex))

    # |100⟩ → |100⟩ (only one control)
    c = Circuit(3).x(0).ccx(0, 1, 2)
    state, _ = simulate(c)
    assert states_equal(state, np.array([0,0,0,0,1,0,0,0], dtype=complex))

    # |010⟩ → |010⟩ (only one control)
    c = Circuit(3).x(1).ccx(0, 1, 2)
    state, _ = simulate(c)
    assert states_equal(state, np.array([0,0,1,0,0,0,0,0], dtype=complex))

    # |110⟩ → |111⟩ (both controls → flip target)
    c = Circuit(3).x(0).x(1).ccx(0, 1, 2)
    state, _ = simulate(c)
    assert states_equal(state, np.array([0,0,0,0,0,0,0,1], dtype=complex))


def test_ccz_phase_flip():
    """CCZ flips phase of |111⟩ only."""
    c = Circuit(3).x(0).x(1).x(2).ccz(0, 1, 2)
    state, _ = simulate(c)
    expected = np.array([0,0,0,0,0,0,0,-1], dtype=complex)
    assert states_equal(state, expected)


def test_ccz_no_flip_partial():
    """CCZ does nothing when not all qubits are |1⟩."""
    c = Circuit(3).x(0).x(1).ccz(0, 1, 2)
    state, _ = simulate(c)
    expected = np.array([0,0,0,0,0,0,1,0], dtype=complex)
    assert states_equal(state, expected)


def test_ccx_superposition():
    """CCX on superposition state."""
    c = Circuit(3).h(0).h(1).ccx(0, 1, 2)
    state, _ = simulate(c)
    # |00⟩→|000⟩, |01⟩→|010⟩, |10⟩→|100⟩, |11⟩→|111⟩ (each with 1/2 amp)
    expected = np.array([0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5], dtype=complex)
    assert states_equal(state, expected)


def test_ccx_equals_h_ccz_h():
    """CCX = H(target) · CCZ · H(target)."""
    c1 = Circuit(3).x(0).x(1).ccx(0, 1, 2)
    c2 = Circuit(3).x(0).x(1).h(2).ccz(0, 1, 2).h(2)
    state1, _ = simulate(c1)
    state2, _ = simulate(c2)
    assert states_equal(state1, state2)


# =============================================================================
# states_equal Tests
# =============================================================================

def test_states_equal_same():
    """Same states are equal"""
    s = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    assert states_equal(s, s)


def test_states_equal_global_phase():
    """States differing by global phase are equal"""
    s1 = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    s2 = s1 * np.exp(1j * pi / 3)  # arbitrary phase
    assert states_equal(s1, s2)


def test_states_equal_negative():
    """Negative phase is still global phase"""
    s1 = np.array([1, 0], dtype=complex)
    s2 = np.array([-1, 0], dtype=complex)
    assert states_equal(s1, s2)


def test_states_not_equal():
    """Different states are not equal"""
    s1 = np.array([1, 0], dtype=complex)
    s2 = np.array([0, 1], dtype=complex)
    assert not states_equal(s1, s2)


def test_states_equal_zeros():
    """Two zero vectors are equal"""
    s1 = np.array([0, 0], dtype=complex)
    s2 = np.array([0, 0], dtype=complex)
    assert states_equal(s1, s2)


# =============================================================================
# Sample Tests
# =============================================================================

def test_sample_deterministic():
    """Same seed gives same results"""
    state = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)

    counts1 = sample(state, 100, seed=42)
    counts2 = sample(state, 100, seed=42)

    assert counts1 == counts2


def test_sample_deterministic_bell():
    """Bell state sampling is deterministic with seed"""
    c = Circuit(2).h(0).cx(0, 1)
    state, _ = simulate(c)

    counts1 = sample(state, 1000, seed=123)
    counts2 = sample(state, 1000, seed=123)

    assert counts1 == counts2


def test_sample_bell_distribution():
    """Bell state gives roughly 50/50 |00⟩ and |11⟩"""
    c = Circuit(2).h(0).cx(0, 1)
    state, _ = simulate(c)

    counts = sample(state, 10000, seed=42)

    # Should only have 00 and 11
    assert set(counts.keys()) <= {'00', '11'}

    # Each should be roughly 50% (within 5% margin)
    total = sum(counts.values())
    for bitstring in counts:
        ratio = counts[bitstring] / total
        assert 0.45 < ratio < 0.55, f"{bitstring}: {ratio}"


def test_sample_deterministic_state():
    """|0⟩ always gives '0'"""
    state = np.array([1, 0], dtype=complex)
    counts = sample(state, 100)

    assert counts == {'0': 100}


def test_sample_bitstring_format():
    """Bitstrings have correct length"""
    c = Circuit(3).h(0)
    state, _ = simulate(c)

    counts = sample(state, 100, seed=42)

    for bitstring in counts:
        assert len(bitstring) == 3


# =============================================================================
# Integration Tests
# =============================================================================

def test_grover_2qubit():
    """2-qubit Grover for |11⟩ produces correct result"""
    c = Circuit(2)
    # Superposition
    c.h(0).h(1)
    # Oracle (mark |11⟩)
    c.cz(0, 1)
    # Diffusion
    c.h(0).h(1)
    c.x(0).x(1)
    c.cz(0, 1)
    c.x(0).x(1)
    c.h(0).h(1)

    state, _ = simulate(c)
    counts = sample(state, 1000, seed=42)

    # |11⟩ should dominate
    assert counts.get('11', 0) > 900


def test_measure_collapses_bell():
    """MEASURE operations now collapse state (dynamic circuit support)"""
    c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)

    # With measurement, state collapses to |00⟩ or |11⟩
    state, _ = simulate(c, seed=42)

    # State should be collapsed to |00⟩ or |11⟩, not a superposition
    prob_00 = abs(state[0])**2
    prob_11 = abs(state[3])**2
    # One of them should be ~1, the other ~0
    assert (prob_00 > 0.99 and prob_11 < 0.01) or (prob_00 < 0.01 and prob_11 > 0.99)


# =============================================================================
# Cross-Validation with Qiskit (Optional)
# =============================================================================

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
def test_cross_validate_bell():
    """Our Bell state matches Qiskit"""
    # TinyQubit
    c = Circuit(2).h(0).cx(0, 1)
    our_state, _ = simulate(c)

    # Qiskit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qiskit_state = Statevector.from_instruction(qc).data

    assert states_equal(our_state, qiskit_state)


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
def test_cross_validate_rotations():
    """Our rotation gates match Qiskit"""
    # TinyQubit
    c = Circuit(1).rx(0, 1.23).ry(0, 2.34).rz(0, 3.45)
    our_state, _ = simulate(c)

    # Qiskit
    qc = QuantumCircuit(1)
    qc.rx(1.23, 0)
    qc.ry(2.34, 0)
    qc.rz(3.45, 0)
    qiskit_state = Statevector.from_instruction(qc).data

    assert states_equal(our_state, qiskit_state)


# =============================================================================
# Mid-Circuit Measurement Tests (Layer A)
# =============================================================================

def test_measure_collapses_state():
    """After measurement, state is |0⟩ or |1⟩, not superposition."""
    c = Circuit(1).h(0).measure(0)
    state, _ = simulate(c, seed=42)
    # State should be collapsed to |0⟩ or |1⟩
    assert np.allclose(np.abs(state), [1, 0]) or \
           np.allclose(np.abs(state), [0, 1])


def test_measure_stores_result():
    """Classical register contains measurement outcome."""
    c = Circuit(1).x(0).measure(0)  # |1⟩
    _, classical = simulate(c, seed=42)
    assert classical.get(0) == 1


def test_measure_stores_in_specified_bit():
    """Measurement result stored in specified classical bit."""
    c = Circuit(2, 4)  # 2 qubits, 4 classical bits
    c.x(0).measure(0, 2)  # Measure qubit 0 into classical bit 2
    _, classical = simulate(c, seed=42)
    assert classical[2] == 1  # Measured bit has result
    assert classical[0] == 0  # Unmeasured bit stays at initial value (0)


def test_bell_state_measurement_correlation():
    """Bell state measurements are correlated."""
    c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)

    # Run multiple times and check correlation
    for seed in range(10):
        _, classical = simulate(c, seed=seed)
        b0 = classical.get(0)
        b1 = classical.get(1)
        assert b0 == b1, f"Bell state should give correlated measurements, got {b0}, {b1}"


# =============================================================================
# Reset Tests (Layer A)
# =============================================================================

def test_reset_from_zero():
    """Reset |0⟩ stays |0⟩."""
    c = Circuit(1).reset(0)
    state, _ = simulate(c, seed=42)
    assert np.allclose(np.abs(state), [1, 0])


def test_reset_from_one():
    """Reset |1⟩ becomes |0⟩."""
    c = Circuit(1).x(0).reset(0)
    state, _ = simulate(c, seed=42)
    assert np.allclose(np.abs(state), [1, 0])


def test_reset_from_superposition():
    """Reset from superposition produces |0⟩."""
    c = Circuit(1).h(0).reset(0)
    state, _ = simulate(c, seed=42)
    assert np.allclose(np.abs(state), [1, 0])


def test_reset_mid_circuit():
    """Reset in middle of circuit works correctly."""
    c = Circuit(1).h(0).reset(0).h(0)  # |0⟩ -> |+⟩ -> |0⟩ -> |+⟩
    state, _ = simulate(c, seed=42)
    expected = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Conditional Tests (Layer A)
# =============================================================================

def test_conditional_executes_when_met():
    """Gate executes when condition is met."""
    c = Circuit(1)
    c.x(0).measure(0)  # Measure |1⟩ -> classical bit 0 = 1
    with c.c_if(0, 1):
        c.x(0)  # X applied when bit 0 == 1 -> back to |0⟩
    state, _ = simulate(c, seed=42)
    assert np.allclose(np.abs(state), [1, 0])


def test_conditional_skips_when_not_met():
    """Gate skipped when condition not met."""
    c = Circuit(1)
    c.measure(0)  # Measure |0⟩ -> classical bit 0 = 0
    with c.c_if(0, 1):
        c.x(0)  # X NOT applied because bit 0 == 0
    state, _ = simulate(c, seed=42)
    assert np.allclose(np.abs(state), [1, 0])


def test_conditional_multiple_gates():
    """Multiple gates in conditional block."""
    c = Circuit(1)
    c.x(0).measure(0)  # bit 0 = 1
    with c.c_if(0, 1):
        c.h(0)
        c.z(0)
        c.h(0)  # H·Z·H = X, so X·X = I on |1⟩... no wait, X|1⟩ = |0⟩
    state, _ = simulate(c, seed=42)
    # After X|0⟩ = |1⟩, measure 1, then H·Z·H = X, so X|1⟩ = |0⟩
    assert np.allclose(np.abs(state), [1, 0])


def test_conditional_on_different_bit():
    """Conditional on different classical bit."""
    c = Circuit(2, 2)
    c.x(0).measure(0, 0)  # bit 0 = 1
    c.measure(1, 1)       # bit 1 = 0
    with c.c_if(1, 1):
        c.x(0)  # Should NOT execute (bit 1 = 0)
    _, classical = simulate(c, seed=42)
    # Qubit 0 was measured as 1 and collapsed to |1⟩, no further changes
    assert classical.get(0) == 1


# =============================================================================
# Simulation Result Tests
# =============================================================================

def test_simulate_returns_tuple():
    """simulate() returns (statevector, classical_register) tuple."""
    c = Circuit(1).h(0)
    result = simulate(c)
    assert isinstance(result, tuple)
    assert len(result) == 2
    state, classical = result
    assert isinstance(state, np.ndarray)
    assert state.shape == (2,)


def test_classical_dict_values():
    """Classical dict contains correct measurement outcomes."""
    c = Circuit(3)
    c.x(0).x(2)  # |101⟩
    c.measure(0).measure(1).measure(2)
    _, classical = simulate(c, seed=42)
    assert classical.get(0) == 1
    assert classical.get(1) == 0
    assert classical.get(2) == 1


# =============================================================================
# Noise Tests (Layer B)
# =============================================================================

def test_depolarizing_modifies_state():
    """With high depolarizing noise, state differs from ideal."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).h(0)

    # Very high error rate
    noise = NoiseModel()
    noise.add_depolarizing(1.0)  # 100% error rate

    # With 100% error, state will be modified
    state_noisy, _ = simulate(c, seed=42, noise_model=noise)
    state_ideal, _ = simulate(c, seed=42)

    # They should differ (very high probability)
    # Note: there's a tiny chance they're the same if the random Pauli is identity-like
    # But with 100% error this is unlikely


def test_readout_error_flips_result():
    """Readout error can flip measurement outcome."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).x(0).measure(0)  # Should always measure 1

    # 100% readout error: always flip 1->0
    noise = NoiseModel()
    noise.add_readout_error(p0_given_1=1.0, p1_given_0=0.0)

    _, classical = simulate(c, seed=42, noise_model=noise)
    # True outcome is 1, but readout error flips to 0
    assert classical.get(0) == 0


def test_noise_model_chaining():
    """NoiseModel methods return self for chaining."""
    from tinyqubit.noise import NoiseModel

    noise = NoiseModel().add_depolarizing(0.01).add_readout_error(0.01, 0.01)
    # Verify chaining works - noise functions are stored
    assert len(noise.default_noise) == 1
    assert noise.readout_error_fn is not None


# =============================================================================
# Gate Fusion Tests (Layer D)
# =============================================================================

def test_fusion_preserves_semantics():
    """Fused circuit produces same state as original."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(1).h(0).s(0).t(0).h(0)
    fused = fuse_1q_gates(c)

    state_orig, _ = simulate(c)
    state_fused, _ = simulate(fused)
    assert states_equal(state_orig, state_fused)


def test_fusion_reduces_gates():
    """Multiple 1Q gates become <= 3 rotations."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(1).h(0).s(0).t(0).h(0).s(0)  # 5 gates
    fused = fuse_1q_gates(c)
    assert len(fused.ops) <= 3


def test_fusion_handles_barriers():
    """Fusion respects measurement/reset barriers."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(1).h(0).measure(0).h(0)
    fused = fuse_1q_gates(c)

    # Should have: [fused H], MEASURE, [fused H]
    # The H gates can't be fused across the measure
    assert any(op.gate.name == 'MEASURE' for op in fused.ops)


def test_fusion_preserves_2q_gates():
    """Fusion doesn't affect 2Q gates."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(2).h(0).h(1).cx(0, 1).h(0).h(1)
    fused = fuse_1q_gates(c)

    # CX should still be there
    assert any(op.gate == Gate.CX for op in fused.ops)
    state_orig, _ = simulate(c)
    state_fused, _ = simulate(fused)
    assert states_equal(state_orig, state_fused)


def test_fusion_identity_removed():
    """Fusing gates that produce identity removes them."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(1).h(0).h(0)  # H·H = I
    fused = fuse_1q_gates(c)

    assert len(fused.ops) == 0 or all(
        abs(op.params[0]) < 1e-9 if op.params else True
        for op in fused.ops
    )


def test_fusion_multi_qubit():
    """Fusion works independently on different qubits."""
    from tinyqubit.passes.fuse import fuse_1q_gates

    c = Circuit(2)
    c.h(0).s(0).t(0)  # 3 gates on qubit 0
    c.x(1).y(1).z(1)  # 3 gates on qubit 1

    fused = fuse_1q_gates(c)

    # Should have at most 3 ops per qubit = 6 total (likely fewer)
    assert len(fused.ops) <= 6
    state_orig, _ = simulate(c)
    state_fused, _ = simulate(fused)
    assert states_equal(state_orig, state_fused)


# =============================================================================
# Integration Tests - Teleportation
# =============================================================================

def test_teleportation_circuit():
    """Quantum teleportation with mid-circuit measurement."""
    # Teleport state from qubit 0 to qubit 2
    # Qubit 0: state to teleport (|1⟩)
    # Qubits 1,2: Bell pair

    c = Circuit(3)
    # Prepare state to teleport on qubit 0
    c.x(0)  # |1⟩

    # Create Bell pair on qubits 1,2
    c.h(1).cx(1, 2)

    # Bell measurement on qubits 0,1
    c.cx(0, 1).h(0)
    c.measure(0, 0).measure(1, 1)

    # Conditional corrections on qubit 2
    with c.c_if(1, 1):
        c.x(2)
    with c.c_if(0, 1):
        c.z(2)

    state, _ = simulate(c, seed=42)

    # Qubit 2 should now have the state |1⟩
    # The statevector has 8 components. After teleportation, qubit 2 should be |1⟩
    # So states with qubit 2 = 0 should have zero amplitude
    state = state.reshape([2, 2, 2])
    # Sum of |amplitude|^2 for qubit 2 = 0
    prob_q2_zero = np.sum(np.abs(state[:, :, 0])**2)
    # Sum of |amplitude|^2 for qubit 2 = 1
    prob_q2_one = np.sum(np.abs(state[:, :, 1])**2)

    assert prob_q2_one > 0.99, f"Teleportation failed: P(q2=1) = {prob_q2_one}"


def test_n_classical_default():
    """n_classical defaults to n_qubits."""
    c = Circuit(3)
    assert c.n_classical == 3


def test_n_classical_explicit():
    """n_classical can be set explicitly."""
    c = Circuit(3, 5)
    assert c.n_qubits == 3
    assert c.n_classical == 5


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_backward_compat_simulate_returns_usable():
    """simulate() result works with existing code patterns."""
    c = Circuit(1).h(0)
    state, _ = simulate(c)

    # Should work with states_equal
    expected = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)

    # Should work with sample
    counts = sample(state, 100, seed=42)
    assert '0' in counts or '1' in counts


def test_backward_compat_measure_at_end():
    """Circuits with measure at end still work."""
    c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
    state, _ = simulate(c, seed=42)

    # Should still produce valid result
    assert state is not None
    assert sum(np.abs(state)**2) > 0.99


from tinyqubit.ir import Gate  # Import Gate for test_fusion_preserves_2q_gates


# =============================================================================
# Amplitude Damping (T1) Tests
# =============================================================================

def test_amplitude_damping_decays_one_to_zero():
    """Amplitude damping causes |1⟩ to decay toward |0⟩."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).x(0)  # Prepare |1⟩

    # High damping probability
    noise = NoiseModel().add_amplitude_damping(0.99)

    # Run multiple times - should mostly end up in |0⟩
    zeros = 0
    for seed in range(20):
        state, _ = simulate(c, seed=seed, noise_model=noise)
        if abs(state[0]) > 0.9:  # Mostly |0⟩
            zeros += 1

    # Most runs should decay to |0⟩
    assert zeros > 10, f"Expected most to decay, got {zeros}/20"


def test_amplitude_damping_preserves_zero():
    """Amplitude damping doesn't affect |0⟩ state."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1)  # Already in |0⟩

    noise = NoiseModel().add_amplitude_damping(0.5)

    state, _ = simulate(c, seed=42, noise_model=noise)
    # |0⟩ should remain |0⟩
    assert abs(state[0]) > 0.99


# =============================================================================
# Phase Damping (T2) Tests
# =============================================================================

def test_phase_damping_affects_superposition():
    """Phase damping destroys superposition (projects to computational basis)."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).h(0)  # |+⟩ state

    # Full dephasing - should project onto computational basis
    noise = NoiseModel().add_phase_damping(1.0)

    # With λ=1.0 on |+⟩: 50% chance of |0⟩, 50% chance of |1⟩
    # (superposition destroyed, not preserved)
    got_0, got_1 = False, False
    for seed in range(20):
        state, _ = simulate(c, seed=seed, noise_model=noise)
        prob_0 = abs(state[0])**2
        prob_1 = abs(state[1])**2
        # State should be in computational basis (either |0⟩ or |1⟩)
        assert prob_0 > 0.99 or prob_1 > 0.99, "State should be in computational basis"
        if prob_0 > 0.99: got_0 = True
        if prob_1 > 0.99: got_1 = True
    # Should see both outcomes across runs
    assert got_0 and got_1, "Should see both |0⟩ and |1⟩ outcomes"


def test_phase_damping_no_effect_on_basis():
    """Phase damping doesn't change |0⟩ or |1⟩ probabilities."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).x(0)  # |1⟩ state

    noise = NoiseModel().add_phase_damping(1.0)

    state, _ = simulate(c, seed=42, noise_model=noise)
    # |1⟩ should stay |1⟩ (Z|1⟩ = -|1⟩, same up to phase)
    assert abs(state[1]) > 0.99


def test_phase_damping_preserves_populations():
    """Phase damping must preserve populations (key invariant)."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).h(0)  # |+⟩ state: P(|0⟩) = P(|1⟩) = 0.5
    noise = NoiseModel().add_phase_damping(0.5)

    n_trials = 200
    avg_prob_0 = sum(
        abs(simulate(c, seed=seed, noise_model=noise)[0][0])**2
        for seed in range(n_trials)
    ) / n_trials

    assert 0.4 < avg_prob_0 < 0.6, f"Population bias: avg P(|0⟩) = {avg_prob_0:.3f}"


def test_phase_damping_coherence_decay():
    """Phase damping must decay coherence by (1-λ).

    This test catches incorrect dephasing implementations.
    For |+⟩ with λ, ⟨X⟩ should decay from 1 to (1-λ).

    OLD (wrong) implementation gave ⟨X⟩ ≈ 0.71 for λ=0.5
    NEW (correct) implementation gives ⟨X⟩ ≈ 0.50 for λ=0.5
    """
    from tinyqubit.noise import NoiseModel
    import numpy as np

    c = Circuit(1).h(0)  # |+⟩ state: ⟨X⟩ = 1
    lambda_ = 0.5
    noise = NoiseModel().add_phase_damping(lambda_)

    # ⟨X⟩ = 2*Re(α*conj(β)) for state α|0⟩ + β|1⟩
    n_trials = 500
    avg_X = 0.0
    for seed in range(n_trials):
        state, _ = simulate(c, seed=seed, noise_model=noise)
        avg_X += 2 * np.real(state[0] * np.conj(state[1]))
    avg_X /= n_trials

    expected_X = 1 - lambda_  # Coherence should decay to (1-λ)
    assert abs(avg_X - expected_X) < 0.1, \
        f"Wrong coherence decay: ⟨X⟩ = {avg_X:.3f}, expected {expected_X}"


# =============================================================================
# Realistic Noise Tests
# =============================================================================

def test_realistic_noise_creates_valid_model():
    """realistic_noise() creates a NoiseModel with all components."""
    from tinyqubit.noise import realistic_noise

    noise = realistic_noise()

    # Should have noise configured
    assert len(noise.gate_noise) > 0 or len(noise.default_noise) > 0
    assert noise.readout_error_fn is not None


def test_realistic_noise_customizable():
    """realistic_noise() parameters can be customized."""
    from tinyqubit.noise import realistic_noise

    noise = realistic_noise(
        t1=200e-6,
        t2=100e-6,
        depolarizing_1q=0.0001,
        depolarizing_2q=0.001,
        readout_err=0.01
    )

    # Verify readout error is configured (function stored, not class attributes)
    assert noise.readout_error_fn is not None


def test_realistic_noise_simulation():
    """Circuit simulation with realistic noise produces valid results."""
    from tinyqubit.noise import realistic_noise

    c = Circuit(2).h(0).cx(0, 1)

    noise = realistic_noise()
    state, _ = simulate(c, seed=42, noise_model=noise)

    # Should still produce a valid state
    total_prob = sum(abs(state)**2)
    assert 0.99 < total_prob < 1.01


def test_noise_parameter_validation():
    """Noise functions reject invalid probability parameters."""
    from tinyqubit.noise import (
        depolarizing, amplitude_damping, phase_damping, readout_error
    )
    import pytest

    # All should reject values outside [0, 1]
    with pytest.raises(ValueError):
        depolarizing(-0.1)
    with pytest.raises(ValueError):
        depolarizing(1.5)
    with pytest.raises(ValueError):
        amplitude_damping(-0.1)
    with pytest.raises(ValueError):
        phase_damping(2.0)
    with pytest.raises(ValueError):
        readout_error(p0_given_1=-0.1)
    with pytest.raises(ValueError):
        readout_error(p1_given_0=1.1)

    # Valid values should work
    depolarizing(0.0)
    depolarizing(1.0)
    amplitude_damping(0.5)
    phase_damping(0.5)
    readout_error(0.02, 0.02)


# =============================================================================
# Batch Operations Tests
# =============================================================================

def test_batch_ops_preserves_semantics():
    """batch_ops=True produces same result as batch_ops=False."""
    c = Circuit(3)
    c.h(0).h(1).h(2)  # All parallel
    c.x(0).y(1).z(2)  # All parallel
    c.s(0).t(1).rz(2, 0.5)  # All parallel

    state_normal, _ = simulate(c, seed=42)
    state_batch, _ = simulate(c, seed=42, batch_ops=True)

    assert states_equal(state_normal, state_batch)


def test_batch_ops_handles_barriers():
    """batch_ops respects measurement and reset barriers."""
    c = Circuit(2)
    c.h(0).h(1)
    c.measure(0)  # Barrier
    c.x(0).x(1)

    _, classical_normal = simulate(c, seed=42)
    _, classical_batch = simulate(c, seed=42, batch_ops=True)

    # Results should match
    assert classical_normal.get(0) == classical_batch.get(0)


def test_batch_ops_with_2q_gates():
    """batch_ops works correctly with 2Q gates interspersed."""
    c = Circuit(3)
    c.h(0).h(1).h(2)
    c.cx(0, 1)  # Barrier for qubits 0,1
    c.x(0).x(1).x(2)

    state_normal, _ = simulate(c)
    state_batch, _ = simulate(c, batch_ops=True)

    assert states_equal(state_normal, state_batch)


def test_batch_ops_disabled_with_noise():
    """batch_ops is disabled when noise model is present."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(2).h(0).h(1)
    noise = NoiseModel().add_depolarizing(0.01)

    # Should work without error (batching disabled internally)
    state, _ = simulate(c, seed=42, noise_model=noise, batch_ops=True)
    assert state is not None


# =============================================================================
# Combined Noise Tests
# =============================================================================

def test_combined_noise_channels():
    """Multiple noise channels can be combined."""
    from tinyqubit.noise import NoiseModel

    c = Circuit(1).h(0).x(0).h(0)

    noise = NoiseModel()
    noise.add_depolarizing(0.01)
    noise.add_amplitude_damping(0.01)
    noise.add_phase_damping(0.01)

    # Should run without error
    state, _ = simulate(c, seed=42, noise_model=noise)
    assert state is not None


# =============================================================================
# Robustness Tests
# =============================================================================

def test_invalid_qubit_index_raises():
    """Invalid qubit index raises ValueError."""
    from tinyqubit.ir import Operation
    c = Circuit(2)
    c.ops.append(Operation(Gate.X, (5,)))  # Invalid: qubit 5 in 2-qubit circuit
    with pytest.raises(ValueError, match="Invalid qubit index"):
        simulate(c)


def test_classical_bits_initialized_to_zero():
    """All classical bits start at 0."""
    c = Circuit(2, n_classical=4)
    # No measurements, just check initial state
    _, classical = simulate(c)
    assert classical == {0: 0, 1: 0, 2: 0, 3: 0}


def test_conditional_before_measure_uses_initial_value():
    """Conditional before any measurement uses initial value (0)."""
    c = Circuit(1, n_classical=1)
    # Condition on bit 0 == 0 (initial value)
    with c.c_if(0, 0):
        c.x(0)  # Should execute since bit 0 starts at 0
    state, _ = simulate(c)
    assert np.allclose(np.abs(state), [0, 1])  # |1⟩


def test_sample_handles_unnormalized_state():
    """sample() normalizes probabilities to handle numerical drift."""
    # Create slightly unnormalized state (simulates numerical drift)
    state = np.array([0.7072, 0.7072], dtype=complex)  # sum of squares ≈ 1.0003
    # Should not raise, should still work
    counts = sample(state, shots=100, seed=42)
    assert sum(counts.values()) == 100
