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
    state = simulate(c)

    expected = np.array([SQRT2_INV, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_bell_state():
    """Bell state: (|00⟩ + |11⟩) / √2"""
    c = Circuit(2).h(0).cx(0, 1)
    state = simulate(c)

    expected = np.array([SQRT2_INV, 0, 0, SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_ghz_state():
    """GHZ state: (|000⟩ + |111⟩) / √2"""
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    state = simulate(c)

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
    state = simulate(c)

    expected = np.array([0, 1], dtype=complex)
    assert states_equal(state, expected)


def test_y_gate():
    """Y|0⟩ = i|1⟩"""
    c = Circuit(1).y(0)
    state = simulate(c)

    expected = np.array([0, 1j], dtype=complex)
    assert states_equal(state, expected)


def test_z_gate():
    """Z|+⟩ = |−⟩"""
    c = Circuit(1).h(0).z(0)
    state = simulate(c)

    # |−⟩ = (|0⟩ - |1⟩) / √2
    expected = np.array([SQRT2_INV, -SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


def test_h_gate_twice():
    """H·H = I"""
    c = Circuit(1).h(0).h(0)
    state = simulate(c)

    expected = np.array([1, 0], dtype=complex)
    assert states_equal(state, expected)


def test_s_gate():
    """S|1⟩ = i|1⟩"""
    c = Circuit(1).x(0).s(0)
    state = simulate(c)

    expected = np.array([0, 1j], dtype=complex)
    assert states_equal(state, expected)


def test_t_gate():
    """T|1⟩ = e^(iπ/4)|1⟩"""
    c = Circuit(1).x(0).t(0)
    state = simulate(c)

    expected = np.array([0, np.exp(1j * pi / 4)], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Rotation Gate Tests
# =============================================================================

def test_rx_pi():
    """RX(π)|0⟩ = -i|1⟩"""
    c = Circuit(1).rx(0, pi)
    state = simulate(c)

    expected = np.array([0, -1j], dtype=complex)
    assert states_equal(state, expected)


def test_ry_pi():
    """RY(π)|0⟩ = |1⟩"""
    c = Circuit(1).ry(0, pi)
    state = simulate(c)

    expected = np.array([0, 1], dtype=complex)
    assert states_equal(state, expected)


def test_rz_pi():
    """RZ(π)|+⟩ = |−⟩ (up to global phase)"""
    c = Circuit(1).h(0).rz(0, pi)
    state = simulate(c)

    # RZ(π) = -iZ, so RZ(π)|+⟩ = -i|−⟩
    expected = np.array([SQRT2_INV, -SQRT2_INV], dtype=complex)
    assert states_equal(state, expected)


# =============================================================================
# Two Qubit Gate Tests
# =============================================================================

def test_cx_control_zero():
    """CX with control=0 does nothing"""
    c = Circuit(2).cx(0, 1)  # control q0=|0⟩
    state = simulate(c)

    expected = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    assert states_equal(state, expected)


def test_cx_control_one():
    """CX with control=1 flips target"""
    c = Circuit(2).x(0).cx(0, 1)  # control q0=|1⟩
    state = simulate(c)

    expected = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
    assert states_equal(state, expected)


def test_cz_gate():
    """CZ|11⟩ = -|11⟩"""
    c = Circuit(2).x(0).x(1).cz(0, 1)
    state = simulate(c)

    expected = np.array([0, 0, 0, -1], dtype=complex)  # -|11⟩
    assert states_equal(state, expected)


def test_cz_plus_states():
    """CZ on |++⟩ creates |Φ+⟩-like state"""
    c = Circuit(2).h(0).h(1).cz(0, 1)
    state = simulate(c)

    # |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    # CZ flips phase of |11⟩
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    assert states_equal(state, expected)


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
    state = simulate(c)

    counts1 = sample(state, 1000, seed=123)
    counts2 = sample(state, 1000, seed=123)

    assert counts1 == counts2


def test_sample_bell_distribution():
    """Bell state gives roughly 50/50 |00⟩ and |11⟩"""
    c = Circuit(2).h(0).cx(0, 1)
    state = simulate(c)

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
    state = simulate(c)

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

    state = simulate(c)
    counts = sample(state, 1000, seed=42)

    # |11⟩ should dominate
    assert counts.get('11', 0) > 900


def test_measure_ignored():
    """MEASURE operations don't affect statevector"""
    c1 = Circuit(2).h(0).cx(0, 1)
    c2 = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)

    state1 = simulate(c1)
    state2 = simulate(c2)

    assert states_equal(state1, state2)


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
    our_state = simulate(c)

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
    our_state = simulate(c)

    # Qiskit
    qc = QuantumCircuit(1)
    qc.rx(1.23, 0)
    qc.ry(2.34, 0)
    qc.rz(3.45, 0)
    qiskit_state = Statevector.from_instruction(qc).data

    assert states_equal(our_state, qiskit_state)
