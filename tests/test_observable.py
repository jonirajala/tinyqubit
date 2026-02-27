"""Tests for observable.py — Pauli observables and expectation values."""
import numpy as np
from math import pi
from tinyqubit import Circuit
from tinyqubit.observable import Observable, X, Y, Z, expectation


# --- Observable algebra ---

def test_single_pauli():
    assert len(Z(0).terms) == 1
    assert Z(0).terms[0] == (1.0, {0: 'Z'})

def test_add():
    obs = Z(0) + X(1)
    assert len(obs.terms) == 2

def test_sub():
    obs = Z(0) - Z(1)
    assert len(obs.terms) == 2
    assert obs.terms[1][0] == -1.0

def test_scalar_mul():
    obs = 0.5 * Z(0)
    assert obs.terms[0][0] == 0.5
    obs2 = Z(0) * 0.5
    assert obs2.terms[0][0] == 0.5

def test_neg():
    obs = -Z(0)
    assert obs.terms[0][0] == -1.0

def test_tensor_product():
    obs = Z(0) @ Z(1)
    assert len(obs.terms) == 1
    assert obs.terms[0] == (1.0, {0: 'Z', 1: 'Z'})

def test_tensor_product_overlap_raises():
    try:
        X(0) @ Z(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_sum_identity():
    """sum() starts with 0, so __radd__ with 0 must work."""
    obs = sum([Z(0), Z(1)])
    assert len(obs.terms) == 2

def test_repr():
    r = repr(Z(0) @ Z(1))
    assert "Z(0)" in r and "Z(1)" in r

def test_complex_observable():
    obs = 0.5 * (Z(0) @ Z(1)) + 0.3 * X(0)
    assert len(obs.terms) == 2


# --- Expectation values ---

def test_z_on_zero():
    """⟨0|Z|0⟩ = 1"""
    qc = Circuit(1)
    assert np.isclose(expectation(qc, Z(0)), 1.0)

def test_z_on_one():
    """⟨1|Z|1⟩ = -1"""
    qc = Circuit(1).x(0)
    assert np.isclose(expectation(qc, Z(0)), -1.0)

def test_x_on_plus():
    """⟨+|X|+⟩ = 1"""
    qc = Circuit(1).h(0)
    assert np.isclose(expectation(qc, X(0)), 1.0)

def test_z_on_plus():
    """⟨+|Z|+⟩ = 0"""
    qc = Circuit(1).h(0)
    assert np.isclose(expectation(qc, Z(0)), 0.0, atol=1e-10)

def test_y_on_eigenstate():
    """RX(pi/2)|0⟩ should give ⟨Y⟩ = -1"""
    qc = Circuit(1).rx(0, pi / 2)
    assert np.isclose(expectation(qc, Y(0)), -1.0)

def test_bell_zz():
    """⟨Bell|Z⊗Z|Bell⟩ = 1"""
    qc = Circuit(2).h(0).cx(0, 1)
    assert np.isclose(expectation(qc, Z(0) @ Z(1)), 1.0)

def test_bell_xx():
    """⟨Bell|X⊗X|Bell⟩ = 1"""
    qc = Circuit(2).h(0).cx(0, 1)
    assert np.isclose(expectation(qc, X(0) @ X(1)), 1.0)

def test_bell_single_z():
    """⟨Bell|Z⊗I|Bell⟩ = 0 (maximally mixed marginal)"""
    qc = Circuit(2).h(0).cx(0, 1)
    assert np.isclose(expectation(qc, Z(0)), 0.0, atol=1e-10)

def test_linearity():
    """E[aA + bB] = a*E[A] + b*E[B]"""
    qc = Circuit(2).h(0).cx(0, 1)
    a, b = 0.3, 0.7
    combined = a * Z(0) @ Z(1) + b * X(0) @ X(1)
    separate = a * expectation(qc, Z(0) @ Z(1)) + b * expectation(qc, X(0) @ X(1))
    assert np.isclose(expectation(qc, combined), separate)

def test_identity_term():
    """Observable with empty Pauli dict acts as identity: ⟨ψ|I|ψ⟩ = 1"""
    qc = Circuit(1).h(0)
    obs = Observable([(2.0, {})])
    assert np.isclose(expectation(qc, obs), 2.0)

def test_multi_qubit_z():
    """Z on qubit 1 of a 3-qubit circuit with X on qubit 1"""
    qc = Circuit(3).x(1)
    assert np.isclose(expectation(qc, Z(1)), -1.0)
    assert np.isclose(expectation(qc, Z(0)), 1.0)
    assert np.isclose(expectation(qc, Z(2)), 1.0)

def test_parametric_circuit():
    """Expectation of RY circuit: ⟨Z⟩ = cos(θ)"""
    theta = 1.23
    qc = Circuit(1).ry(0, theta)
    assert np.isclose(expectation(qc, Z(0)), np.cos(theta))
