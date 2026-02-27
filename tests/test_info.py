import numpy as np
import pytest
from tinyqubit import state_fidelity, partial_trace, entanglement_entropy, concurrence, mutual_information


BELL = np.array([1, 0, 0, 1]) / np.sqrt(2)
KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1]) / np.sqrt(2)
KET00 = np.array([1, 0, 0, 0], dtype=complex)


# --- Fidelity ---

def test_fidelity_identical():
    assert state_fidelity(KET0, KET0) == pytest.approx(1.0)

def test_fidelity_orthogonal():
    assert state_fidelity(KET0, KET1) == pytest.approx(0.0)

def test_fidelity_overlap():
    assert state_fidelity(KET0, PLUS) == pytest.approx(0.5)


# --- Partial trace ---

def test_partial_trace_product():
    rho = partial_trace(KET00, keep=[0])
    assert rho == pytest.approx(np.array([[1, 0], [0, 0]]))

def test_partial_trace_bell():
    rho = partial_trace(BELL, keep=[0])
    assert rho == pytest.approx(np.eye(2) / 2)


# --- Entanglement entropy ---

def test_entropy_product():
    assert entanglement_entropy(KET00, [0]) == pytest.approx(0.0)

def test_entropy_bell():
    assert entanglement_entropy(BELL, [0]) == pytest.approx(1.0)


# --- Concurrence ---

def test_concurrence_product():
    assert concurrence(KET00) == pytest.approx(0.0)

def test_concurrence_bell():
    assert concurrence(BELL) == pytest.approx(1.0)


# --- Mutual information ---

def test_mutual_information_bell():
    assert mutual_information(BELL, [0], [1]) == pytest.approx(2.0)

def test_mutual_information_product():
    assert mutual_information(KET00, [0], [1]) == pytest.approx(0.0)
