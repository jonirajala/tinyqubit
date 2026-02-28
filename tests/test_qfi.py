"""Tests for Quantum Fisher Information and QNG optimizer."""
import numpy as np
from tinyqubit import Circuit, Parameter, Z, quantum_fisher_information
from tinyqubit.qml import QNG


def _ry_z_circuit():
    qc = Circuit(1)
    qc.ry(0, Parameter('theta'))
    return qc, Z(0)


def test_qfi_shape():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.ry(1, Parameter('b'))
    F = quantum_fisher_information(qc, {'a': 0.5, 'b': 1.0})
    assert F.shape == (2, 2)


def test_qfi_symmetric():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.rx(1, Parameter('b'))
    qc.cx(0, 1)
    F = quantum_fisher_information(qc, {'a': 0.5, 'b': 1.0})
    np.testing.assert_allclose(F, F.T, atol=1e-12)


def test_qfi_psd():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.cx(0, 1)
    qc.rz(1, Parameter('b'))
    F = quantum_fisher_information(qc, {'a': 0.3, 'b': 0.7})
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-12)


def test_qfi_single_ry():
    # RY(θ)|0⟩ has QFI = 1 for any θ (standard result)
    qc = Circuit(1)
    qc.ry(0, Parameter('theta'))
    F = quantum_fisher_information(qc, {'theta': 0.7})
    np.testing.assert_allclose(F[0, 0], 1.0, atol=1e-10)


def test_qfi_deterministic():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.rx(1, Parameter('b'))
    vals = {'a': 0.5, 'b': 1.0}
    F1 = quantum_fisher_information(qc, vals)
    F2 = quantum_fisher_information(qc, vals)
    np.testing.assert_array_equal(F1, F2)


def test_qng_convergence():
    qc, obs = _ry_z_circuit()
    opt = QNG(stepsize=0.1)
    params = {'theta': 0.5}
    for _ in range(50):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.99


def test_qng_multi_param():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.ry(1, Parameter('b'))
    obs = Z(0) + Z(1)
    opt = QNG(stepsize=0.1)
    params = {'a': 0.5, 'b': 1.0}
    for _ in range(50):
        params = opt.step(params, qc, obs)
    assert np.cos(params['a']) < -0.99
    assert np.cos(params['b']) < -0.99
