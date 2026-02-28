"""Tests for optimizers: GradientDescent, Adam, SPSA."""
import numpy as np
from tinyqubit import Circuit, Parameter, Z, parameter_shift_gradient
from tinyqubit.qml import GradientDescent, Adam, SPSA


def _ry_z_circuit():
    qc = Circuit(1)
    qc.ry(0, Parameter('theta'))
    return qc, Z(0)


def test_gd_convergence():
    qc, obs = _ry_z_circuit()
    opt = GradientDescent(stepsize=0.4)
    params = {'theta': 0.5}
    for _ in range(50):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.99


def test_adam_convergence():
    qc, obs = _ry_z_circuit()
    opt = Adam(stepsize=0.1)
    params = {'theta': 0.5}
    for _ in range(100):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.99


def test_spsa_convergence_exact():
    qc, obs = _ry_z_circuit()
    opt = SPSA(stepsize=0.1, perturbation=0.1, seed=42)
    params = {'theta': 0.5}
    for _ in range(200):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.9


def test_spsa_determinism():
    qc, obs = _ry_z_circuit()
    results = []
    for _ in range(2):
        opt = SPSA(stepsize=0.1, perturbation=0.1, seed=123)
        params = {'theta': 0.5}
        for _ in range(10):
            params = opt.step(params, qc, obs)
        results.append(params['theta'])
    assert results[0] == results[1]


def test_gd_custom_grad_fn():
    qc, obs = _ry_z_circuit()
    opt = GradientDescent(stepsize=0.4, grad_fn=parameter_shift_gradient)
    params = {'theta': 0.5}
    for _ in range(50):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.99


def test_adam_multi_param():
    qc = Circuit(2)
    qc.ry(0, Parameter('a'))
    qc.ry(1, Parameter('b'))
    obs = Z(0) + Z(1)
    opt = Adam(stepsize=0.1)
    params = {'a': 0.5, 'b': 1.0}
    for _ in range(100):
        params = opt.step(params, qc, obs)
    assert np.cos(params['a']) < -0.99
    assert np.cos(params['b']) < -0.99


def test_spsa_with_shots():
    qc, obs = _ry_z_circuit()
    opt = SPSA(stepsize=0.1, perturbation=0.2, shots=1000, seed=42)
    params = {'theta': 0.5}
    for _ in range(300):
        params = opt.step(params, qc, obs)
    assert np.cos(params['theta']) < -0.7
