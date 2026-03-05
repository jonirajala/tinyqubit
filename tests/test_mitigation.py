"""Tests for error mitigation: ZNE and readout mitigation."""
import numpy as np
from tinyqubit import Circuit, NoiseModel, Observable, Z, sample, simulate, zne, calibration_matrix, mitigate_readout
from tinyqubit.mitigation import _fold_circuit


def test_fold_circuit_lengths():
    c = Circuit(2)
    c.h(0).cx(0, 1).rz(0, 0.5).x(1).z(0)
    assert len(c.ops) == 5
    assert len(_fold_circuit(c, 1).ops) == 5
    assert len(_fold_circuit(c, 3).ops) == 15
    assert len(_fold_circuit(c, 5).ops) == 25


def test_zne_noiseless():
    c = Circuit(1)
    c.x(0)
    noise = NoiseModel().add_depolarizing(1e-10)
    result = zne(c, Z(0), noise, seed=42)
    assert abs(result - (-1.0)) < 0.1


def test_zne_improves_estimate():
    c = Circuit(1)
    c.h(0)
    noise = NoiseModel().add_depolarizing(0.05)
    # Ideal: ⟨H|Z|H⟩ = 0
    raw = zne(c, Z(0), noise, scale_factors=[1], seed=42)
    mitigated = zne(c, Z(0), noise, scale_factors=[1, 3, 5], seed=42)
    assert abs(mitigated) <= abs(raw) + 0.05  # ZNE should be at least as good (with tolerance)


def test_zne_shots():
    c = Circuit(1)
    c.x(0)
    noise = NoiseModel().add_depolarizing(0.01)
    result = zne(c, Z(0), noise, shots=200, seed=42)
    assert abs(result - (-1.0)) < 0.3


def test_calibration_matrix_ideal():
    noise = NoiseModel()
    cal = calibration_matrix(1, noise, shots=100, seed=42)
    np.testing.assert_allclose(cal, np.eye(2), atol=1e-10)


def test_calibration_matrix_rows_sum_to_one():
    noise = NoiseModel().add_readout_error(0.1, 0.1)
    cal = calibration_matrix(1, noise, shots=500, seed=42)
    np.testing.assert_allclose(cal.sum(axis=1), [1.0, 1.0], atol=1e-10)


def test_mitigate_readout_corrects():
    noise = NoiseModel().add_readout_error(0.15, 0.15)
    # Prepare |1⟩ and measure with readout error
    c = Circuit(1)
    c.x(0).measure(0, 0)
    counts = {}
    rng = np.random.default_rng(42)
    for _ in range(1000):
        _, cl = simulate(c, seed=int(rng.integers(2**32)), noise_model=noise)
        bs = str(cl[0])
        counts[bs] = counts.get(bs, 0) + 1
    cal = calibration_matrix(1, noise, shots=1000, seed=123)
    corrected = mitigate_readout(counts, cal)
    assert corrected.get('1', 0) > 0.85


def test_mitigate_readout_identity_cal():
    counts = {'00': 250, '01': 250, '10': 250, '11': 250}
    cal = np.eye(4)
    corrected = mitigate_readout(counts, cal)
    for bs in ['00', '01', '10', '11']:
        assert abs(corrected[bs] - 0.25) < 1e-10
