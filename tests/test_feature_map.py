"""Tests for feature_map.py — data feature map circuits."""
import numpy as np
from math import pi
from tinyqubit import Circuit, Gate, simulate, probabilities
from tinyqubit.qml.feature_map import angle_feature_map, basis_feature_map, amplitude_feature_map, zz_feature_map


# --- Angle feature map ---

def test_angle_feature_map():
    qc = Circuit(3)
    angle_feature_map(qc, [0.1, 0.2, 0.3], wires=[0, 1, 2])
    assert len(qc.ops) == 3
    assert all(op.gate == Gate.RY for op in qc.ops)

def test_angle_feature_map_rx():
    qc = Circuit(2)
    angle_feature_map(qc, [0.5, 1.0], wires=[0, 1], rotation=Gate.RX)
    assert all(op.gate == Gate.RX for op in qc.ops)

def test_angle_feature_map_values():
    """RY(theta)|0⟩ = cos(t/2)|0⟩ + sin(t/2)|1⟩"""
    theta = 1.2
    qc = Circuit(1)
    angle_feature_map(qc, [theta], wires=[0])
    sv, _ = simulate(qc)
    assert np.isclose(abs(sv[0]) ** 2, np.cos(theta / 2) ** 2)
    assert np.isclose(abs(sv[1]) ** 2, np.sin(theta / 2) ** 2)


# --- Basis feature map ---

def test_basis_feature_map():
    qc = Circuit(4)
    basis_feature_map(qc, [1, 0, 1, 0], wires=[0, 1, 2, 3])
    assert len(qc.ops) == 2
    assert all(op.gate == Gate.X for op in qc.ops)
    assert qc.ops[0].qubits == (0,)
    assert qc.ops[1].qubits == (2,)

def test_basis_feature_map_all_zeros():
    qc = Circuit(3)
    basis_feature_map(qc, [0, 0, 0], wires=[0, 1, 2])
    assert len(qc.ops) == 0


# --- IQP feature map ---

def test_zz_feature_map_structure():
    """3 wires, 2 reps: each rep has 3 H + 3 RZ + 3 CZ + 3 RZ = 12 ops per rep."""
    qc = Circuit(3)
    zz_feature_map(qc, [0.1, 0.2, 0.3], wires=[0, 1, 2], reps=2)
    n_wires = 3
    n_pairs = n_wires * (n_wires - 1) // 2  # 3
    ops_per_rep = n_wires + n_wires + n_pairs + n_pairs  # H + RZ + CZ + RZ
    assert len(qc.ops) == ops_per_rep * 2

def test_zz_feature_map_deterministic():
    features = [0.5, 1.0, 1.5]
    sv1, _ = simulate(Circuit(3).h(0))  # dummy to warm up
    qc1, qc2 = Circuit(3), Circuit(3)
    zz_feature_map(qc1, features, wires=[0, 1, 2])
    zz_feature_map(qc2, features, wires=[0, 1, 2])
    sv1, _ = simulate(qc1)
    sv2, _ = simulate(qc2)
    assert np.allclose(sv1, sv2)


# --- Amplitude feature map ---

def test_amplitude_feature_map():
    features = [0.5, 0.5, 0.5, 0.5]
    qc = Circuit(2)
    amplitude_feature_map(qc, features, wires=[0, 1])
    sv, _ = simulate(qc)
    # After normalization, each amplitude = 0.5
    assert np.allclose(np.abs(sv) ** 2, [0.25] * 4)

def test_amplitude_feature_map_partial_wires():
    """Encode into wires [1,2] of a 3-qubit circuit. Wire 0 stays |0⟩."""
    features = [1.0, 0.0, 0.0, 0.0]  # |00⟩ on wires 1,2
    qc = Circuit(3)
    amplitude_feature_map(qc, features, wires=[1, 2])
    sv, _ = simulate(qc)
    # All amplitude on |000⟩
    assert np.isclose(abs(sv[0]) ** 2, 1.0)
