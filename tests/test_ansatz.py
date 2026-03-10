"""Tests for feature maps and ansatz templates."""
import numpy as np
from tinyqubit import Circuit, Gate, Parameter, simulate
from tinyqubit.qml.layers import zz_feature_map, pauli_feature_map, strongly_entangling_layers, basic_entangler_layers


# --- Pauli feature map ---

def test_pauli_feature_map_z():
    """paulis='Z' should produce same circuit as iqp_encoding."""
    features = [0.5, 1.0, 1.5]
    qc1 = zz_feature_map([0, 1, 2], features)
    qc2 = pauli_feature_map([0, 1, 2], features, paulis="Z")
    sv1, _ = simulate(qc1)
    sv2, _ = simulate(qc2)
    assert np.allclose(sv1, sv2)

def test_pauli_feature_map_x():
    qc = pauli_feature_map([0, 1, 2], [0.1, 0.2, 0.3], paulis="X", reps=1)
    rx_ops = [op for op in qc.ops if op.gate == Gate.RX]
    assert len(rx_ops) == 3

def test_pauli_feature_map_mixed():
    qc = pauli_feature_map([0, 1, 2], [0.1, 0.2, 0.3], paulis="XYZ", reps=1)
    assert qc.ops[3].gate == Gate.RX   # first rotation after 3 H gates
    assert qc.ops[4].gate == Gate.RY
    assert qc.ops[5].gate == Gate.RZ


# --- Strongly entangling layers ---

def test_strongly_entangling_structure():
    qc = strongly_entangling_layers(4, n_layers=3)
    n_qubits, n_layers = 4, 3
    expected = n_layers * (2 * n_qubits + n_qubits)  # RY + RZ + CX per layer
    assert len(qc.ops) == expected

def test_strongly_entangling_parameters():
    qc = strongly_entangling_layers(3, n_layers=2)
    params = [op.params[0] for op in qc.ops if op.params and isinstance(op.params[0], Parameter)]
    assert len(params) == 2 * 3 * 2  # 2 params * 3 qubits * 2 layers

def test_strongly_entangling_2_qubits():
    """Regression: 2 qubits with layer offset must not produce self-CX."""
    qc = strongly_entangling_layers(2, n_layers=2)
    for op in qc.ops:
        if op.gate == Gate.CX:
            assert op.qubits[0] != op.qubits[1]

def test_strongly_entangling_deterministic():
    def build():
        qc = strongly_entangling_layers(3, n_layers=2)
        bindings = {p.params[0].name: 0.5 for p in qc.ops if p.params and isinstance(p.params[0], Parameter)}
        return simulate(qc.bind(bindings))
    sv1, _ = build()
    sv2, _ = build()
    assert np.allclose(sv1, sv2)


# --- Basic entangler layers ---

def test_basic_entangler_structure():
    qc = basic_entangler_layers(4, n_layers=3)
    n_qubits, n_layers = 4, 3
    expected = n_layers * (n_qubits + (n_qubits - 1))  # RY + CX per layer
    assert len(qc.ops) == expected

def test_basic_entangler_parameters():
    qc = basic_entangler_layers(3, n_layers=2)
    params = [op.params[0] for op in qc.ops if op.params and isinstance(op.params[0], Parameter)]
    assert len(params) == 3 * 2  # n_qubits * n_layers

def test_basic_entangler_trainable():
    """Bind parameters and simulate — result should not be |0...0⟩."""
    qc = basic_entangler_layers(3, n_layers=2)
    bindings = {p.params[0].name: 1.0 for p in qc.ops if p.params and isinstance(p.params[0], Parameter)}
    sv, _ = simulate(qc.bind(bindings))
    assert not np.isclose(abs(sv[0]) ** 2, 1.0)
