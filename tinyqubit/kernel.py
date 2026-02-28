"""Quantum kernel methods — compute similarity in Hilbert space."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .simulator import simulate
from .info import state_fidelity


def quantum_kernel(feature_map_fn, x1, x2, n_qubits=None, wires=None) -> float:
    """Compute K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|² for a given feature map."""
    n_qubits = n_qubits or len(x1)
    wires = wires or list(range(n_qubits))
    c1, c2 = Circuit(n_qubits), Circuit(n_qubits)
    feature_map_fn(c1, x1, wires)
    feature_map_fn(c2, x2, wires)
    sv1, _ = simulate(c1)
    sv2, _ = simulate(c2)
    return state_fidelity(sv1, sv2)


def _statevectors(feature_map_fn, X, n_qubits, wires):
    svs = []
    for x in X:
        c = Circuit(n_qubits)
        feature_map_fn(c, x, wires)
        sv, _ = simulate(c)
        svs.append(sv)
    return svs


def kernel_matrix(feature_map_fn, X, X2=None, n_qubits=None, wires=None) -> np.ndarray:
    """Gram matrix of pairwise kernel values. Symmetric N×N if X2 is None, else rectangular N1×N2."""
    n_qubits = n_qubits or len(X[0])
    wires = wires or list(range(n_qubits))
    svs = _statevectors(feature_map_fn, X, n_qubits, wires)
    if X2 is None:
        n = len(X)
        K = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = state_fidelity(svs[i], svs[j])
        return K
    svs2 = _statevectors(feature_map_fn, X2, n_qubits, wires)
    return np.array([[state_fidelity(s1, s2) for s2 in svs2] for s1 in svs])
