"""Quantum kernel methods — compute similarity in Hilbert space."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit
from ..simulator import simulate
from ..analysis.observable import state_fidelity


def quantum_kernel(feature_map_fn, x1, x2, qubits=None) -> float:
    """Compute K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|² for a given feature map."""
    qubits = qubits or list(range(len(x1)))
    c1 = feature_map_fn(qubits, x1)
    c2 = feature_map_fn(qubits, x2)
    sv1, _ = simulate(c1)
    sv2, _ = simulate(c2)
    return state_fidelity(sv1, sv2)


def _statevectors(feature_map_fn, X, qubits):
    svs = []
    for x in X:
        c = feature_map_fn(qubits, x)
        sv, _ = simulate(c)
        svs.append(sv)
    return svs


def kernel_matrix(feature_map_fn, X, X2=None, qubits=None) -> np.ndarray:
    """Gram matrix of pairwise kernel values. Symmetric N×N if X2 is None, else rectangular N1×N2."""
    qubits = qubits or list(range(len(X[0])))
    svs = _statevectors(feature_map_fn, X, qubits)
    if X2 is None:
        n = len(X)
        K = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = state_fidelity(svs[i], svs[j])
        return K
    svs2 = _statevectors(feature_map_fn, X2, qubits)
    return np.array([[state_fidelity(s1, s2) for s2 in svs2] for s1 in svs])
