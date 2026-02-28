"""Cost functions for variational quantum circuits."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .observable import Observable, Z, expectation
from .simulator import simulate
from .info import state_fidelity


def cross_entropy_cost(circuit: Circuit, X: np.ndarray, y: np.ndarray,
                       observable: Observable | None = None) -> float:
    """Binary cross-entropy. y ∈ {0,1}, prediction p = (1+⟨O⟩)/2."""
    obs = observable or Z(0)
    names = sorted(p.name for p in circuit.parameters)
    total = 0.0
    for xi, yi in zip(X, y):
        p = np.clip((1 + expectation(circuit.bind(dict(zip(names, xi))), obs)) / 2, 1e-12, 1 - 1e-12)
        total -= yi * np.log(p) + (1 - yi) * np.log(1 - p)
    return total / len(X)


def mse_cost(circuit: Circuit, X: np.ndarray, y: np.ndarray,
             observable: Observable | None = None) -> float:
    """Mean squared error: mean((⟨O⟩ᵢ - yᵢ)²)."""
    obs = observable or Z(0)
    names = sorted(p.name for p in circuit.parameters)
    return sum((expectation(circuit.bind(dict(zip(names, xi))), obs) - yi) ** 2
               for xi, yi in zip(X, y)) / len(X)


def fidelity_cost(circuit: Circuit, target_state: np.ndarray) -> float:
    """State preparation cost: 1 - |⟨ψ|target⟩|²."""
    state, _ = simulate(circuit)
    return 1 - state_fidelity(state, target_state)
