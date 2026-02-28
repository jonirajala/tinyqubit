"""Cost functions for variational quantum circuits."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit
from ..observable import Observable, Z, expectation
from ..simulator import simulate
from ..info import state_fidelity


def predict(circuit: Circuit, X: np.ndarray, observable: Observable | None = None) -> np.ndarray:
    """Return ⟨O⟩ for each sample in X. Features are bound by sorted parameter name."""
    obs = observable or Z(0)
    names = sorted(p.name for p in circuit.parameters)
    return np.array([expectation(circuit.bind(dict(zip(names, xi))), obs) for xi in X])


def cross_entropy_cost(circuit: Circuit, X: np.ndarray, y: np.ndarray,
                       observable: Observable | None = None) -> float:
    """Binary cross-entropy. y ∈ {0,1} or {-1,+1}, prediction p = (1+⟨O⟩)/2."""
    y = (y + 1) / 2 if y.min() < 0 else y  # accept {-1,+1} or {0,1}
    p = np.clip((1 + predict(circuit, X, observable)) / 2, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def mse_cost(circuit: Circuit, X: np.ndarray, y: np.ndarray,
             observable: Observable | None = None) -> float:
    """Mean squared error: mean((⟨O⟩ᵢ - yᵢ)²)."""
    return np.mean((predict(circuit, X, observable) - y) ** 2)


def fidelity_cost(circuit: Circuit, target_state: np.ndarray) -> float:
    """State preparation cost: 1 - |⟨ψ|target⟩|²."""
    state, _ = simulate(circuit)
    return 1 - state_fidelity(state, target_state)
