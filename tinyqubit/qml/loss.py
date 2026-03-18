"""Loss functions for variational quantum circuits."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit
from ..measurement.observable import Observable, Z, expectation, state_fidelity
from ..simulator import simulate


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


class _DiffLoss:
    """Callable loss with analytical gradient for use with backprop_gradient."""
    __slots__ = ('_fn', 'grad')
    def __init__(self, fn, grad): self._fn, self.grad = fn, grad
    def __call__(self, *args): return self._fn(*args)


def _kl(p, q):
    mask = p > 1e-12
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def kl_divergence(target: np.ndarray, q: np.ndarray | None = None):
    """KL(target || q). With one arg, returns a loss function for use with backprop_gradient."""
    if q is not None: return _kl(target, q)
    return _DiffLoss(lambda q: _kl(target, q),
                     lambda q: np.where(target > 1e-12, -target / np.clip(q, 1e-12, None), 0.0))


def mse(target: np.ndarray):
    """MSE loss factory: returns loss(probs) -> float."""
    return _DiffLoss(lambda p: float(np.sum((p - target) ** 2)),
                     lambda p: 2 * (p - target))


def fidelity_cost(circuit: Circuit, target_state: np.ndarray) -> float:
    """State preparation cost: 1 - |⟨ψ|target⟩|²."""
    state, _ = simulate(circuit)
    return 1 - state_fidelity(state, target_state)
