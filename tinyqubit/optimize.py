"""Optimizers for variational circuits: GradientDescent, Adam, SPSA."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .observable import Observable, expectation
from .gradient import adjoint_gradient, _shot_expectation


class GradientDescent:
    """Vanilla gradient descent."""
    def __init__(self, stepsize: float = 0.1, grad_fn=None):
        self.stepsize = stepsize
        self._grad_fn = grad_fn or adjoint_gradient

    def step(self, params: dict[str, float], circuit: Circuit, observable: Observable) -> dict[str, float]:
        grad = self._grad_fn(circuit, observable, params)
        return {k: params[k] - self.stepsize * grad[k] for k in params}


class Adam:
    """Adam optimizer with bias-corrected moments."""
    def __init__(self, stepsize: float = 0.01, grad_fn=None,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.stepsize, self.beta1, self.beta2, self.eps = stepsize, beta1, beta2, eps
        self._grad_fn = grad_fn or adjoint_gradient
        self._m: dict[str, float] = {}
        self._v: dict[str, float] = {}
        self._t = 0

    def step(self, params: dict[str, float], circuit: Circuit, observable: Observable) -> dict[str, float]:
        grad = self._grad_fn(circuit, observable, params)
        self._t += 1
        result = {}
        for k in params:
            self._m[k] = self.beta1 * self._m.get(k, 0) + (1 - self.beta1) * grad[k]
            self._v[k] = self.beta2 * self._v.get(k, 0) + (1 - self.beta2) * grad[k] ** 2
            m_hat = self._m[k] / (1 - self.beta1 ** self._t)
            v_hat = self._v[k] / (1 - self.beta2 ** self._t)
            result[k] = params[k] - self.stepsize * m_hat / (np.sqrt(v_hat) + self.eps)
        return result


class SPSA:
    """Simultaneous Perturbation Stochastic Approximation — gradient-free, works with shot noise."""
    def __init__(self, stepsize: float = 0.1, perturbation: float = 0.1,
                 shots: int | None = None, seed: int | None = None):
        self.stepsize, self.perturbation = stepsize, perturbation
        self._shots = shots
        self._rng = np.random.default_rng(seed)

    def _eval(self, circuit: Circuit, observable: Observable) -> float:
        if self._shots is None: return expectation(circuit, observable)
        return _shot_expectation(circuit, observable, self._shots, seed=int(self._rng.integers(2**31)))

    def step(self, params: dict[str, float], circuit: Circuit, observable: Observable) -> dict[str, float]:
        keys = sorted(params)
        delta = self._rng.choice([-1, 1], size=len(keys))
        p_plus = {k: params[k] + self.perturbation * d for k, d in zip(keys, delta)}
        p_minus = {k: params[k] - self.perturbation * d for k, d in zip(keys, delta)}
        e_plus = self._eval(circuit.bind(p_plus), observable)
        e_minus = self._eval(circuit.bind(p_minus), observable)
        g_est = (e_plus - e_minus) / (2 * self.perturbation)
        return {k: params[k] - self.stepsize * g_est / d for k, d in zip(keys, delta)}
