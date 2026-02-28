"""Optimizers for variational circuits: GradientDescent, Adam, SPSA, QNG."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit
from ..observable import Observable, expectation
from ..gradient import adjoint_gradient, quantum_fisher_information, _shot_expectation


class GradientDescent:
    """Vanilla gradient descent."""
    def __init__(self, stepsize: float = 0.1, grad_fn=None):
        self.stepsize = stepsize
        self._grad_fn = grad_fn or adjoint_gradient

    def _update(self, params: dict[str, float], grad: dict[str, float]) -> dict[str, float]:
        return {k: params[k] - self.stepsize * grad[k] for k in params}

    def step(self, params: dict[str, float], circuit: Circuit = None, observable: Observable = None, grad: dict[str, float] = None) -> dict[str, float]:
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        return self._update(params, grad)


class Adam:
    """Adam optimizer with bias-corrected moments."""
    def __init__(self, stepsize: float = 0.01, grad_fn=None,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.stepsize, self.beta1, self.beta2, self.eps = stepsize, beta1, beta2, eps
        self._grad_fn = grad_fn or adjoint_gradient
        self._m: dict[str, float] = {}
        self._v: dict[str, float] = {}
        self._t = 0

    def _update(self, params: dict[str, float], grad: dict[str, float]) -> dict[str, float]:
        self._t += 1
        result = {}
        for k in params:
            self._m[k] = self.beta1 * self._m.get(k, 0) + (1 - self.beta1) * grad[k]
            self._v[k] = self.beta2 * self._v.get(k, 0) + (1 - self.beta2) * grad[k] ** 2
            m_hat = self._m[k] / (1 - self.beta1 ** self._t)
            v_hat = self._v[k] / (1 - self.beta2 ** self._t)
            result[k] = params[k] - self.stepsize * m_hat / (np.sqrt(v_hat) + self.eps)
        return result

    def step(self, params: dict[str, float], circuit: Circuit = None, observable: Observable = None, grad: dict[str, float] = None) -> dict[str, float]:
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        return self._update(params, grad)


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

    def _perturb_and_step(self, params: dict[str, float], eval_fn) -> dict[str, float]:
        keys = sorted(params)
        delta = self._rng.choice([-1, 1], size=len(keys))
        p_plus = {k: params[k] + self.perturbation * d for k, d in zip(keys, delta)}
        p_minus = {k: params[k] - self.perturbation * d for k, d in zip(keys, delta)}
        g_est = (eval_fn(p_plus) - eval_fn(p_minus)) / (2 * self.perturbation)
        return {k: params[k] - self.stepsize * g_est / d for k, d in zip(keys, delta)}

    def step(self, params: dict[str, float], circuit: Circuit, observable: Observable) -> dict[str, float]:
        return self._perturb_and_step(params, lambda p: self._eval(circuit.bind(p), observable))


class QNG:
    """Quantum Natural Gradient — preconditions gradient with inverse QFI matrix."""
    def __init__(self, stepsize: float = 0.01, epsilon: float = 1e-3, grad_fn=None):
        self.stepsize, self.epsilon = stepsize, epsilon
        self._grad_fn = grad_fn or adjoint_gradient

    def step(self, params: dict[str, float], circuit: Circuit, observable: Observable = None, grad: dict[str, float] = None) -> dict[str, float]:
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        F = quantum_fisher_information(circuit, params)
        F_reg = F + self.epsilon * np.eye(len(F))
        keys = sorted(params)
        g = np.array([grad[k] for k in keys])
        nat_grad = np.linalg.solve(F_reg, g)
        return {k: params[k] - self.stepsize * ng for k, ng in zip(keys, nat_grad)}
