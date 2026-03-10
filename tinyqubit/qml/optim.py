"""Optimizers and gradient computation for variational circuits."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate, Parameter, _GATE_ADJOINT, _PARAM_GATES, _get_gate_matrix
from ..analysis.observable import Observable, expectation, _PAULI_MATRIX
from ..simulator import simulate, sample, _apply_single_qubit, _apply_two_qubit, _apply_three_qubit


# Gradient computation -------

_GENERATORS = {Gate.RX: 'X', Gate.RY: 'Y', Gate.RZ: 'Z'}


def _shot_expectation(circuit: Circuit, observable: Observable, shots: int, seed: int | None = None) -> float:
    """Estimate ⟨ψ|O|ψ⟩ by measurement sampling with basis rotations."""
    rng = np.random.default_rng(seed)
    result = 0.0
    for coeff, paulis in observable.terms:
        rc = Circuit(circuit.n_qubits, circuit.n_classical)
        rc._initial_state, rc.ops = circuit._initial_state, list(circuit.ops)
        for q, p in paulis.items():
            if p == 'X': rc.h(q)
            elif p == 'Y':
                rc.rz(q, -np.pi / 2)
                rc.h(q)
        state, _ = simulate(rc)
        counts = sample(state, shots, seed=int(rng.integers(2**31)))
        ev = sum((-1) ** (sum(int(bs[q]) for q in paulis) % 2) * cnt for bs, cnt in counts.items())
        result += coeff * ev / shots
    return result


def parameter_shift_gradient(circuit: Circuit, observable: Observable,
                             params: dict[str, float] | None = None, shots: int | None = None) -> dict[str, float]:
    """Compute gradient via the parameter-shift rule: df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2"""
    if params is None: params = circuit.param_values
    shift = np.pi / 2
    exp_fn = expectation if shots is None else lambda c, o: _shot_expectation(c, o, shots)
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**params, param.name: params[param.name] + shift}
        v_minus = {**params, param.name: params[param.name] - shift}
        e_plus = exp_fn(circuit.bind(v_plus), observable)
        e_minus = exp_fn(circuit.bind(v_minus), observable)
        grad[param.name] = (e_plus - e_minus) / 2
    return grad


def finite_difference_gradient(circuit: Circuit, observable: Observable,
                               params: dict[str, float] | None = None, epsilon: float = 1e-7) -> dict[str, float]:
    """Compute gradient via symmetric finite differences: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / 2ε"""
    if params is None: params = circuit.param_values
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**params, param.name: params[param.name] + epsilon}
        v_minus = {**params, param.name: params[param.name] - epsilon}
        e_plus = expectation(circuit.bind(v_plus), observable)
        e_minus = expectation(circuit.bind(v_minus), observable)
        grad[param.name] = (e_plus - e_minus) / (2 * epsilon)
    return grad


def _unapply_op(state, op, n):
    gate = _GATE_ADJOINT.get(op.gate, op.gate)
    params = tuple(-p for p in op.params) if op.gate in _PARAM_GATES else op.params
    if gate.n_qubits == 1:
        return _apply_single_qubit(state, _get_gate_matrix(gate, params), op.qubits[0], n)
    if gate.n_qubits == 2:
        return _apply_two_qubit(state, gate, op.qubits[0], op.qubits[1], n, params)
    return _apply_three_qubit(state, gate, *op.qubits, n)


def adjoint_gradient(circuit: Circuit, observable: Observable, params: dict[str, float] | None = None) -> dict[str, float]:
    """Compute all gradients in one forward + backward pass (adjoint differentiation)."""
    if params is None: params = circuit.param_values
    bound = circuit.bind(params)
    n = bound.n_qubits
    state, _ = simulate(bound)
    # |λ⟩ = O|ψ⟩
    lam = np.zeros_like(state)
    for coeff, paulis in observable.terms:
        tmp = state.copy()
        for qubit, pauli in paulis.items():
            tmp = _apply_single_qubit(tmp, _PAULI_MATRIX[pauli], qubit, n)
        lam += coeff * tmp
    # Map op indices to parameter names (from unbound circuit)
    param_map = {}
    for i, op in enumerate(circuit.ops):
        if op.params and isinstance(op.params[0], Parameter):
            param_map[i] = op.params[0].name
    grad = {p.name: 0.0 for p in circuit.parameters}
    # Backward pass: at step k, state = |ψ_k⟩, lam = |λ_k⟩
    for k in range(len(bound.ops) - 1, -1, -1):
        op = bound.ops[k]
        if k in param_map:
            name = param_map[k]
            if op.gate in _GENERATORS:
                g_psi = _apply_single_qubit(state, _PAULI_MATRIX[_GENERATORS[op.gate]], op.qubits[0], n)
                grad[name] += np.vdot(lam, g_psi).imag
            elif op.gate == Gate.CP:
                q0, q1 = op.qubits
                idx = [slice(None)] * n
                idx[q0], idx[q1] = 1, 1
                idx = tuple(idx)
                grad[name] += -2 * np.vdot(lam.reshape([2] * n)[idx], state.reshape([2] * n)[idx]).imag
        state = _unapply_op(state, op, n)
        lam = _unapply_op(lam, op, n)
    return grad


def quantum_fisher_information(circuit: Circuit, params: dict[str, float] | None = None) -> np.ndarray:
    """Compute the Quantum Fisher Information matrix via parameter-shift statevectors."""
    if params is None: params = circuit.param_values
    psi, _ = simulate(circuit.bind(params))
    sorted_params = sorted(circuit.parameters, key=lambda p: p.name)
    shift = np.pi / 2
    dpsi = []
    for p in sorted_params:
        sp, _ = simulate(circuit.bind({**params, p.name: params[p.name] + shift}))
        sm, _ = simulate(circuit.bind({**params, p.name: params[p.name] - shift}))
        dpsi.append((sp - sm) / 2)
    n = len(sorted_params)
    F = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            F[i, j] = F[j, i] = 2 * np.real(np.vdot(dpsi[i], dpsi[j]) - np.vdot(dpsi[i], psi) * np.vdot(psi, dpsi[j]))
    return F


def gradient_landscape(circuit: Circuit, param_names: list[str], observable: Observable,
                       base_values: dict[str, float] | None = None, n_points: int = 50,
                       ranges: list[tuple[float, float]] | None = None) -> np.ndarray:
    """2D expectation sweep over two parameters. Returns (n_points, n_points) array."""
    if base_values is None: base_values = circuit.param_values
    ranges = ranges or [(0, 2 * np.pi)] * 2
    ax0 = np.linspace(*ranges[0], n_points)
    ax1 = np.linspace(*ranges[1], n_points)
    work = circuit.bind({})
    result = np.empty((n_points, n_points))
    for i, v0 in enumerate(ax0):
        for j, v1 in enumerate(ax1):
            work.bind_params({**base_values, param_names[0]: v0, param_names[1]: v1})
            result[i, j] = expectation(work, observable)
    return result


def cost_gradient(circuit: Circuit, cost_fn, params=None, *args, eps: float = 0.01) -> dict[str, float]:
    """Finite-difference gradient of cost_fn(circuit.bind(params), *args)."""
    if params is not None and not isinstance(params, dict):
        args = (params,) + args
        params = None
    if params is None: params = circuit.param_values
    grad = {}
    for k in params:
        plus, minus = dict(params), dict(params)
        plus[k] += eps
        minus[k] -= eps
        grad[k] = (cost_fn(circuit.bind(plus), *args) - cost_fn(circuit.bind(minus), *args)) / (2 * eps)
    return grad


# Optimizers -------

class GradientDescent:
    """Vanilla gradient descent."""
    def __init__(self, stepsize: float = 0.1, grad_fn=None):
        self.stepsize = stepsize
        self._grad_fn = grad_fn or adjoint_gradient

    def _update(self, params: dict[str, float], grad: dict[str, float]) -> dict[str, float]:
        return {k: params[k] - self.stepsize * grad[k] for k in params}

    def step(self, params_or_circuit, circuit=None, observable=None, grad=None):
        if isinstance(params_or_circuit, Circuit):
            circuit, observable = params_or_circuit, circuit
            params = circuit.param_values
        else:
            params = params_or_circuit
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        result = self._update(params, grad)
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result


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

    def step(self, params_or_circuit, circuit=None, observable=None, grad=None):
        if isinstance(params_or_circuit, Circuit):
            circuit, observable = params_or_circuit, circuit
            params = circuit.param_values
        else:
            params = params_or_circuit
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        result = self._update(params, grad)
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
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

    def _perturb_and_step(self, params: dict[str, float], eval_fn) -> dict[str, float]:
        keys = sorted(params)
        delta = self._rng.choice([-1, 1], size=len(keys))
        p_plus = {k: params[k] + self.perturbation * d for k, d in zip(keys, delta)}
        p_minus = {k: params[k] - self.perturbation * d for k, d in zip(keys, delta)}
        g_est = (eval_fn(p_plus) - eval_fn(p_minus)) / (2 * self.perturbation)
        return {k: params[k] - self.stepsize * g_est / d for k, d in zip(keys, delta)}

    def step(self, params_or_circuit, circuit=None, observable=None):
        if isinstance(params_or_circuit, Circuit):
            circuit, observable = params_or_circuit, circuit
            params = circuit.param_values
        else:
            params = params_or_circuit
        result = self._perturb_and_step(params, lambda p: self._eval(circuit.bind(p), observable))
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result


class QNG:
    """Quantum Natural Gradient — preconditions gradient with inverse QFI matrix."""
    def __init__(self, stepsize: float = 0.01, epsilon: float = 1e-3, grad_fn=None):
        self.stepsize, self.epsilon = stepsize, epsilon
        self._grad_fn = grad_fn or adjoint_gradient

    def step(self, params_or_circuit, circuit=None, observable=None, grad=None):
        if isinstance(params_or_circuit, Circuit):
            circuit, observable = params_or_circuit, circuit
            params = circuit.param_values
        else:
            params = params_or_circuit
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        F = quantum_fisher_information(circuit, params)
        F_reg = F + self.epsilon * np.eye(len(F))
        keys = sorted(params)
        g = np.array([grad[k] for k in keys])
        nat_grad = np.linalg.solve(F_reg, g)
        result = {k: params[k] - self.stepsize * ng for k, ng in zip(keys, nat_grad)}
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result
