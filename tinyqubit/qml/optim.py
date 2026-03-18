"""Optimizers and gradient computation for variational circuits."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate, Parameter, _GATE_ADJOINT, _PARAM_GATES, _get_gate_matrix, _SQRT2_INV
from ..measurement.observable import Observable, expectation, _PAULI_MATRIX
from ..simulator import simulate, sample, _apply_single_qubit, _apply_three_qubit, _DIAG_PHASE


# Gradient computation -------


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


def _make_2q_idx(q0: int, q1: int, n: int):
    def idx(v0, v1):
        i = [slice(None)] * n; i[q0], i[q1] = v0, v1
        return tuple(i)
    return idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)


def _adjoint_backward(circuit: Circuit, bound: Circuit, state: np.ndarray, lam: np.ndarray) -> dict[str, float]:
    """Shared backward pass: given forward state and seed λ, extract all parameter gradients."""
    n = bound.n_qubits
    param_map = {}
    for i, op in enumerate(circuit.ops):
        if op.params and isinstance(op.params[0], Parameter):
            param_map[i] = op.params[0].name
    grad = {p.name: 0.0 for p in circuit.parameters}

    adjoint_info = []
    for op in bound.ops:
        gate = _GATE_ADJOINT.get(op.gate, op.gate)
        params = tuple(-p for p in op.params) if op.gate in _PARAM_GATES else op.params
        nq = gate.n_qubits
        if nq == 1:
            q = op.qubits[0]
            i0 = [slice(None)] * n; i0[q] = 0
            i1 = [slice(None)] * n; i1[q] = 1
            i0t, i1t = tuple(i0), tuple(i1)
            if gate == Gate.RZ:
                t = params[0]
                adjoint_info.append((gate, params, 1, (np.exp(-1j * t / 2), np.exp(1j * t / 2)), (i0t, i1t)))
            elif gate in _DIAG_PHASE:
                adjoint_info.append((gate, params, 1, _DIAG_PHASE[gate], (i0t, i1t)))
            else:
                adjoint_info.append((gate, params, 1, _get_gate_matrix(gate, params), (i0t, i1t)))
        elif nq == 2:
            adjoint_info.append((gate, params, 2, None, _make_2q_idx(op.qubits[0], op.qubits[1], n)))
        else:
            adjoint_info.append((gate, params, 3, None, op.qubits))

    state = state.reshape(-1)
    lam = lam.reshape(-1)
    buf_s = np.empty_like(state)
    buf_l = np.empty_like(lam)

    for k in range(len(bound.ops) - 1, -1, -1):
        op = bound.ops[k]
        agate, aparams, anq, mat_or_phase, idxs = adjoint_info[k]

        if k in param_map:
            name = param_map[k]
            st, la = state.reshape([2] * n), lam.reshape([2] * n)
            if op.gate == Gate.RY:
                i0, i1 = idxs
                grad[name] += (np.vdot(la[i1], st[i0]) - np.vdot(la[i0], st[i1])).real
            elif op.gate == Gate.RX:
                i0, i1 = idxs
                grad[name] += (np.vdot(la[i0], st[i1]) + np.vdot(la[i1], st[i0])).imag
            elif op.gate == Gate.RZ:
                i0, i1 = idxs
                grad[name] += (np.vdot(la[i0], st[i0]) - np.vdot(la[i1], st[i1])).imag
            elif op.gate == Gate.CP:
                _, _, _, _, (_, _, _, i11) = adjoint_info[k]
                grad[name] += -2 * np.vdot(la[i11], st[i11]).imag

        if anq == 1:
            if agate == Gate.RZ:
                i0, i1 = idxs
                e0, e1 = mat_or_phase
                st, la = state.reshape([2] * n), lam.reshape([2] * n)
                st[i0] *= e0; st[i1] *= e1
                la[i0] *= e0; la[i1] *= e1
            elif agate in _DIAG_PHASE:
                i0, i1 = idxs
                st, la = state.reshape([2] * n), lam.reshape([2] * n)
                st[i1] *= mat_or_phase; la[i1] *= mat_or_phase
            else:
                mat = mat_or_phase
                qubit = op.qubits[0]
                nq, nr = 1 << qubit, 1 << (n - qubit - 1)
                if min(nq, nr) > 1:
                    np.matmul(mat, state.reshape(nq, 2, nr), out=buf_s.reshape(nq, 2, nr))
                    np.matmul(mat, lam.reshape(nq, 2, nr), out=buf_l.reshape(nq, 2, nr))
                else:
                    i0, i1 = idxs
                    st, bs = state.reshape([2] * n), buf_s.reshape([2] * n)
                    ss0, ss1 = st[i0], st[i1]
                    bs[i0] = mat[0, 0] * ss0 + mat[0, 1] * ss1
                    bs[i1] = mat[1, 0] * ss0 + mat[1, 1] * ss1
                    la, bl = lam.reshape([2] * n), buf_l.reshape([2] * n)
                    ls0, ls1 = la[i0], la[i1]
                    bl[i0] = mat[0, 0] * ls0 + mat[0, 1] * ls1
                    bl[i1] = mat[1, 0] * ls0 + mat[1, 1] * ls1
                state, buf_s = buf_s, state
                lam, buf_l = buf_l, lam
        elif anq == 2:
            st, la = state.reshape([2] * n), lam.reshape([2] * n)
            i00, i01, i10, i11 = idxs
            if agate == Gate.CX:
                tmp = st[i10].copy(); st[i10] = st[i11]; st[i11] = tmp
                tmp = la[i10].copy(); la[i10] = la[i11]; la[i11] = tmp
            elif agate == Gate.CZ:
                st[i11] *= -1; la[i11] *= -1
            elif agate == Gate.SWAP:
                tmp = st[i01].copy(); st[i01] = st[i10]; st[i10] = tmp
                tmp = la[i01].copy(); la[i01] = la[i10]; la[i10] = tmp
            elif agate == Gate.CP:
                phase = np.exp(1j * aparams[0])
                st[i11] *= phase; la[i11] *= phase
            elif agate == Gate.RZZ:
                t = aparams[0]
                em, ep = np.exp(-1j * t / 2), np.exp(1j * t / 2)
                st[i00] *= em; st[i01] *= ep; st[i10] *= ep; st[i11] *= em
                la[i00] *= em; la[i01] *= ep; la[i10] *= ep; la[i11] *= em
            elif agate == Gate.ECR:
                s00, s01, s10, s11 = st[i00].copy(), st[i01].copy(), st[i10].copy(), st[i11].copy()
                st[i00] = _SQRT2_INV * (s10 + 1j * s11); st[i01] = _SQRT2_INV * (1j * s10 + s11)
                st[i10] = _SQRT2_INV * (s00 - 1j * s01); st[i11] = _SQRT2_INV * (-1j * s00 + s01)
                l00, l01, l10, l11 = la[i00].copy(), la[i01].copy(), la[i10].copy(), la[i11].copy()
                la[i00] = _SQRT2_INV * (l10 + 1j * l11); la[i01] = _SQRT2_INV * (1j * l10 + l11)
                la[i10] = _SQRT2_INV * (l00 - 1j * l01); la[i11] = _SQRT2_INV * (-1j * l00 + l01)
        else:
            state = _apply_three_qubit(state, agate, *op.qubits, n)
            lam = _apply_three_qubit(lam, agate, *op.qubits, n)
            buf_s = np.empty_like(state); buf_l = np.empty_like(lam)
    return grad


def adjoint_gradient(circuit: Circuit, observable: Observable, params: dict[str, float] | None = None) -> dict[str, float]:
    """Compute all gradients in one forward + backward pass (adjoint differentiation)."""
    if params is None: params = circuit.param_values
    bound = circuit.bind(params)
    state, _ = simulate(bound)
    n = bound.n_qubits
    lam_t = np.zeros(([2] * n), dtype=state.dtype)
    state_t = state.reshape([2] * n)
    for coeff, paulis in observable.terms:
        types = set(paulis.values())
        qubits = tuple(paulis.keys())
        if types <= {'Z'}:
            t = state_t.copy()
            for q in qubits:
                idx1 = [slice(None)] * n; idx1[q] = 1
                t[tuple(idx1)] *= -1
            lam_t += coeff * t
        elif types <= {'X'}:
            lam_t += coeff * np.flip(state_t, axis=qubits)
        elif types <= {'Y'}:
            flipped = np.flip(state_t, axis=qubits)
            t = flipped.copy()
            # Y|0⟩=i|1⟩, Y|1⟩=-i|0⟩ → after flip, position 0 came from |1⟩ (coeff -i), position 1 from |0⟩ (coeff i)
            for q in qubits:
                i0 = [slice(None)] * n; i0[q] = 0
                i1 = [slice(None)] * n; i1[q] = 1
                t[tuple(i0)] *= -1j; t[tuple(i1)] *= 1j
            lam_t += coeff * t
        else:
            tmp = state.copy()
            for qubit, pauli in paulis.items():
                tmp = _apply_single_qubit(tmp, _PAULI_MATRIX[pauli], qubit, n)
            lam_t += tmp.reshape([2] * n) * coeff
    lam = lam_t.reshape(-1)
    return _adjoint_backward(circuit, bound, state, lam)


def backprop_gradient(circuit: Circuit, loss_fn, params: dict[str, float] | None = None, eps: float = 1e-7) -> dict[str, float]:
    """Backprop gradient for loss(probabilities). One forward + one backward pass."""
    if params is None: params = circuit.param_values
    bound = circuit.bind(params)
    state, _ = simulate(bound)
    probs = np.abs(state) ** 2
    # dloss/dp via finite differences on the small 2^n classical vector
    loss0 = loss_fn(probs)
    dloss_dp = np.empty_like(probs)
    for i in range(len(probs)):
        probs[i] += eps
        dloss_dp[i] = (loss_fn(probs) - loss0) / eps
        probs[i] -= eps
    # Seed: λ_i = (dloss/dp_i) · ψ_i  (chain rule through |ψ_i|²)
    return _adjoint_backward(circuit, bound, state, dloss_dp * state)


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

def _compute_grad(circuit, objective, params, default_grad_fn):
    """Dispatch gradient: Observable → adjoint, callable → backprop."""
    if callable(objective) and not isinstance(objective, Observable):
        return backprop_gradient(circuit, objective, params)
    return default_grad_fn(circuit, objective, params)


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
        if grad is None: grad = _compute_grad(circuit, observable, params, self._grad_fn)
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
        if grad is None: grad = _compute_grad(circuit, observable, params, self._grad_fn)
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
