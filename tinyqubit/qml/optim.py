"""Optimizers and gradient computation for variational circuits."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate, Parameter, ScaledParam, _is_param, _GATE_ADJOINT, _PARAM_GATES, _get_gate_matrix, _SQRT2_INV
from ..measurement.observable import Observable, expectation, _PAULI_MATRIX
from ..simulator import simulate, sample, _apply_single_qubit, _apply_two_qubit, _apply_three_qubit, _DIAG_PHASE, _get_perm


# Gradient computation -------


_H_MAT = _SQRT2_INV * np.array([[1,1],[1,-1]], dtype=complex)
_Y2Z_MAT = _SQRT2_INV * np.array([[1,-1j],[1,1j]], dtype=complex)
_BACKWARD_PERM_GATES = frozenset({Gate.CX, Gate.SWAP})


def _shot_expectation(circuit: Circuit, observable: Observable, shots: int, seed: int | None = None) -> float:
    """Estimate ⟨ψ|O|ψ⟩ by measurement sampling with basis rotations."""
    rng = np.random.default_rng(seed)
    state, _ = simulate(circuit)
    n = circuit.n_qubits
    result = 0.0
    for coeff, paulis in observable.terms:
        if not paulis:
            result += coeff; continue
        rotated = state
        for q, p in paulis.items():
            if p == 'X': rotated = _apply_single_qubit(rotated, _H_MAT, q, n)
            elif p == 'Y': rotated = _apply_single_qubit(rotated, _Y2Z_MAT, q, n)
        counts = sample(rotated, shots, seed=int(rng.integers(2**31)))
        ev = sum((-1) ** (sum(int(bs[q]) for q in paulis) % 2) * cnt for bs, cnt in counts.items())
        result += coeff * ev / shots
    return result


def _has_scaled_params(circuit: Circuit) -> bool:
    return any(isinstance(p, ScaledParam) for op in circuit.ops for p in op.params)


def _shift_gradient(circuit: Circuit, observable: Observable, params: dict[str, float],
                    shift: float, divisor: float, exp_fn) -> dict[str, float]:
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**params, param.name: params[param.name] + shift}
        v_minus = {**params, param.name: params[param.name] - shift}
        grad[param.name] = (exp_fn(circuit.bind(v_plus), observable) - exp_fn(circuit.bind(v_minus), observable)) / divisor
    return grad


def parameter_shift_gradient(circuit: Circuit, observable: Observable,
                             params: dict[str, float] | None = None, shots: int | None = None) -> dict[str, float]:
    """Compute gradient via parameter-shift rule (single-gate params) or finite-difference (multi-gate/scaled)."""
    if params is None: params = circuit.param_values
    exp_fn = expectation if shots is None else lambda c, o: _shot_expectation(c, o, shots)
    # NOTE: parameter-shift only works for params appearing in a single rotation gate.
    # Trotterized UCCSD uses ScaledParam where each param feeds multiple RZ gates —
    # fall back to finite-difference for correctness.
    if _has_scaled_params(circuit):
        # Central finite-difference: ε=1e-7 balances truncation vs floating-point error
        return _shift_gradient(circuit, observable, params, 1e-7, 2e-7, exp_fn)
    return _shift_gradient(circuit, observable, params, np.pi / 2, 2, exp_fn)


def finite_difference_gradient(circuit: Circuit, observable: Observable,
                               params: dict[str, float] | None = None, epsilon: float = 1e-7) -> dict[str, float]:
    """Compute gradient via symmetric finite differences: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / 2ε"""
    if params is None: params = circuit.param_values
    return _shift_gradient(circuit, observable, params, epsilon, 2 * epsilon, expectation)


def _make_2q_idx(q0: int, q1: int, n: int):
    def idx(v0, v1):
        i = [slice(None)] * n; i[q0], i[q1] = v0, v1
        return tuple(i)
    return idx(0, 0), idx(0, 1), idx(1, 0), idx(1, 1)


def _build_adjoint_info(circuit: Circuit, bound: Circuit):
    """Build adjoint_info list (cacheable structure) and list of parametric RZ indices."""
    n = bound.n_qubits
    param_map = {}
    for i, op in enumerate(circuit.ops):
        p0 = op.params[0] if op.params else None
        if isinstance(p0, ScaledParam): param_map[i] = (p0.param.name, p0.scale)
        elif isinstance(p0, Parameter): param_map[i] = (p0.name, 1.0)

    adjoint_info = []
    param_indices = []  # indices of parametric gates that need updating per step
    for k, op in enumerate(bound.ops):
        gate = _GATE_ADJOINT.get(op.gate, op.gate)
        params = tuple(-p for p in op.params) if op.gate in _PARAM_GATES else op.params
        nq = gate.n_qubits
        is_parametric = k in param_map
        if nq == 1:
            q = op.qubits[0]
            i0 = [slice(None)] * n; i0[q] = 0
            i1 = [slice(None)] * n; i1[q] = 1
            i0t, i1t = tuple(i0), tuple(i1)
            if gate == Gate.RZ:
                t = params[0]
                adjoint_info.append([gate, params, 1, (np.exp(-1j * t / 2), np.exp(1j * t / 2)), (i0t, i1t)])
            elif gate in _DIAG_PHASE:
                adjoint_info.append([gate, params, 1, _DIAG_PHASE[gate], (i0t, i1t)])
            else:
                adjoint_info.append([gate, params, 1, _get_gate_matrix(gate, params), (i0t, i1t)])
        elif nq == 2:
            adjoint_info.append([gate, params, 2, None, _make_2q_idx(op.qubits[0], op.qubits[1], n)])
        elif nq == 3:
            adjoint_info.append([gate, params, 3, None, op.qubits])
        else:
            adjoint_info.append([gate, params, 4, None, op.qubits])
        if is_parametric:
            param_indices.append(k)
    return adjoint_info, param_indices, param_map


def _adjoint_backward(circuit: Circuit, bound: Circuit, state: np.ndarray, lam: np.ndarray) -> dict[str, float]:
    # NOTE: Diagonal 1Q gates are accumulated into a combined phase vector and flushed in one
    # pass. Safe because phases are unit-magnitude: vdot(e·λ, e·ψ) = vdot(λ, ψ).
    n = bound.n_qubits

    # Cache adjoint_info structure on the circuit — update all parametric gate values each call
    ops_id = id(circuit.ops), len(circuit.ops)
    cache = getattr(circuit, '_adj_cache', None)
    if cache is not None and cache[0] == ops_id:
        _, adjoint_info, param_indices, param_map = cache
        for k in param_indices:
            op = bound.ops[k]
            gate = _GATE_ADJOINT.get(op.gate, op.gate)
            params = tuple(-p for p in op.params) if op.gate in _PARAM_GATES else op.params
            adjoint_info[k][1] = params
            if adjoint_info[k][2] == 1:  # 1Q gate
                if gate == Gate.RZ:
                    adjoint_info[k][3] = (np.exp(-1j * params[0] / 2), np.exp(1j * params[0] / 2))
                else:
                    adjoint_info[k][3] = _get_gate_matrix(gate, params)
    else:
        adjoint_info, param_indices, param_map = _build_adjoint_info(circuit, bound)
        circuit._adj_cache = (ops_id, adjoint_info, param_indices, param_map)

    grad = {p.name: 0.0 for p in circuit.parameters}

    dim = len(state.reshape(-1))
    # NOTE: Pack state+lam into contiguous (2, dim) pair for fused 1Q matmul
    sl = np.empty((2, dim), dtype=state.dtype)
    sl[0] = state.reshape(-1); sl[1] = lam.reshape(-1)
    buf_sl = np.empty_like(sl)
    state, lam = sl[0], sl[1]  # views into sl
    buf_s, buf_l = buf_sl[0], buf_sl[1]
    diag_phase = np.ones(dim, dtype=state.dtype)
    diag_dirty = False

    def _flush_diag():
        nonlocal sl, buf_sl, state, lam, buf_s, buf_l, diag_dirty
        if not diag_dirty: return
        sl[0] *= diag_phase
        sl[1] *= diag_phase
        diag_phase[:] = 1.0
        diag_dirty = False

    # Precompute CX/SWAP block boundaries for batch permutation in backward traversal
    perm_block_start = {}  # block_end_k -> (block_start_k, perm_inv)
    if n >= 10:
        j = len(bound.ops) - 1
        while j >= 0:
            if bound.ops[j].gate in _BACKWARD_PERM_GATES and j not in param_map:
                end = j
                while j > 0 and bound.ops[j - 1].gate in _BACKWARD_PERM_GATES and (j - 1) not in param_map:
                    j -= 1
                start = j
                if end > start:
                    ops_rev = tuple(
                        ('CX' if bound.ops[i].gate == Gate.CX else 'SWAP', bound.ops[i].qubits[0], bound.ops[i].qubits[1])
                        for i in range(end, start - 1, -1))
                    perm_block_start[end] = (start, _get_perm(ops_rev, n))
            j -= 1

    k = len(bound.ops) - 1
    while k >= 0:
        op = bound.ops[k]
        agate, aparams, anq, mat_or_phase, idxs = adjoint_info[k]
        is_diag_1q = anq == 1 and (agate == Gate.RZ or agate in _DIAG_PHASE)

        if is_diag_1q:
            # Gradient extraction is invariant under pending diagonal phases
            if k in param_map:
                name, scale = param_map[k]
                st, la = state.reshape([2] * n), lam.reshape([2] * n)
                i0, i1 = idxs
                grad[name] += scale * (np.vdot(la[i0], st[i0]) - np.vdot(la[i1], st[i1])).imag

            dp = diag_phase.reshape([2] * n)
            if agate == Gate.RZ:
                e0, e1 = mat_or_phase
                dp[idxs[0]] *= e0; dp[idxs[1]] *= e1
            else:
                dp[idxs[1]] *= mat_or_phase
            diag_dirty = True
        elif k in perm_block_start:
            _flush_diag()
            start, perm_inv = perm_block_start[k]
            np.take(sl[0], perm_inv, out=buf_sl[0])
            np.take(sl[1], perm_inv, out=buf_sl[1])
            sl, buf_sl = buf_sl, sl
            state, lam = sl[0], sl[1]
            buf_s, buf_l = buf_sl[0], buf_sl[1]
            k = start - 1; continue
        else:
            _flush_diag()

            if k in param_map:
                name, scale = param_map[k]
                st, la = state.reshape([2] * n), lam.reshape([2] * n)
                if op.gate == Gate.RY:
                    i0, i1 = idxs
                    grad[name] += scale * (np.vdot(la[i1], st[i0]) - np.vdot(la[i0], st[i1])).real
                elif op.gate == Gate.RX:
                    i0, i1 = idxs
                    grad[name] += scale * (np.vdot(la[i0], st[i1]) + np.vdot(la[i1], st[i0])).imag
                elif op.gate == Gate.CP:
                    _, _, _, _, (_, _, _, i11) = adjoint_info[k]
                    grad[name] += scale * -2 * np.vdot(la[i11], st[i11]).imag
                elif op.gate == Gate.RZZ:
                    i00, i01, i10, i11 = idxs
                    grad[name] += scale * (np.vdot(la[i00], st[i00]) - np.vdot(la[i01], st[i01])
                                   - np.vdot(la[i10], st[i10]) + np.vdot(la[i11], st[i11])).imag
                elif op.gate == Gate.SEXC:
                    _, _, _, _, (_, i01, i10, _) = adjoint_info[k]
                    grad[name] += scale * (np.vdot(la[i10], st[i01]) - np.vdot(la[i01], st[i10])).real
                elif op.gate == Gate.DEXC:
                    q0, q1, q2, q3 = op.qubits
                    def idx4(v0, v1, v2, v3):
                        i = [slice(None)] * n; i[q0], i[q1], i[q2], i[q3] = v0, v1, v2, v3
                        return tuple(i)
                    i0011, i1100 = idx4(0,0,1,1), idx4(1,1,0,0)
                    grad[name] += scale * (np.vdot(la[i1100], st[i0011]) - np.vdot(la[i0011], st[i1100])).real

            if anq == 1:
                mat = mat_or_phase
                qubit = op.qubits[0]
                nql, nr = 1 << qubit, 1 << (n - qubit - 1)
                if nr <= 1:
                    # NOTE: 2D GEMM much faster than 3D batch for nr=1
                    np.matmul(sl.reshape(2 * nql, 2), mat.T, out=buf_sl.reshape(2 * nql, 2))
                elif nql == 1:
                    np.matmul(mat, sl.reshape(2, 2, nr), out=buf_sl.reshape(2, 2, nr))
                else:
                    np.matmul(mat, sl.reshape(2, nql, 2, nr), out=buf_sl.reshape(2, nql, 2, nr))
                sl, buf_sl = buf_sl, sl
                state, lam = sl[0], sl[1]
                buf_s, buf_l = buf_sl[0], buf_sl[1]
            elif anq == 2:
                q0, q1 = op.qubits[0], op.qubits[1]
                _apply_two_qubit(state, agate, q0, q1, n, aparams)
                _apply_two_qubit(lam, agate, q0, q1, n, aparams)
            elif anq == 3:
                # 3Q/4Q gates may reallocate; copy back into sl
                s2 = _apply_three_qubit(state, agate, *op.qubits, n)
                l2 = _apply_three_qubit(lam, agate, *op.qubits, n)
                sl[0] = s2; sl[1] = l2
                state, lam = sl[0], sl[1]
            else:  # 4Q (DEXC)
                from ..simulator.statevector import _apply_four_qubit
                s2 = _apply_four_qubit(state, agate, *op.qubits, n, aparams)
                l2 = _apply_four_qubit(lam, agate, *op.qubits, n, aparams)
                sl[0] = s2; sl[1] = l2
                state, lam = sl[0], sl[1]

        k -= 1
    _flush_diag()
    return grad


def adjoint_gradient(circuit: Circuit, observable: Observable, params: dict[str, float] | None = None, return_cost: bool = False):
    """Compute all gradients in one forward + backward pass (adjoint differentiation)."""
    if params is None: params = circuit.param_values
    # Reuse cached work circuit to avoid per-call bind() allocation
    work = getattr(circuit, '_adj_work', None)
    if work is None or work._structure_key() != circuit._structure_key():
        work = circuit.bind(params)
        circuit._adj_work = work
    else:
        work.bind_params(params)
    state, _ = simulate(work)
    n = work.n_qubits
    bound = work
    if observable._matrix is not None:
        lam = observable._matrix @ state
    else:
        dim = 1 << n
        lam_t = np.zeros(([2] * n), dtype=state.dtype)
        state_t = state.reshape([2] * n)
        # Batch Z-only terms via popcount parity (avoids per-term state copy)
        z_terms = [(c, p) for c, p in observable.terms if p and set(p.values()) <= {'Z'}]
        if z_terms:
            idx = np.arange(dim, dtype=np.int32)
            weights = np.zeros(dim, dtype=np.float64)
            for coeff, paulis in z_terms:
                mask = sum(1 << (n - 1 - q) for q in paulis)
                v = idx & mask
                v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
                weights += coeff * (1 - 2 * (v & 1))
            lam_t += (state.reshape(-1) * weights).reshape([2] * n)
        for coeff, paulis in observable.terms:
            if not paulis: lam_t += coeff * state_t; continue
            types = set(paulis.values())
            if types <= {'Z'}: continue  # already handled above
            qubits = tuple(paulis.keys())
            if types <= {'X'}:
                lam_t += coeff * np.flip(state_t, axis=qubits)
            elif types <= {'Y'}:
                flipped = np.flip(state_t, axis=qubits)
                t = flipped.copy()
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
    grad = _adjoint_backward(circuit, bound, state, lam)
    if return_cost:
        return grad, np.vdot(state, lam).real
    return grad


def backprop_gradient(circuit: Circuit, loss_fn, params: dict[str, float] | None = None, eps: float = 1e-7, return_cost: bool = False):
    """Backprop gradient for loss(probabilities). One forward + one backward pass."""
    if params is None: params = circuit.param_values
    bound = circuit.bind(params)
    state, _ = simulate(bound)
    probs = np.abs(state) ** 2
    cost = float(loss_fn(probs)) if return_cost else None
    if hasattr(loss_fn, 'grad'):
        dloss_dp = loss_fn.grad(probs)
    else:
        loss0 = cost if return_cost else loss_fn(probs)
        dloss_dp = np.empty_like(probs)
        for i in range(len(probs)):
            orig = probs[i]
            probs[i] = orig + eps
            dloss_dp[i] = (loss_fn(probs) - loss0) / eps
            probs[i] = orig
    # Seed: λ_i = (dloss/dp_i) · ψ_i  (chain rule through |ψ_i|²)
    grad = _adjoint_backward(circuit, bound, state, dloss_dp * state)
    return (grad, cost) if return_cost else grad


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

def _unpack_args(params_or_circuit, circuit, observable):
    """Unpack the flexible (circuit, observable) or (params, circuit, observable) calling convention."""
    if isinstance(params_or_circuit, Circuit):
        return params_or_circuit.param_values, params_or_circuit, circuit
    return params_or_circuit, circuit, observable


def _compute_grad(circuit, objective, params, default_grad_fn):
    """Dispatch gradient: Observable → adjoint, callable → backprop.
    Falls back to parameter_shift for >28Q circuits (MPS) since adjoint/backprop need statevector."""
    if callable(objective) and not isinstance(objective, Observable):
        if circuit.n_qubits > 28:
            raise ValueError("backprop_gradient not supported for >28Q circuits. Use parameter_shift_gradient.")
        return backprop_gradient(circuit, objective, params)
    if default_grad_fn is adjoint_gradient and circuit.n_qubits > 28:
        return parameter_shift_gradient(circuit, objective, params)
    return default_grad_fn(circuit, objective, params)


class _GradOptimizer:
    """Base for gradient-based optimizers. Subclasses implement _update()."""
    def step(self, params_or_circuit, circuit=None, observable=None, grad=None):
        params, circuit, observable = _unpack_args(params_or_circuit, circuit, observable)
        if grad is None: grad = _compute_grad(circuit, observable, params, self._grad_fn)
        result = self._update(params, grad)
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result

    def step_and_cost(self, params_or_circuit, circuit=None, observable=None):
        params, circuit, observable = _unpack_args(params_or_circuit, circuit, observable)
        if callable(observable) and not isinstance(observable, Observable):
            grad, cost = backprop_gradient(circuit, observable, params, return_cost=True)
        elif self._grad_fn is adjoint_gradient and circuit.n_qubits <= 28:
            grad, cost = adjoint_gradient(circuit, observable, params, return_cost=True)
        elif circuit.n_qubits > 28:
            # MPS path: use parameter-shift (adjoint/backprop need statevector)
            grad = parameter_shift_gradient(circuit, observable, params)
            cost = expectation(circuit.bind(params), observable)
        else:
            grad = self._grad_fn(circuit, observable, params)
            cost = expectation(circuit.bind(params), observable)
        result = self._update(params, grad)
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result, cost


class GradientDescent(_GradOptimizer):
    """Vanilla gradient descent."""
    def __init__(self, stepsize: float = 0.1, grad_fn=None):
        self.stepsize = stepsize
        self._grad_fn = grad_fn or adjoint_gradient

    def _update(self, params: dict[str, float], grad: dict[str, float]) -> dict[str, float]:
        return {k: params[k] - self.stepsize * grad[k] for k in params}


class Adam(_GradOptimizer):
    """Adam optimizer with bias-corrected moments."""
    def __init__(self, stepsize: float = 0.01, grad_fn=None,
                 beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-8):
        self.stepsize, self.beta1, self.beta2, self.eps = stepsize, beta1, beta2, eps
        self._grad_fn = grad_fn or adjoint_gradient
        self._m: dict[str, float] = {}
        self._v: dict[str, float] = {}
        self._t = 0

    def reset(self):
        """Clear moment accumulators for multi-start optimization."""
        self._m.clear()
        self._v.clear()
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
        params, circuit, observable = _unpack_args(params_or_circuit, circuit, observable)
        result = self._perturb_and_step(params, lambda p: self._eval(circuit.bind(p), observable))
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result

    def step_and_cost(self, params_or_circuit, circuit=None, observable=None):
        params, circuit, observable = _unpack_args(params_or_circuit, circuit, observable)
        cost = self._eval(circuit.bind(params), observable)
        result = self._perturb_and_step(params, lambda p: self._eval(circuit.bind(p), observable))
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result, cost


class QNG:
    """Quantum Natural Gradient — preconditions gradient with inverse QFI matrix."""
    def __init__(self, stepsize: float = 0.01, epsilon: float = 1e-3, grad_fn=None):
        self.stepsize, self.epsilon = stepsize, epsilon
        self._grad_fn = grad_fn or adjoint_gradient

    def step(self, params_or_circuit, circuit=None, observable=None, grad=None):
        params, circuit, observable = _unpack_args(params_or_circuit, circuit, observable)
        if grad is None: grad = self._grad_fn(circuit, observable, params)
        F = quantum_fisher_information(circuit, params)
        F_reg = F + self.epsilon * np.eye(len(F))
        keys = sorted(params)
        g = np.array([grad[k] for k in keys])
        nat_grad = np.linalg.solve(F_reg, g)
        result = {k: params[k] - self.stepsize * ng for k, ng in zip(keys, nat_grad)}
        if isinstance(params_or_circuit, Circuit): params_or_circuit.param_values = result
        return result
