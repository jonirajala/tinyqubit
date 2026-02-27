"""Parameter-shift, finite-difference, and adjoint gradient computation."""
from __future__ import annotations
import numpy as np
from .ir import Circuit, Gate, Parameter, _GATE_ADJOINT, _PARAM_GATES
from .observable import Observable, expectation, _PAULI_MATRIX
from .simulator import simulate, sample, _apply_single_qubit, _apply_two_qubit, _apply_three_qubit, _get_gate_matrix

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
                             values: dict[str, float], shots: int | None = None) -> dict[str, float]:
    """Compute gradient via the parameter-shift rule: df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2"""
    shift = np.pi / 2
    exp_fn = expectation if shots is None else lambda c, o: _shot_expectation(c, o, shots)
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**values, param.name: values[param.name] + shift}
        v_minus = {**values, param.name: values[param.name] - shift}
        e_plus = exp_fn(circuit.bind(v_plus), observable)
        e_minus = exp_fn(circuit.bind(v_minus), observable)
        grad[param.name] = (e_plus - e_minus) / 2
    return grad


def finite_difference_gradient(circuit: Circuit, observable: Observable,
                               values: dict[str, float], epsilon: float = 1e-7) -> dict[str, float]:
    """Compute gradient via symmetric finite differences: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / 2ε"""
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**values, param.name: values[param.name] + epsilon}
        v_minus = {**values, param.name: values[param.name] - epsilon}
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


def adjoint_gradient(circuit: Circuit, observable: Observable, values: dict[str, float]) -> dict[str, float]:
    """Compute all gradients in one forward + backward pass (adjoint differentiation)."""
    bound = circuit.bind(values)
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
                # 1Q rotation (exp(-iθG/2)): grad += Im(⟨λ|G|ψ⟩)
                g_psi = _apply_single_qubit(state, _PAULI_MATRIX[_GENERATORS[op.gate]], op.qubits[0], n)
                grad[name] += np.vdot(lam, g_psi).imag
            elif op.gate == Gate.CP:
                # CP(θ) = diag(1,1,1,e^iθ): grad += -2 Im(⟨λ_{11}|ψ_{11}⟩)
                q0, q1 = op.qubits
                idx = [slice(None)] * n
                idx[q0], idx[q1] = 1, 1
                idx = tuple(idx)
                grad[name] += -2 * np.vdot(lam.reshape([2] * n)[idx], state.reshape([2] * n)[idx]).imag
        state = _unapply_op(state, op, n)
        lam = _unapply_op(lam, op, n)
    return grad


def gradient_landscape(circuit: Circuit, param_names: list[str], observable: Observable,
                       base_values: dict[str, float], n_points: int = 50,
                       ranges: list[tuple[float, float]] | None = None) -> np.ndarray:
    """2D expectation sweep over two parameters. Returns (n_points, n_points) array."""
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
