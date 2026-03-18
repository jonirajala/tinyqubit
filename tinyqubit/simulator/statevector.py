"""Statevector simulator with mid-circuit measurement, reset, conditionals, and noise.

NOTE: Noise uses quantum trajectory method (stochastic Kraus sampling on pure states).
For accurate ensemble statistics, run multiple shots with different seeds.
"""
from __future__ import annotations
import numpy as np
from math import pi
from typing import TYPE_CHECKING
from ..ir import Circuit, Gate, _has_parameter, _SQRT2_INV, _GATE_1Q_CACHE, _get_gate_matrix

if TYPE_CHECKING:
    from .noise import NoiseModel

_DIAG_PHASE = {Gate.Z: -1, Gate.S: 1j, Gate.SDG: -1j,
               Gate.T: np.exp(1j * pi / 4), Gate.TDG: np.exp(-1j * pi / 4)}

def _apply_diagonal_1q(state: np.ndarray, gate: Gate, qubit: int, n: int, params: tuple) -> np.ndarray:
    state = state.reshape([2] * n)
    idx1 = [slice(None)] * n; idx1[qubit] = 1
    if gate == Gate.RZ:
        idx0 = [slice(None)] * n; idx0[qubit] = 0
        t = params[0]
        state[tuple(idx0)] *= np.exp(-1j * t / 2)
        state[tuple(idx1)] *= np.exp(1j * t / 2)
    else:
        state[tuple(idx1)] *= _DIAG_PHASE[gate]
    return state.reshape(-1)

def _apply_single_qubit(state: np.ndarray, matrix: np.ndarray, qubit: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    idx0 = [slice(None)] * n; idx0[qubit] = 0
    idx1 = [slice(None)] * n; idx1[qubit] = 1
    idx0, idx1 = tuple(idx0), tuple(idx1)
    s0, s1 = state[idx0], state[idx1]
    out = np.empty_like(state)
    out[idx0] = matrix[0, 0] * s0 + matrix[0, 1] * s1
    out[idx1] = matrix[1, 0] * s0 + matrix[1, 1] * s1
    return out.reshape(-1)

def _apply_two_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, n: int, params: tuple = ()) -> np.ndarray:
    state = state.reshape([2] * n)
    def idx(v0, v1):
        i = [slice(None)] * n
        i[q0], i[q1] = v0, v1
        return tuple(i)
    if gate == Gate.CX: state[idx(1, 0)], state[idx(1, 1)] = state[idx(1, 1)].copy(), state[idx(1, 0)].copy()
    elif gate == Gate.CZ: state[idx(1, 1)] *= -1
    elif gate == Gate.SWAP: state[idx(0, 1)], state[idx(1, 0)] = state[idx(1, 0)].copy(), state[idx(0, 1)].copy()
    elif gate == Gate.CP: state[idx(1, 1)] *= np.exp(1j * params[0])
    elif gate == Gate.RZZ:
        t = params[0]
        em, ep = np.exp(-1j * t / 2), np.exp(1j * t / 2)
        state[idx(0, 0)] *= em; state[idx(0, 1)] *= ep
        state[idx(1, 0)] *= ep; state[idx(1, 1)] *= em
    elif gate == Gate.ECR:
        s00, s01, s10, s11 = state[idx(0,0)].copy(), state[idx(0,1)].copy(), state[idx(1,0)].copy(), state[idx(1,1)].copy()
        # ECR matrix: [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]] / sqrt(2)
        state[idx(0,0)] = _SQRT2_INV * (s10 + 1j * s11)
        state[idx(0,1)] = _SQRT2_INV * (1j * s10 + s11)
        state[idx(1,0)] = _SQRT2_INV * (s00 - 1j * s01)
        state[idx(1,1)] = _SQRT2_INV * (-1j * s00 + s01)
    return state.reshape(-1)

def _apply_three_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, q2: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    def idx(v0, v1, v2):
        i = [slice(None)] * n; i[q0], i[q1], i[q2] = v0, v1, v2
        return tuple(i)
    if gate == Gate.CCX:
        state[idx(1, 1, 0)], state[idx(1, 1, 1)] = state[idx(1, 1, 1)].copy(), state[idx(1, 1, 0)].copy()
    elif gate == Gate.CCZ:
        state[idx(1, 1, 1)] *= -1
    return state.reshape(-1)

def _apply_gate_noise(state: np.ndarray, op, noise_model, n: int, rng) -> np.ndarray:
    if noise_model is None: return state
    noise_list = noise_model.gate_noise.get(op.gate, noise_model.default_noise)
    if not noise_list: return state
    state = state.reshape([2] * n)
    for noise_fn in noise_list:
        for q in op.qubits: state = noise_fn(state, q, n, rng)

    return state.reshape(-1)

def _apply_measure(state: np.ndarray, qubit: int, n: int, rng) -> tuple[np.ndarray, int]:
    state = state.reshape([2] * n)
    probs = np.sum(np.abs(state) ** 2, axis=tuple(i for i in range(n) if i != qubit))
    outcome = 1 if rng.random() < probs[1] else 0
    idx = [slice(None)] * n
    idx[qubit] = 1 - outcome
    state[tuple(idx)] = 0.0
    norm = np.sqrt(probs[outcome])

    return (state / norm if norm > 1e-10 else state).reshape(-1), outcome

def _apply_reset(state: np.ndarray, qubit: int, n: int, rng) -> np.ndarray:
    state, outcome = _apply_measure(state, qubit, n, rng)
    return _apply_single_qubit(state, _GATE_1Q_CACHE[Gate.X], qubit, n) if outcome == 1 else state

def _collect_1q_block(ops: list, start: int) -> tuple[list[tuple[np.ndarray, int]], int]:
    fused, i = {}, start
    while i < len(ops):
        op = ops[i]
        if op.gate in (Gate.MEASURE, Gate.RESET) or op.condition is not None or op.gate.n_qubits >= 2: break
        q = op.qubits[0]
        mat = _get_gate_matrix(op.gate, op.params)
        fused[q] = mat @ fused[q] if q in fused else mat
        i += 1
    return [(m, q) for q, m in fused.items()], i

def _apply_1q_matmul(state: np.ndarray, buf: np.ndarray, matrix: np.ndarray, qubit: int, n: int, tmp: np.ndarray):
    """Apply 1Q gate via matmul broadcast (middle qubits) or ufunc out= (edge qubits)."""
    nq, nr = 1 << qubit, 1 << (n - qubit - 1)
    if min(nq, nr) > 1:
        np.matmul(matrix, state.reshape(nq, 2, nr), out=buf.reshape(nq, 2, nr))
    else:
        st, bt = state.reshape([2] * n), buf.reshape([2] * n)
        idx0 = [slice(None)] * n; idx0[qubit] = 0
        idx1 = [slice(None)] * n; idx1[qubit] = 1
        idx0, idx1 = tuple(idx0), tuple(idx1)
        s0, s1 = st[idx0], st[idx1]
        t = tmp[:s0.size].reshape(s0.shape)
        np.multiply(matrix[0, 0], s0, out=t); np.multiply(matrix[0, 1], s1, out=bt[idx0]); np.add(t, bt[idx0], out=bt[idx0])
        np.multiply(matrix[1, 0], s0, out=t); np.multiply(matrix[1, 1], s1, out=bt[idx1]); np.add(t, bt[idx1], out=bt[idx1])

def _apply_batch_1q(state: np.ndarray, gates: list[tuple[np.ndarray, int]], n: int,
                    buf: np.ndarray | None = None, tmp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not gates: return state, buf, tmp
    if buf is None: buf = np.empty_like(state)
    if tmp is None: tmp = np.empty(1 << (n - 1), dtype=state.dtype)
    for matrix, qubit in gates:
        _apply_1q_matmul(state, buf, matrix, qubit, n, tmp)
        state, buf = buf, state
    return state, buf, tmp


def simulate_statevector(circuit: Circuit, n: int, seed, noise_model, batch_ops) -> tuple[np.ndarray, dict[int, int]]:
    rng = np.random.default_rng(seed)
    classical = {i: 0 for i in range(circuit.n_classical)}
    if circuit._initial_state is not None:
        state = circuit._initial_state.copy()
    else:
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

    ops, ops_iter = circuit.ops, iter(enumerate(circuit.ops))
    buf = tmp = None
    if batch_ops and noise_model is None and n >= 10:
        buf, tmp = np.empty_like(state), np.empty(1 << (n - 1), dtype=state.dtype)
    for i, op in ops_iter:
        if op.condition is not None and classical.get(op.condition[0]) != op.condition[1]: continue
        if op.gate == Gate.MEASURE:
            state, outcome = _apply_measure(state, op.qubits[0], n, rng)
            if op.classical_bit is not None:
                if noise_model is not None and noise_model.readout_error_fn is not None:
                    outcome = noise_model.readout_error_fn(outcome, rng)
                classical[op.classical_bit] = outcome
        elif op.gate == Gate.RESET:
            state = _apply_reset(state, op.qubits[0], n, rng)
        elif op.gate.n_qubits == 1:
            if buf is not None:
                group, end_i = _collect_1q_block(ops, i)
                if len(group) > 1:
                    state, buf, tmp = _apply_batch_1q(state, group, n, buf, tmp)
                    for _ in range(end_i - i - 1): next(ops_iter)
                    continue
            if op.gate in _DIAG_PHASE or op.gate == Gate.RZ:
                state = _apply_diagonal_1q(state, op.gate, op.qubits[0], n, op.params)
            else:
                state = _apply_single_qubit(state, _get_gate_matrix(op.gate, op.params), op.qubits[0], n)
            state = _apply_gate_noise(state, op, noise_model, n, rng)
        elif op.gate.n_qubits == 2:
            state = _apply_two_qubit(state, op.gate, op.qubits[0], op.qubits[1], n, op.params)
            state = _apply_gate_noise(state, op, noise_model, n, rng)
        else:  # 3Q
            state = _apply_three_qubit(state, op.gate, *op.qubits, n)
            state = _apply_gate_noise(state, op, noise_model, n, rng)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10, "statevector norm drifted"
    return state, classical


def _build_gate_unitary(op) -> np.ndarray:
    k = op.gate.n_qubits
    if k == 1: return _get_gate_matrix(op.gate, op.params)
    dim = 2 ** k
    U = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex); e[j] = 1.0
        U[:, j] = _apply_two_qubit(e, op.gate, 0, 1, k, op.params) if k == 2 else _apply_three_qubit(e, op.gate, 0, 1, 2, k)
    return U
