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
    if gate == Gate.CX:
        tmp = state[idx(1, 0)].copy(); state[idx(1, 0)] = state[idx(1, 1)]; state[idx(1, 1)] = tmp
    elif gate == Gate.CZ: state[idx(1, 1)] *= -1
    elif gate == Gate.SWAP:
        tmp = state[idx(0, 1)].copy(); state[idx(0, 1)] = state[idx(1, 0)]; state[idx(1, 0)] = tmp
    elif gate == Gate.CP: state[idx(1, 1)] *= np.exp(1j * params[0])
    elif gate == Gate.RZZ:
        t = params[0]
        em, ep = np.exp(-1j * t / 2), np.exp(1j * t / 2)
        state[idx(0, 0)] *= em; state[idx(0, 1)] *= ep
        state[idx(1, 0)] *= ep; state[idx(1, 1)] *= em
    elif gate == Gate.ECR:
        s00, s01, s10, s11 = state[idx(0,0)].copy(), state[idx(0,1)].copy(), state[idx(1,0)].copy(), state[idx(1,1)].copy()
        state[idx(0,0)] = _SQRT2_INV * (s10 + 1j * s11)
        state[idx(0,1)] = _SQRT2_INV * (1j * s10 + s11)
        state[idx(1,0)] = _SQRT2_INV * (s00 - 1j * s01)
        state[idx(1,1)] = _SQRT2_INV * (-1j * s00 + s01)
    elif gate == Gate.SEXC:
        t = params[0]
        c, s = np.cos(t / 2), np.sin(t / 2)
        s01, s10 = state[idx(0, 1)].copy(), state[idx(1, 0)].copy()
        state[idx(0, 1)] = c * s01 - s * s10
        state[idx(1, 0)] = s * s01 + c * s10
    return state.reshape(-1)

def _apply_three_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, q2: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    def idx(v0, v1, v2):
        i = [slice(None)] * n; i[q0], i[q1], i[q2] = v0, v1, v2
        return tuple(i)
    if gate == Gate.CCX:
        tmp = state[idx(1, 1, 0)].copy(); state[idx(1, 1, 0)] = state[idx(1, 1, 1)]; state[idx(1, 1, 1)] = tmp
    elif gate == Gate.CCZ:
        state[idx(1, 1, 1)] *= -1
    return state.reshape(-1)

def _apply_four_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, q2: int, q3: int, n: int, params: tuple = ()) -> np.ndarray:
    state = state.reshape([2] * n)
    def idx(v0, v1, v2, v3):
        i = [slice(None)] * n; i[q0], i[q1], i[q2], i[q3] = v0, v1, v2, v3
        return tuple(i)
    if gate == Gate.DEXC:
        # Givens rotation in |0011⟩ ↔ |1100⟩ subspace
        t = params[0]
        c, s = np.cos(t / 2), np.sin(t / 2)
        s0011 = state[idx(0, 0, 1, 1)].copy()
        s1100 = state[idx(1, 1, 0, 0)].copy()
        state[idx(0, 0, 1, 1)] = c * s0011 - s * s1100
        state[idx(1, 1, 0, 0)] = s * s0011 + c * s1100
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

def _build_perm(gate_ops: tuple[tuple[str, int, int], ...], n: int) -> np.ndarray:
    N = 1 << n
    identity = np.arange(N, dtype=np.int64)
    perm = identity.copy()
    single = np.empty(N, dtype=np.int64)
    for gate, q0, q1 in gate_ops:
        if gate == 'CX':
            np.copyto(single, identity)
            single[(single & (1 << (n - 1 - q0))) != 0] ^= (1 << (n - 1 - q1))
            perm = perm[single]
        else:  # SWAP: XOR both bits where they differ
            b0, b1 = 1 << (n - 1 - q0), 1 << (n - 1 - q1)
            np.copyto(single, identity)
            mask = ((single >> (n - 1 - q0)) ^ (single >> (n - 1 - q1))) & 1
            single ^= mask * (b0 | b1)
            perm = perm[single]
    return perm

_perm_cache: dict[tuple, np.ndarray] = {}
_PERM_CACHE_MAX = 64

def _get_perm(gate_ops: tuple[tuple[str, int, int], ...], n: int) -> np.ndarray:
    key = (gate_ops, n)
    if key not in _perm_cache:
        if len(_perm_cache) >= _PERM_CACHE_MAX:
            _perm_cache.pop(next(iter(_perm_cache)))
        _perm_cache[key] = _build_perm(gate_ops, n)
    return _perm_cache[key]

def _collect_perm_block(ops: list, start: int) -> tuple[tuple[tuple[str, int, int], ...] | None, int]:
    i, gate_ops = start, []
    while i < len(ops):
        op = ops[i]
        if op.condition is not None or op.gate not in (Gate.CX, Gate.SWAP): break
        gate_ops.append(('CX' if op.gate == Gate.CX else 'SWAP', op.qubits[0], op.qubits[1]))
        i += 1
    return (tuple(gate_ops), i) if len(gate_ops) >= 2 else (None, start)


def _collect_diag_2q_block(ops: list, start: int) -> tuple[list | None, int]:
    i, block = start, []
    while i < len(ops):
        op = ops[i]
        if op.gate not in _DIAG_2Q or op.condition is not None: break
        block.append(op)
        i += 1
    return (block, i) if len(block) >= 2 else (None, start)

_DIAG_2Q = frozenset({Gate.CZ, Gate.CP, Gate.RZZ})

def _collect_fusable_block(ops: list, start: int) -> tuple[list[tuple[np.ndarray, int]], list, int]:
    fused, diag_2q, i, all_diag = {}, [], start, True
    while i < len(ops):
        op = ops[i]
        if op.condition is not None or op.gate in (Gate.MEASURE, Gate.RESET): break
        if op.gate.n_qubits == 1:
            q = op.qubits[0]
            mat = _get_gate_matrix(op.gate, op.params)
            fused[q] = mat @ fused[q] if q in fused else mat
            if all_diag and (mat[0, 1] != 0j or mat[1, 0] != 0j): all_diag = False
            i += 1
        elif op.gate in _DIAG_2Q and all_diag:
            diag_2q.append(op)
            i += 1
        else:
            break
    return [(m, q) for q, m in fused.items()], diag_2q, i

def _apply_1q_matmul(state: np.ndarray, buf: np.ndarray, matrix: np.ndarray, qubit: int, n: int, tmp: np.ndarray):
    """Apply 1Q gate via ufunc (edge qubits) or matmul broadcast (middle qubits)."""
    nq, nr = 1 << qubit, 1 << (n - qubit - 1)
    if nq == 1 or nr <= 1:
        # Manual ufunc: faster than BLAS for extreme aspect ratios
        s = state.reshape(nq, 2, nr)
        b = buf.reshape(nq, 2, nr)
        t = tmp[:nq * nr].reshape(nq, nr)
        np.multiply(matrix[0, 0], s[:, 0, :], out=t)
        np.multiply(matrix[0, 1], s[:, 1, :], out=b[:, 0, :])
        np.add(t, b[:, 0, :], out=b[:, 0, :])
        np.multiply(matrix[1, 0], s[:, 0, :], out=t)
        np.multiply(matrix[1, 1], s[:, 1, :], out=b[:, 1, :])
        np.add(t, b[:, 1, :], out=b[:, 1, :])
        return
    np.matmul(matrix, state.reshape(nq, 2, nr), out=buf.reshape(nq, 2, nr))

def _apply_batch_1q(state: np.ndarray, gates: list[tuple[np.ndarray, int]], n: int,
                    buf: np.ndarray | None = None, tmp: np.ndarray | None = None,
                    diag_2q: list | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not gates and not diag_2q: return state, buf, tmp
    if buf is None: buf = np.empty_like(state)
    if tmp is None: tmp = np.empty(1 << (n - 1), dtype=state.dtype)
    # Separate diagonal and non-diagonal gates
    diag, non_diag = [], []
    for matrix, qubit in gates:
        (diag if matrix[0, 1] == 0j and matrix[1, 0] == 0j else non_diag).append((matrix, qubit))
    # Group adjacent qubits into kron matmul to reduce state passes
    non_diag.sort(key=lambda x: x[1])
    nd_i = 0
    while nd_i < len(non_diag):
        # Find longest run of consecutive qubits (up to 5 → 32×32 matmul)
        run = 1
        while run < 5 and nd_i + run < len(non_diag) and non_diag[nd_i + run][1] == non_diag[nd_i][1] + run:
            run += 1
        if run >= 2:
            q_first = non_diag[nd_i][1]
            q_last = non_diag[nd_i + run - 1][1]
            combined = non_diag[nd_i][0]
            for j in range(1, run):
                b = non_diag[nd_i + j][0]
                combined = (combined[:, np.newaxis, :, np.newaxis] * b[np.newaxis, :, np.newaxis, :]).reshape(
                    combined.shape[0] * 2, combined.shape[1] * 2)
            dim = 1 << run
            nq = 1 << q_first
            nr = 1 << (n - q_last - 1)
            if nr <= 1:
                # NOTE: 2D GEMM much faster than 3D batch matmul for nr=1
                np.matmul(state.reshape(nq, dim), combined.T, out=buf.reshape(nq, dim))
            elif nq == 1:
                np.matmul(combined, state.reshape(dim, nr), out=buf.reshape(dim, nr))
            else:
                np.matmul(combined, state.reshape(nq, dim, nr), out=buf.reshape(nq, dim, nr))
            state, buf = buf, state
            nd_i += run
        else:
            _apply_1q_matmul(state, buf, non_diag[nd_i][0], non_diag[nd_i][1], n, tmp)
            state, buf = buf, state
            nd_i += 1
    # Fuse diagonal 1Q + diagonal 2Q gates into single phase vector (reuse buf to avoid alloc)
    if diag or diag_2q:
        buf[:] = 1.0
        phase = buf
        for matrix, qubit in diag:
            nq, nr = 1 << qubit, 1 << (n - qubit - 1)
            p = phase.reshape(nq, 2, nr)
            p[:, 0, :] *= matrix[0, 0]
            p[:, 1, :] *= matrix[1, 1]
        if diag_2q:
            pt = phase.reshape([2] * n)
            sl = slice(None)
            for op in diag_2q:
                q0, q1 = op.qubits
                i11 = [sl] * n; i11[q0] = 1; i11[q1] = 1; i11 = tuple(i11)
                if op.gate == Gate.CZ:
                    pt[i11] *= -1
                elif op.gate == Gate.CP:
                    pt[i11] *= np.exp(1j * op.params[0])
                else:  # RZZ
                    t = op.params[0]
                    em, ep = np.exp(-1j * t / 2), np.exp(1j * t / 2)
                    i00 = [sl] * n; i00[q0] = 0; i00[q1] = 0
                    i01 = [sl] * n; i01[q0] = 0; i01[q1] = 1
                    i10 = [sl] * n; i10[q0] = 1; i10[q1] = 0
                    pt[tuple(i00)] *= em; pt[tuple(i01)] *= ep
                    pt[tuple(i10)] *= ep; pt[i11] *= em
        state *= phase
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
        elif (nq := op.gate.n_qubits) == 1:
            if buf is not None:
                group, diag_2q_ops, end_i = _collect_fusable_block(ops, i)
                if len(group) > 1 or diag_2q_ops:
                    state, buf, tmp = _apply_batch_1q(state, group, n, buf, tmp, diag_2q_ops or None)
                    for _ in range(end_i - i - 1): next(ops_iter)
                    continue
            if op.gate in _DIAG_PHASE or op.gate == Gate.RZ:
                state = _apply_diagonal_1q(state, op.gate, op.qubits[0], n, op.params)
            else:
                state = _apply_single_qubit(state, _get_gate_matrix(op.gate, op.params), op.qubits[0], n)
            if noise_model is not None: state = _apply_gate_noise(state, op, noise_model, n, rng)
        elif nq == 2:
            if buf is not None and noise_model is None:
                if op.gate in (Gate.CX, Gate.SWAP):
                    perm_ops, end_i = _collect_perm_block(ops, i)
                    if perm_ops is not None:
                        np.take(state, _get_perm(perm_ops, n), out=buf)
                        state, buf = buf, state
                        for _ in range(end_i - i - 1): next(ops_iter)
                        continue
                elif op.gate in _DIAG_2Q:
                    d2q_block, end_i = _collect_diag_2q_block(ops, i)
                    if d2q_block is not None:
                        state, buf, tmp = _apply_batch_1q(state, [], n, buf, tmp, d2q_block)
                        for _ in range(end_i - i - 1): next(ops_iter)
                        continue
            state = _apply_two_qubit(state, op.gate, op.qubits[0], op.qubits[1], n, op.params)
            if noise_model is not None: state = _apply_gate_noise(state, op, noise_model, n, rng)
        elif nq == 3:
            state = _apply_three_qubit(state, op.gate, *op.qubits, n)
            if noise_model is not None: state = _apply_gate_noise(state, op, noise_model, n, rng)
        else:  # 4Q
            state = _apply_four_qubit(state, op.gate, *op.qubits, n, op.params)
            if noise_model is not None: state = _apply_gate_noise(state, op, noise_model, n, rng)
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
