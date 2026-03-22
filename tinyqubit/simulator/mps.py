"""MPS simulator — O(n * chi^2) memory, truncates at max_bond_dim."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate, _has_parameter, _get_gate_matrix, _GATE_1Q_CACHE
from .statevector import _build_gate_unitary

_SWAP4 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
_PAULI = {'X': np.array([[0,1],[1,0]], dtype=complex), 'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
          'Z': np.array([[1,0],[0,-1]], dtype=complex), 'I': np.eye(2, dtype=complex)}


class MPSState:
    """Lightweight wrapper around MPS tensors for dispatch."""
    __slots__ = ('tensors', 'n_qubits')
    def __init__(self, tensors: list[np.ndarray], n_qubits: int):
        self.tensors = tensors
        self.n_qubits = n_qubits


def _mps_init(n: int) -> list[np.ndarray]:
    return [np.array([[[1.0], [0.0]]], dtype=complex) for _ in range(n)]


def _mps_apply_1q(tensors: list[np.ndarray], q: int, U: np.ndarray):
    tensors[q] = np.einsum('ij,ajb->aib', U, tensors[q])


def _mps_apply_2q_adjacent(tensors: list[np.ndarray], q: int, U4: np.ndarray, max_chi: int):
    tl, tr = tensors[q], tensors[q + 1]
    chi_l, chi_r = tl.shape[0], tr.shape[2]
    theta = np.einsum('ail,ljb->aijb', tl, tr).reshape(chi_l, 4, chi_r)
    theta = np.einsum('ij,ajb->aib', U4, theta).reshape(chi_l * 2, 2 * chi_r)
    U, S, Vh = np.linalg.svd(theta, full_matrices=False)
    chi = min(len(S), max_chi)
    U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
    tensors[q] = (U * S).reshape(chi_l, 2, chi)
    tensors[q + 1] = Vh.reshape(chi, 2, chi_r)


def _mps_apply_2q(tensors: list[np.ndarray], q0: int, q1: int, U4: np.ndarray, max_chi: int):
    if abs(q0 - q1) == 1:
        lo = min(q0, q1)
        gate = _SWAP4 @ U4 @ _SWAP4 if q0 > q1 else U4
        _mps_apply_2q_adjacent(tensors, lo, gate, max_chi)
        return
    lo, hi = (q0, q1) if q0 < q1 else (q1, q0)
    for i in range(hi - 1, lo, -1):
        _mps_apply_2q_adjacent(tensors, i, _SWAP4, max_chi)
    gate = U4 if q0 < q1 else _SWAP4 @ U4 @ _SWAP4
    _mps_apply_2q_adjacent(tensors, lo, gate, max_chi)
    for i in range(lo + 1, hi):
        _mps_apply_2q_adjacent(tensors, i, _SWAP4, max_chi)


def _mps_apply_3q(tensors: list[np.ndarray], q0: int, q1: int, q2: int, U8: np.ndarray, max_chi: int):
    qs = sorted([(q0, 0), (q1, 1), (q2, 2)], key=lambda x: x[0])
    positions = [x[0] for x in qs]
    orig_order = [x[1] for x in qs]

    # SWAP route to adjacent positions
    anchor = positions[0]
    swaps_done = []
    for idx in range(1, 3):
        target = anchor + idx
        current = positions[idx]
        for i in range(current - 1, target - 1, -1):
            _mps_apply_2q_adjacent(tensors, i, _SWAP4, max_chi)
            swaps_done.append(i)
        positions[idx] = target

    # Permute gate matrix: physical site anchor+k holds original operand orig_order[k]
    perm = [0] * 3
    for k in range(3):
        perm[orig_order[k]] = k
    P = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        bits = [(i >> (2 - j)) & 1 for j in range(3)]
        new_bits = [bits[perm[j]] for j in range(3)]
        P[sum(b << (2 - k) for k, b in enumerate(new_bits)), i] = 1.0
    gate = P @ U8 @ P.T

    t0, t1, t2 = tensors[anchor], tensors[anchor + 1], tensors[anchor + 2]
    chi_l, chi_r = t0.shape[0], t2.shape[2]
    theta = np.einsum('ail,ljb,bkc->aijkc', t0, t1, t2).reshape(chi_l, 8, chi_r)
    theta = np.einsum('ij,ajb->aib', gate, theta).reshape(chi_l * 2, 4 * chi_r)
    # Two SVDs to split back into 3 tensors
    U, S, Vh = np.linalg.svd(theta, full_matrices=False)
    chi1 = min(len(S), max_chi)
    tensors[anchor] = (U[:, :chi1] * S[:chi1]).reshape(chi_l, 2, chi1)
    rest = Vh[:chi1, :].reshape(chi1 * 2, 2 * chi_r)
    U2, S2, Vh2 = np.linalg.svd(rest, full_matrices=False)
    chi2 = min(len(S2), max_chi)
    tensors[anchor + 1] = (U2[:, :chi2] * S2[:chi2]).reshape(chi1, 2, chi2)
    tensors[anchor + 2] = Vh2[:chi2, :].reshape(chi2, 2, chi_r)

    for i in reversed(swaps_done):
        _mps_apply_2q_adjacent(tensors, i, _SWAP4, max_chi)


def _mps_left_canonicalize(tensors: list[np.ndarray], up_to: int):
    for i in range(up_to):
        t = tensors[i]
        chi_l, d, chi_r = t.shape
        Q, R = np.linalg.qr(t.reshape(chi_l * d, chi_r))
        new_chi = Q.shape[1]
        tensors[i] = Q.reshape(chi_l, d, new_chi)
        tensors[i + 1] = np.einsum('ij,jkl->ikl', R, tensors[i + 1])


def _mps_measure(tensors: list[np.ndarray], q: int, rng: np.random.Generator) -> int:
    _mps_left_canonicalize(tensors, q)
    t = tensors[q]
    probs = np.array([np.linalg.norm(t[:, b, :]) ** 2 for b in range(2)])
    total = probs.sum()
    if total > 1e-15:
        probs /= total
    outcome = 1 if rng.random() < probs[1] else 0
    t = t.copy()
    t[:, 1 - outcome, :] = 0.0
    norm = np.linalg.norm(t)
    if norm > 1e-15:
        t /= norm
    tensors[q] = t
    return outcome


def _mps_reset(tensors: list[np.ndarray], q: int, rng: np.random.Generator):
    outcome = _mps_measure(tensors, q, rng)
    if outcome == 1:
        _mps_apply_1q(tensors, q, _GATE_1Q_CACHE[Gate.X])


def mps_to_statevector(tensors: list[np.ndarray]) -> np.ndarray:
    n = len(tensors)
    if n > 25:
        return np.zeros(0, dtype=complex)
    result = tensors[0]
    for i in range(1, n):
        result = np.einsum('...i,ijk->...jk', result, tensors[i])
    return result.reshape(-1)


def simulate_mps(circuit: Circuit, max_bond_dim: int = 256, seed: int | None = None) -> tuple[list[np.ndarray], dict[int, int]]:
    """MPS simulation. Returns (tensors, classical_bits)."""
    n = circuit.n_qubits
    for op in circuit.ops:
        if _has_parameter(op.params):
            raise TypeError(f"Cannot simulate: {op.gate.name} has unbound Parameter. Call circuit.bind() first.")
        for q in op.qubits:
            if not (0 <= q < n):
                raise ValueError(f"Invalid qubit index {q} for {n}-qubit circuit in {op.gate.name}")

    rng = np.random.default_rng(seed)
    classical = {i: 0 for i in range(circuit.n_classical)}
    tensors = _mps_init(n)

    for op in circuit.ops:
        if op.condition is not None and classical.get(op.condition[0]) != op.condition[1]:
            continue
        if op.gate == Gate.MEASURE:
            outcome = _mps_measure(tensors, op.qubits[0], rng)
            if op.classical_bit is not None:
                classical[op.classical_bit] = outcome
        elif op.gate == Gate.RESET:
            _mps_reset(tensors, op.qubits[0], rng)
        elif op.gate.n_qubits == 1:
            _mps_apply_1q(tensors, op.qubits[0], _get_gate_matrix(op.gate, op.params))
        elif op.gate.n_qubits == 2:
            U4 = _build_gate_unitary(op)
            _mps_apply_2q(tensors, op.qubits[0], op.qubits[1], U4, max_bond_dim)
        else:  # 3Q
            U8 = _build_gate_unitary(op)
            _mps_apply_3q(tensors, op.qubits[0], op.qubits[1], op.qubits[2], U8, max_bond_dim)

    return tensors, classical


# --- MPS-native measurement functions (no statevector conversion) ---

def mps_expectation(tensors: list[np.ndarray], observable) -> float:
    """Compute ⟨ψ|O|ψ⟩ from MPS tensors via transfer matrix contraction.
    Observable must have .terms attribute: list of (coeff, {qubit: pauli_str}).
    Cost: O(n · χ⁴ · |terms|) per term (χ⁴ from χ²×χ² matrix multiply)."""
    n = len(tensors)
    if hasattr(observable, '_matrix') and observable._matrix is not None:
        if n > 25:
            raise ValueError("Dense-matrix observable not supported for MPS with >25 qubits. Use Pauli-sum observables.")
        sv = mps_to_statevector(tensors)
        return np.vdot(sv, observable._matrix @ sv).real
    result = 0.0
    for coeff, paulis in observable.terms:
        tm = np.array([[1.0 + 0j]])  # 1×1 seed
        for q in range(n):
            T = tensors[q]
            P = _PAULI[paulis[q]] if q in paulis else _PAULI['I']
            # Transfer matrix: ⟨T| P |T⟩ contracted over physical index
            local = np.einsum('aib,ij,cjd->acbd', T.conj(), P, T)
            local = local.reshape(T.shape[0] * T.shape[0], T.shape[2] * T.shape[2])
            tm = tm @ local
        result += coeff * tm.item()
    return float(result.real)


def mps_sample(tensors: list[np.ndarray], shots: int, seed: int | None = None) -> dict[str, int]:
    """Sample bitstrings from MPS via sequential left-to-right conditional sampling.
    Left-canonicalizes once, then samples each shot in O(n·χ²)."""
    rng = np.random.default_rng(seed)
    n = len(tensors)
    canon = [t.copy() for t in tensors]
    _mps_left_canonicalize(canon, n - 1)
    counts: dict[str, int] = {}
    for _ in range(shots):
        bits = []
        env = np.ones((1, 1), dtype=complex)
        for q in range(n):
            T = canon[q]
            p = np.zeros(2)
            for b in range(2):
                v = env @ T[:, b, :]
                p[b] = np.sum(np.abs(v) ** 2).real
            total = p.sum()
            if total > 1e-15:
                p /= total
            outcome = 0 if rng.random() < p[0] else 1
            bits.append(str(outcome))
            env = env @ T[:, outcome, :]
            norm = np.linalg.norm(env)
            if norm > 1e-15:
                env = env / norm
        bs = ''.join(bits)
        counts[bs] = counts.get(bs, 0) + 1
    return counts


def mps_probabilities(tensors: list[np.ndarray], wires: list[int] | None = None) -> np.ndarray:
    """Compute measurement probabilities from MPS.
    If wires is None and n ≤ 25: full probability vector via statevector conversion.
    If wires is specified (≤12 qubits): marginal probabilities via partial contraction.
    """
    n = len(tensors)
    if wires is None:
        sv = mps_to_statevector(tensors)
        if len(sv) == 0:
            raise ValueError("Full probabilities for >25 qubits requires too much memory. Use wires= for marginals.")
        return np.abs(sv) ** 2
    if len(wires) > 12:
        raise ValueError(f"Marginal over {len(wires)} wires too large (max 12).")
    # Marginal probabilities via transfer matrix contraction over all 2^|wires| outcomes
    wire_set = set(wires)
    n_marginal = len(wires)
    probs = np.zeros(2 ** n_marginal)
    # For each outcome bitstring on the target wires
    for idx in range(2 ** n_marginal):
        bits = [(idx >> (n_marginal - 1 - k)) & 1 for k in range(n_marginal)]
        wire_bits = dict(zip(wires, bits))
        # Contract MPS with projectors on target wires, identity on rest
        tm = np.ones((1, 1), dtype=complex)
        for q in range(n):
            T = tensors[q]
            if q in wire_set:
                b = wire_bits[q]
                # Project onto |b⟩: ⟨T[:,b,:]|T[:,b,:]⟩
                local = np.einsum('ai,ci->ac', T[:, b, :].conj(), T[:, b, :])
            else:
                # Trace over physical index: sum_b ⟨T[:,b,:]|T[:,b,:]⟩
                local = np.einsum('aib,cid->ac', T.conj(), T)
            # local is (chi_l, chi_l) but we need (chi_l², chi_r²) format
            # Actually for projection/trace, the bra and ket bond dims don't mix
            tm = np.einsum('ac,cd->ad', tm, local)
        probs[idx] = tm.item().real
    probs = np.clip(probs, 0, None)
    total = probs.sum()
    if total > 1e-15:
        probs /= total
    return probs
