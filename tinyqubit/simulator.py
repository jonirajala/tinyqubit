"""
Statevector simulator with mid-circuit measurement, reset, conditionals, and noise.

Noise simulation uses quantum trajectory method (stochastic unraveling):
- Each noise function maps pure state → pure state by sampling Kraus outcomes
- This is NOT a general density matrix simulator
- For accurate ensemble statistics, run multiple shots with different seeds

Classical register behavior:
- Bits initialize to 0
- Conditionals check against these values
- MEASURE updates the bit; MEASURE without classical_bit doesn't store result
"""
from __future__ import annotations
import numpy as np
from math import sqrt, cos, sin, pi
from typing import TYPE_CHECKING
from .ir import Circuit, Gate, _has_parameter

if TYPE_CHECKING:
    from .noise import NoiseModel

# Gate matrices
_SQRT2_INV, _T = 1 / sqrt(2), np.exp(1j * pi / 4)
_GATE_1Q_CACHE = {
    Gate.X: np.array([[0, 1], [1, 0]], dtype=complex),
    Gate.Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
    Gate.Z: np.array([[1, 0], [0, -1]], dtype=complex),
    Gate.H: np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT2_INV,
    Gate.S: np.array([[1, 0], [0, 1j]], dtype=complex),
    Gate.SDG: np.array([[1, 0], [0, -1j]], dtype=complex),
    Gate.T: np.array([[1, 0], [0, _T]], dtype=complex),
    Gate.TDG: np.array([[1, 0], [0, np.conj(_T)]], dtype=complex),
}
_GATE_1Q_PARAM = {
    Gate.RX: lambda t: np.array([[cos(t/2), -1j*sin(t/2)], [-1j*sin(t/2), cos(t/2)]], dtype=complex),
    Gate.RY: lambda t: np.array([[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]], dtype=complex),
    Gate.RZ: lambda t: np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=complex),
}

def _get_gate_matrix(gate: Gate, params: tuple) -> np.ndarray:
    return _GATE_1Q_CACHE[gate] if gate in _GATE_1Q_CACHE else _GATE_1Q_PARAM[gate](params[0])

def _apply_single_qubit(state: np.ndarray, matrix: np.ndarray, qubit: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    axes = list(range(n))
    axes[qubit], axes[-1] = axes[-1], axes[qubit]
    state = np.transpose(state, axes)
    state = np.tensordot(state, matrix, axes=([-1], [1]))

    return np.transpose(state, axes).reshape(-1)

def _apply_two_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, n: int, params: tuple = ()) -> np.ndarray:
    state = state.reshape([2] * n)
    new = state.copy()
    
    def idx(v0, v1):
        i = [slice(None)] * n
        i[q0], i[q1] = v0, v1
        return tuple(i)

    if gate == Gate.CX: new[idx(1,0)], new[idx(1,1)] = state[idx(1,1)].copy(), state[idx(1,0)].copy()
    elif gate == Gate.CZ: new[idx(1,1)] *= -1
    elif gate == Gate.SWAP: new[idx(0,1)], new[idx(1,0)] = state[idx(1,0)].copy(), state[idx(0,1)].copy()
    elif gate == Gate.CP: new[idx(1,1)] *= np.exp(1j * params[0])

    return new.reshape(-1)

def _apply_three_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, q2: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    new = state.copy()
    def idx(v0, v1, v2):
        i = [slice(None)] * n
        i[q0], i[q1], i[q2] = v0, v1, v2
        return tuple(i)
    if gate == Gate.CCX:
        new[idx(1,1,0)], new[idx(1,1,1)] = state[idx(1,1,1)].copy(), state[idx(1,1,0)].copy()
    elif gate == Gate.CCZ:
        new[idx(1,1,1)] *= -1
    return new.reshape(-1)

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

def _find_parallel_1q_groups(ops: list, start: int) -> tuple[list[tuple[np.ndarray, int]], int]:
    group, used, i = [], set(), start
    while i < len(ops):
        op = ops[i]
        if op.gate in (Gate.MEASURE, Gate.RESET) or op.condition is not None or op.gate.n_qubits >= 2: break
        q = op.qubits[0]
        if q in used: break
        group.append((_get_gate_matrix(op.gate, op.params), q))
        used.add(q)
        i += 1

    return group, i

def _apply_batch_1q(state: np.ndarray, gates: list[tuple[np.ndarray, int]], n: int) -> np.ndarray:
    if not gates: return state
    if len(gates) == 1: return _apply_single_qubit(state, gates[0][0], gates[0][1], n)
    indices = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if n + len(gates) > len(indices): raise ValueError("Too many indices for einsum")
    state = state.reshape([2] * n)
    state_idx, out_idx = list(indices[:n]), list(indices[:n])
    gate_strs, operands, next_new = [], [], n
    for matrix, qubit in gates:
        old, new = state_idx[qubit], indices[next_new]
        next_new += 1
        gate_strs.append(new + old)
        operands.append(matrix.astype(state.dtype, copy=False))
        out_idx[qubit] = new

    return np.einsum(','.join(gate_strs) + ',' + ''.join(state_idx) + '->' + ''.join(out_idx),
                     *operands, state, optimize=True).reshape(-1)

def simulate(circuit: Circuit, seed: int | None = None, noise_model: "NoiseModel | None" = None,
             batch_ops: bool = False) -> tuple[np.ndarray, dict[int, int]]:
    """Simulate circuit. Returns (statevector, classical_bits dict)."""
    n = circuit.n_qubits
    rng = np.random.default_rng(seed)
    classical = {i: 0 for i in range(circuit.n_classical)}  # Initialize all bits to 0
    if circuit._initial_state is not None:
        state = circuit._initial_state.copy()
    else:
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0

    # Validate qubit indices and unbound parameters
    for op in circuit.ops:
        if _has_parameter(op.params):
            raise TypeError(f"Cannot simulate: {op.gate.name} has unbound Parameter. Call circuit.bind() first.")
        for q in op.qubits:
            if not (0 <= q < n):
                raise ValueError(f"Invalid qubit index {q} for {n}-qubit circuit in {op.gate.name}")

    ops, ops_iter = circuit.ops, iter(enumerate(circuit.ops))
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
            if batch_ops and noise_model is None:
                group, end_i = _find_parallel_1q_groups(ops, i)
                if len(group) > 1:
                    state = _apply_batch_1q(state, group, n)
                    for _ in range(end_i - i - 1): next(ops_iter)
                    continue
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

def states_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if two states are equal up to global phase."""
    if a.shape != b.shape: return False
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < tol or norm_b < tol: return norm_a < tol and norm_b < tol
    return np.isclose(np.abs(np.vdot(a, b)) / (norm_a * norm_b), 1.0, atol=tol)

def sample(state: np.ndarray, shots: int, seed: int = None) -> dict[str, int]:
    """Sample measurement outcomes from statevector. Returns {bitstring: count}."""
    rng = np.random.default_rng(seed)
    probs = np.abs(state) ** 2
    probs /= probs.sum()  # Normalize to handle numerical drift
    outcomes = rng.choice(len(state), size=shots, p=probs)
    values, counts = np.unique(outcomes, return_counts=True)
    n_bits = int(np.log2(len(state)))
    return {format(v, f'0{n_bits}b'): int(c) for v, c in zip(values, counts)}

def to_unitary(circuit: Circuit) -> np.ndarray:
    """Build full unitary matrix of the circuit. Max 12 qubits."""
    n = circuit.n_qubits
    if n > 12: raise ValueError("to_unitary supports at most 12 qubits")
    if circuit._initial_state is not None: raise ValueError("to_unitary does not support initialized circuits")
    for op in circuit.ops:
        if op.gate in (Gate.MEASURE, Gate.RESET): raise ValueError(f"to_unitary does not support {op.gate.name}")
        if op.condition is not None: raise ValueError("to_unitary does not support conditional operations")
        if _has_parameter(op.params): raise TypeError(f"Cannot compute unitary: {op.gate.name} has unbound Parameter")
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for op in circuit.ops:
        if op.gate.n_qubits == 1:
            mat = _get_gate_matrix(op.gate, op.params)
            for k in range(dim): U[:, k] = _apply_single_qubit(U[:, k], mat, op.qubits[0], n)
        elif op.gate.n_qubits == 2:
            for k in range(dim): U[:, k] = _apply_two_qubit(U[:, k], op.gate, op.qubits[0], op.qubits[1], n, op.params)
        else:
            for k in range(dim): U[:, k] = _apply_three_qubit(U[:, k], op.gate, *op.qubits, n)
    return U

def probabilities(circuit: Circuit, wires: list[int] | None = None, seed: int | None = None) -> np.ndarray:
    """Compute measurement probabilities. Optionally marginal over specified wires."""
    state, _ = simulate(circuit, seed=seed)
    probs = np.abs(state) ** 2
    if wires is None: return probs
    n = circuit.n_qubits
    probs = probs.reshape([2] * n)
    trace_out = tuple(i for i in range(n) if i not in wires)
    if trace_out: probs = probs.sum(axis=trace_out)
    order = [sorted(wires).index(w) for w in wires]
    result = probs.transpose(order).flatten()
    assert abs(result.sum() - 1.0) < 1e-10, "probabilities do not sum to 1"
    return result

def marginal_counts(counts: dict[str, int], wires: list[int]) -> dict[str, int]:
    """Extract marginal counts for specified wire positions."""
    result = {}
    for bitstring, count in counts.items():
        key = ''.join(bitstring[w] for w in wires)
        result[key] = result.get(key, 0) + count
    return result

def simulate_batch(circuits: list[Circuit]) -> list[tuple[np.ndarray, dict[int, int]]]:
    """Simulate multiple circuits. Returns list of (statevector, classical_bits)."""
    return [simulate(c) for c in circuits]


def verify(original: Circuit, compiled: Circuit, tracker=None, tol: float = 1e-9, n_samples: int = 5) -> bool:
    """Check circuit equivalence, including across random parameter values for parametric circuits."""

    def _check_equiv(c1: Circuit, c2: Circuit) -> bool:
        n = c1.n_qubits
        has_nonunitary = any(op.gate in (Gate.MEASURE, Gate.RESET) or op.condition is not None for op in c1.ops)
        if n <= 12 and not has_nonunitary:
            U1, U2 = to_unitary(c1), to_unitary(c2)
            if tracker is not None:
                perm = [tracker.logical_to_phys(i) for i in range(n)]
                U2 = np.transpose(U2.reshape([2]*n + [2]*n), perm + [p + n for p in perm]).reshape(2**n, 2**n)
            return abs(np.trace(U1.conj().T @ U2)) / (2**n) > 1 - tol
        # Statevector fallback
        s1, _ = simulate(c1)
        s2, _ = simulate(c2)
        if tracker is not None:
            perm = [tracker.logical_to_phys(i) for i in range(n)]
            s2 = np.transpose(s2.reshape([2] * n), perm).reshape(-1)
        return states_equal(s1, s2, tol)

    if original.is_parameterized:
        rng = np.random.default_rng(42)
        param_names = sorted(p.name for p in original.parameters)
        for _ in range(n_samples):
            vals = {name: rng.uniform(0, 2 * pi) for name in param_names}
            c1, c2 = original.bind(vals), compiled.bind(vals)
            if not _check_equiv(c1, c2):
                return False
        return True
    return _check_equiv(original, compiled)
