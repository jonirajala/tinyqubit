"""
Minimal statevector simulator for testing.


Contains:
    - simulate(circuit) -> statevector
    - states_equal(a, b) -> bool (up to global phase)
    - sample(state, shots, seed) -> counts

Doesnt yet support mid circuit measure

"""
import numpy as np
from math import sqrt, cos, sin, pi
from .ir import Circuit, Gate


# Gate Matrices

_SQRT2_INV = 1 / sqrt(2)
_T_PHASE = np.exp(1j * pi / 4)

def _rx(theta: float) -> np.ndarray:
    c, s = cos(theta / 2), sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def _ry(theta: float) -> np.ndarray:
    c, s = cos(theta / 2), sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

# All single-qubit gates (2Q gates handled separately via direct indexing)
GATE_1Q = {
    Gate.X: lambda _: np.array([[0, 1], [1, 0]], dtype=complex),
    Gate.Y: lambda _: np.array([[0, -1j], [1j, 0]], dtype=complex),
    Gate.Z: lambda _: np.array([[1, 0], [0, -1]], dtype=complex),
    Gate.H: lambda _: np.array([[1, 1], [1, -1]], dtype=complex) * _SQRT2_INV,
    Gate.S: lambda _: np.array([[1, 0], [0, 1j]], dtype=complex),
    Gate.T: lambda _: np.array([[1, 0], [0, _T_PHASE]], dtype=complex),
    Gate.RX: lambda p: _rx(p[0]),
    Gate.RY: lambda p: _ry(p[0]),
    Gate.RZ: lambda p: _rz(p[0]),
}


# State Operations

def _apply_single_qubit(state: np.ndarray, matrix: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate to state."""
    # Reshape state to tensor of shape (2, 2, ..., 2)
    state = state.reshape([2] * n_qubits)

    # Apply matrix via einsum
    # Move target qubit to last axis, apply matrix, move back
    axes = list(range(n_qubits))
    axes[qubit], axes[-1] = axes[-1], axes[qubit]
    state = np.transpose(state, axes)

    # Apply matrix to last axis
    state = np.tensordot(state, matrix, axes=([-1], [1]))

    # Transpose back
    state = np.transpose(state, axes)

    return state.reshape(-1)


def _apply_two_qubit(state: np.ndarray, gate: Gate, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """Apply two-qubit gate (CX, CZ, SWAP) to state."""
    state = state.reshape([2] * n_qubits)
    new_state = state.copy()

    if gate == Gate.CX:
        # Flip target when control is 1: |10⟩ <-> |11⟩
        idx_10 = [slice(None)] * n_qubits
        idx_10[q0] = 1
        idx_10[q1] = 0
        idx_11 = [slice(None)] * n_qubits
        idx_11[q0] = 1
        idx_11[q1] = 1
        new_state[tuple(idx_10)], new_state[tuple(idx_11)] = state[tuple(idx_11)].copy(), state[tuple(idx_10)].copy()

    elif gate == Gate.CZ:
        # Apply -1 phase to |11⟩
        idx_11 = [slice(None)] * n_qubits
        idx_11[q0] = 1
        idx_11[q1] = 1
        new_state[tuple(idx_11)] *= -1

    elif gate == Gate.SWAP:
        # Swap |01⟩ <-> |10⟩
        idx_01 = [slice(None)] * n_qubits
        idx_01[q0] = 0
        idx_01[q1] = 1
        idx_10 = [slice(None)] * n_qubits
        idx_10[q0] = 1
        idx_10[q1] = 0
        new_state[tuple(idx_01)], new_state[tuple(idx_10)] = state[tuple(idx_10)].copy(), state[tuple(idx_01)].copy()

    return new_state.reshape(-1)


# Public API

def simulate(circuit: Circuit) -> np.ndarray:
    """
    Simulate circuit and return final statevector.

    MEASURE operations are skipped (use sample() to get measurement results).
    """
    n = circuit.n_qubits
    state = np.zeros(2**n, dtype=complex) # The amplitude of every possible basis state.
    state[0] = 1.0 # set state into |00...0⟩, 100% probability to that state

    for op in circuit.ops:
        if op.gate == Gate.MEASURE: continue
        if op.gate.n_qubits == 1: state = _apply_single_qubit(state, GATE_1Q[op.gate](op.params), op.qubits[0], n)
        else: state = _apply_two_qubit(state, op.gate, op.qubits[0], op.qubits[1], n)
    return state


def states_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if two states are equal up to global phase.

    Uses fidelity: |⟨a|b⟩| = ||a|| · ||b|| iff states equal up to phase.
    """
    if a.shape != b.shape: return False

    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)

    if norm_a < tol and norm_b < tol: return True
    if norm_a < tol or norm_b < tol:return False

    fidelity = np.abs(np.vdot(a, b)) / (norm_a * norm_b)
    return np.isclose(fidelity, 1.0, atol=tol)


def sample(state: np.ndarray, shots: int, seed: int = None) -> dict[str, int]:
    """
    Sample measurement outcomes from statevector.

    Args:
        state: Statevector to sample from
        shots: Number of measurements
        seed: Random seed for deterministic results

    Returns:
        Dictionary mapping bitstrings to counts, e.g. {'00': 512, '11': 488}
    """
    rng = np.random.default_rng(seed)

    outcomes = rng.choice(len(state), size=shots, p=np.abs(state) ** 2)
    values, counts = np.unique(outcomes, return_counts=True)

    return {format(v, f'0{int(np.log2(len(state)))}b'): int(c) for v, c in zip(values, counts)}
