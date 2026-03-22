"""Pauli observables, expectation values, and quantum state analysis."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate as _Gate, _GATE_1Q_CACHE
from ..simulator import simulate, _apply_single_qubit
from ..simulator.mps import MPSState, mps_expectation


class Observable:
    """Quantum observable: sum of Pauli terms (O = Σ cᵢ · Pᵢ) or a Hermitian matrix."""
    __slots__ = ('terms', '_matrix')

    def __init__(self, terms_or_matrix):
        if isinstance(terms_or_matrix, np.ndarray):
            self.terms = []
            self._matrix = np.asarray(terms_or_matrix, dtype=complex)
        else:
            self.terms = terms_or_matrix
            self._matrix = None

    def __add__(self, other: Observable) -> Observable:
        if self._matrix is not None or other._matrix is not None:
            raise TypeError("Cannot add matrix observables with +; combine matrices directly")
        return Observable(self.terms + other.terms)

    def __radd__(self, other) -> Observable:
        if other == 0: return self
        return NotImplemented

    def __sub__(self, other: Observable) -> Observable:
        return self + (-other)

    def __neg__(self) -> Observable:
        return Observable([(-c, p) for c, p in self.terms])

    def __mul__(self, scalar) -> Observable:
        return Observable([(c * scalar, p) for c, p in self.terms])

    def __rmul__(self, scalar) -> Observable:
        return self.__mul__(scalar)

    def __matmul__(self, other: Observable) -> Observable:
        terms = []
        for c1, p1 in self.terms:
            for c2, p2 in other.terms:
                if p1.keys() & p2.keys():
                    raise ValueError(f"Overlapping qubits in tensor product: {p1.keys() & p2.keys()}")
                terms.append((c1 * c2, {**p1, **p2}))
        return Observable(terms)

    def __repr__(self) -> str:
        if not self.terms: return "0"
        parts = []
        for c, paulis in self.terms:
            pauli_str = " @ ".join(f"{p}({q})" for q, p in sorted(paulis.items())) or "I"
            if c == 1: parts.append(pauli_str)
            elif c == -1: parts.append(f"-{pauli_str}")
            else: parts.append(f"{c} * {pauli_str}")
        return " + ".join(parts)


def I() -> Observable: return Observable([(1.0, {})])
def X(qubit: int) -> Observable: return Observable([(1.0, {qubit: 'X'})])
def Y(qubit: int) -> Observable: return Observable([(1.0, {qubit: 'Y'})])
def Z(qubit: int) -> Observable: return Observable([(1.0, {qubit: 'Z'})])

_PAULI_MATRIX = {'X': _GATE_1Q_CACHE[_Gate.X], 'Y': _GATE_1Q_CACHE[_Gate.Y], 'Z': _GATE_1Q_CACHE[_Gate.Z]}


def expectation(circuit_or_state, observable: Observable, n_qubits: int | None = None) -> float:
    """Compute ⟨ψ|O|ψ⟩ for a Pauli observable. Accepts Circuit, statevector, or MPSState."""
    if isinstance(circuit_or_state, MPSState):
        return mps_expectation(circuit_or_state.tensors, observable)
    if isinstance(circuit_or_state, np.ndarray):
        state, n = circuit_or_state, n_qubits if n_qubits is not None else int(np.log2(len(circuit_or_state)))
    else:
        circuit = circuit_or_state
        if circuit.backend is not None:
            return circuit.backend(circuit, observable)
        if circuit.is_parameterized and circuit.param_values:
            circuit = circuit.bind()
        state, _ = simulate(circuit)
        if isinstance(state, MPSState):
            return mps_expectation(state.tensors, observable)
        n = circuit.n_qubits
    if observable._matrix is not None:
        return np.vdot(state, observable._matrix @ state).real
    result = 0.0
    probs = None  # lazy: only computed if Z-only terms exist
    for coeff, paulis in observable.terms:
        if not paulis:
            result += coeff; continue
        # Z-only fast path: ⟨Z⟩ = sum(|ψ_i|² × sign_i), no state copy needed
        if all(p == 'Z' for p in paulis.values()):
            if probs is None: probs = np.abs(state) ** 2
            mask = sum(1 << (n - 1 - q) for q in paulis)
            idx = np.arange(len(state), dtype=np.int32)
            v = idx & mask
            v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
            result += coeff * float(np.sum(probs * (1 - 2 * (v & 1))))
        else:
            psi = state.copy()
            for qubit, pauli in paulis.items():
                psi = _apply_single_qubit(psi, _PAULI_MATRIX[pauli], qubit, n)
            result += coeff * np.vdot(state, psi)
    return result.real


def expectation_batch(circuits: list[Circuit], observable: Observable) -> np.ndarray:
    """Compute ⟨ψ|O|ψ⟩ for each circuit in a list."""
    return np.array([expectation(c, observable) for c in circuits])


def expectation_z(state, qubits: list[int] | None = None) -> np.ndarray:
    """Compute ⟨Z_q⟩ for each qubit from a statevector or MPSState."""
    if isinstance(state, MPSState):
        if qubits is None: qubits = list(range(state.n_qubits))
        return np.array([mps_expectation(state.tensors, Observable([(1.0, {q: 'Z'})])) for q in qubits])
    n = int(np.log2(len(state)))
    if qubits is None: qubits = list(range(n))
    probs = np.abs(state) ** 2
    return np.array([float(np.sum(probs * np.array([1 - 2 * ((i >> (n - 1 - q)) & 1) for i in range(len(probs))])))
                     for q in qubits])


def expectation_sweep(circuit: Circuit, param_name: str, values: np.ndarray,
                      observable: Observable, base_values: dict[str, float] | None = None) -> np.ndarray:
    """Sweep one parameter and compute ⟨ψ(θ)|O|ψ(θ)⟩ for each value."""
    base = base_values or {}
    work = circuit.bind({})
    result = np.empty(len(values))
    for i, v in enumerate(values):
        work.bind_params({**base, param_name: v})
        result[i] = expectation(work, observable)
    return result


# State analysis -------

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


def partial_trace(statevector: np.ndarray, keep: list[int]) -> np.ndarray:
    n = int(np.log2(len(statevector)))
    keep_sorted = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep_sorted]
    psi = statevector.reshape([2] * n)
    psi = np.transpose(psi, keep_sorted + trace_out)
    d_keep = 2 ** len(keep_sorted)
    psi = psi.reshape(d_keep, -1)
    return psi @ psi.conj().T


def _von_neumann_entropy(rho: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def entanglement_entropy(statevector: np.ndarray, partition: list[int]) -> float:
    return _von_neumann_entropy(partial_trace(statevector, partition))


def concurrence(statevector: np.ndarray) -> float:
    if len(statevector) != 4:
        raise ValueError("concurrence requires a 2-qubit state")
    a, b, c, d = statevector
    return float(2 * np.abs(a * d - b * c))


def mutual_information(statevector: np.ndarray, partition_a: list[int], partition_b: list[int]) -> float:
    sa = entanglement_entropy(statevector, partition_a)
    sb = entanglement_entropy(statevector, partition_b)
    sab = entanglement_entropy(statevector, partition_a + partition_b)
    return sa + sb - sab
