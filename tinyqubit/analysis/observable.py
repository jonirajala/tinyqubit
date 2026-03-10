"""Pauli observables, expectation values, and quantum state analysis."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate as _Gate, _GATE_1Q_CACHE
from ..simulator import simulate, _apply_single_qubit


class Observable:
    """Sum of weighted Pauli terms: O = Σ cᵢ · Pᵢ"""
    __slots__ = ('terms',)

    def __init__(self, terms: list[tuple[complex, dict[int, str]]]):
        self.terms = terms

    def __add__(self, other: Observable) -> Observable:
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


def expectation(circuit: Circuit, observable: Observable) -> float:
    """Compute ⟨ψ|O|ψ⟩ for a Pauli observable."""
    state, _ = simulate(circuit)
    n = circuit.n_qubits
    result = 0.0
    for coeff, paulis in observable.terms:
        psi = state.copy()
        for qubit, pauli in paulis.items():
            psi = _apply_single_qubit(psi, _PAULI_MATRIX[pauli], qubit, n)
        result += coeff * np.vdot(state, psi)
    return result.real


def expectation_batch(circuits: list[Circuit], observable: Observable) -> np.ndarray:
    """Compute ⟨ψ|O|ψ⟩ for each circuit in a list."""
    return np.array([expectation(c, observable) for c in circuits])


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
