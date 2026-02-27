"""Pauli observables and expectation values."""
from __future__ import annotations
import numpy as np
from .ir import Circuit, Gate as _Gate
from .simulator import simulate, _apply_single_qubit, _GATE_1Q_CACHE


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
