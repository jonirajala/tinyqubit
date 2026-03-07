"""Density matrix simulator."""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from ..ir import Circuit, Gate, _has_parameter, _GATE_1Q_CACHE
from .statevector import _build_gate_unitary

if TYPE_CHECKING:
    from .noise import NoiseModel


def _dm_apply_unitary(rho: np.ndarray, U: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply ρ' = U ρ U† via tensor contraction on [2]*2n shaped rho."""
    k = len(qubits)
    Ur = U.reshape([2] * (2 * k))
    # U on ket side
    rho = np.tensordot(Ur, rho, axes=(list(range(k, 2 * k)), list(qubits)))
    rho = np.moveaxis(rho, list(range(k)), list(qubits))
    # U† on bra side: Σ_b ρ[...,b] * conj(U[j,b]) — contract bra with U's col indices
    bra_axes = [n + q for q in qubits]
    rho = np.tensordot(rho, Ur.conj(), axes=(bra_axes, list(range(k, 2 * k))))
    rho = np.moveaxis(rho, list(range(2 * n - k, 2 * n)), bra_axes)
    return rho

def _dm_apply_noise(rho: np.ndarray, op, noise_model, n: int) -> np.ndarray:
    if noise_model is None: return rho
    kraus_list = noise_model.gate_kraus.get(op.gate, noise_model.default_kraus)
    if not kraus_list: return rho
    for kraus_ops in kraus_list:
        for q in op.qubits:
            rho_new = np.zeros_like(rho)
            for E in kraus_ops: rho_new += _dm_apply_unitary(rho, E, (q,), n)
            rho = rho_new
    return rho

def _dm_measure(rho: np.ndarray, qubit: int, n: int, rng) -> tuple[np.ndarray, int]:
    probs = np.zeros(2)
    for b in range(2):
        idx = [slice(None)] * (2 * n)
        idx[qubit] = b; idx[n + qubit] = b
        sub = rho[tuple(idx)]
        for k in range(n - 1): sub = np.trace(sub, axis1=0, axis2=n - 1 - k)
        probs[b] = sub.real
    outcome = 1 if rng.random() < probs[1] else 0
    rho = rho.copy()
    for ax in [qubit, n + qubit]:
        idx = [slice(None)] * (2 * n)
        idx[ax] = 1 - outcome
        rho[tuple(idx)] = 0.0
    if probs[outcome] > 1e-10: rho /= probs[outcome]
    return rho, outcome

def _dm_reset(rho: np.ndarray, qubit: int, n: int, rng) -> np.ndarray:
    rho, outcome = _dm_measure(rho, qubit, n, rng)
    return _dm_apply_unitary(rho, _GATE_1Q_CACHE[Gate.X], (qubit,), n) if outcome == 1 else rho

def simulate_density(circuit: Circuit, noise_model: "NoiseModel | None" = None,
                     seed: int | None = None) -> tuple[np.ndarray, dict[int, int]]:
    """Density matrix simulation. Returns (rho, classical_bits) where rho is 2D."""
    n = circuit.n_qubits
    for op in circuit.ops:
        if _has_parameter(op.params):
            raise TypeError(f"Cannot simulate: {op.gate.name} has unbound Parameter. Call circuit.bind() first.")
        for q in op.qubits:
            if not (0 <= q < n):
                raise ValueError(f"Invalid qubit index {q} for {n}-qubit circuit in {op.gate.name}")

    rng = np.random.default_rng(seed)
    classical = {i: 0 for i in range(circuit.n_classical)}

    if circuit._initial_state is not None:
        psi = circuit._initial_state.copy()
        rho = np.outer(psi, psi.conj()).reshape([2] * (2 * n))
    else:
        rho = np.zeros([2] * (2 * n), dtype=complex)
        rho[tuple([0] * (2 * n))] = 1.0

    for op in circuit.ops:
        if op.condition is not None and classical.get(op.condition[0]) != op.condition[1]: continue
        if op.gate == Gate.MEASURE:
            rho, outcome = _dm_measure(rho, op.qubits[0], n, rng)
            if op.classical_bit is not None:
                if noise_model is not None and noise_model.readout_error_fn is not None:
                    outcome = noise_model.readout_error_fn(outcome, rng)
                classical[op.classical_bit] = outcome
        elif op.gate == Gate.RESET:
            rho = _dm_reset(rho, op.qubits[0], n, rng)
        else:
            U = _build_gate_unitary(op)
            rho = _dm_apply_unitary(rho, U, op.qubits, n)
            rho = _dm_apply_noise(rho, op, noise_model, n)

    rho_mat = rho.reshape(2**n, 2**n)
    assert abs(np.trace(rho_mat).real - 1.0) < 1e-10, "density matrix trace drifted"
    return rho_mat, classical
