"""Error mitigation: ZNE (zero noise extrapolation) and readout mitigation."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .simulator import simulate, simulate_density, _apply_single_qubit
from .observable import Observable, _PAULI_MATRIX
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .noise import NoiseModel


def _fold_circuit(circuit: Circuit, scale: int) -> Circuit:
    """Fold circuit: C → C (C†C)^((scale-1)/2) for odd scale."""
    if scale == 1: return circuit
    folded = Circuit(circuit.n_qubits, circuit.n_classical)
    folded.ops = list(circuit.ops)
    inv_ops = circuit.inverse().ops
    for _ in range((scale - 1) // 2):
        folded.ops.extend(inv_ops)
        folded.ops.extend(circuit.ops)
    return folded


def _noisy_exp(circuit: Circuit, observable: Observable, noise_model: NoiseModel,
               shots: int | None, rng: np.random.Generator) -> float:
    n = circuit.n_qubits
    if shots is None:
        # Density matrix path: Tr(ρ·O) via tensordot on ket indices
        rho, _ = simulate_density(circuit, noise_model=noise_model)
        rho_t = rho.reshape([2] * (2 * n))
        result, trace_idx = 0.0, list(range(n)) + list(range(n))
        for coeff, paulis in observable.terms:
            tmp = rho_t
            for qubit, pauli in paulis.items():
                tmp = np.tensordot(_PAULI_MATRIX[pauli], tmp, axes=([1], [qubit]))
                tmp = np.moveaxis(tmp, 0, qubit)
            result += coeff * np.einsum(tmp, trace_idx).real
        return result
    # Trajectory simulation: average ⟨ψ|O|ψ⟩ over shots
    total = 0.0
    for _ in range(shots):
        state, _ = simulate(circuit, seed=int(rng.integers(2**32)), noise_model=noise_model)
        for coeff, paulis in observable.terms:
            psi = state.copy()
            for qubit, pauli in paulis.items():
                psi = _apply_single_qubit(psi, _PAULI_MATRIX[pauli], qubit, n)
            total += coeff * np.vdot(state, psi).real
    return total / shots


def zne(circuit: Circuit, observable: Observable, noise_model: "NoiseModel",
        scale_factors: list[int] | None = None, shots: int | None = None,
        seed: int | None = None) -> float:
    """Zero noise extrapolation via circuit folding + Richardson extrapolation."""
    if scale_factors is None: scale_factors = [1, 3, 5]
    rng = np.random.default_rng(seed)
    values = [_noisy_exp(_fold_circuit(circuit, s), observable, noise_model, shots, rng)
              for s in scale_factors]
    coeffs = np.polyfit(scale_factors, values, len(scale_factors) - 1)
    return float(np.polyval(coeffs, 0))


def calibration_matrix(n_qubits: int, noise_model: "NoiseModel",
                       shots: int = 1000, seed: int | None = None) -> np.ndarray:
    """Build readout calibration matrix: cal[i,j] = P(measure j | prepared i)."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    cal = np.zeros((dim, dim))
    for i in range(dim):
        c = Circuit(n_qubits)
        for q in range(n_qubits):
            if (i >> (n_qubits - 1 - q)) & 1: c.x(q)
        for q in range(n_qubits): c.measure(q, q)
        for _ in range(shots):
            _, cl = simulate(c, seed=int(rng.integers(2**32)), noise_model=noise_model)
            j = sum(cl[q] << (n_qubits - 1 - q) for q in range(n_qubits))
            cal[i, j] += 1
    return cal / shots


def mitigate_readout(noisy_counts: dict[str, int], cal_matrix: np.ndarray) -> dict[str, float]:
    """Correct measurement counts using calibration matrix via least-squares."""
    n = int(np.log2(cal_matrix.shape[0]))
    dim = cal_matrix.shape[0]
    total = sum(noisy_counts.values())
    p_noisy = np.zeros(dim)
    for bs, count in noisy_counts.items():
        p_noisy[int(bs, 2)] = count / total
    p_ideal, *_ = np.linalg.lstsq(cal_matrix.T, p_noisy, rcond=None)
    p_ideal = np.clip(p_ideal, 0, None)
    p_ideal /= p_ideal.sum()
    return {format(i, f'0{n}b'): float(p) for i, p in enumerate(p_ideal) if p > 1e-10}
