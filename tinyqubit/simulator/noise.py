"""Noise models for quantum simulation via Monte Carlo trajectories."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from ..ir import Gate, _get_gate_matrix
from .statevector import _apply_single_qubit, _get_1q_idx

NoiseFn = Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray]

def _check(val: float, name: str) -> None:
    if not 0 <= val <= 1: raise ValueError(f"{name} must be in [0,1], got {val}")

_PAULI_XYZ = None
def _get_pauli_xyz():
    global _PAULI_XYZ
    if _PAULI_XYZ is None:
        _PAULI_XYZ = [_get_gate_matrix(Gate.X, ()), _get_gate_matrix(Gate.Y, ()), _get_gate_matrix(Gate.Z, ())]
    return _PAULI_XYZ

def depolarizing(p: float) -> NoiseFn:
    """Apply random Pauli X/Y/Z with probability p."""
    _check(p, "probability")
    def apply(state, qubit, n, rng):
        if rng.random() >= p: return state
        i0, i1 = _get_1q_idx(n, qubit)
        st = state.reshape([2] * n)
        choice = rng.integers(3)
        if choice == 0:  # X: swap
            tmp = st[i0].copy(); st[i0] = st[i1]; st[i1] = tmp
        elif choice == 1:  # Y: swap + phase
            tmp = st[i0].copy(); st[i0] = 1j * st[i1]; st[i1] = -1j * tmp
        else:  # Z: negate |1⟩
            st[i1] *= -1
        return state
    return apply

def amplitude_damping(gamma: float) -> NoiseFn:
    """T1 decay: |1⟩ → |0⟩ with probability gamma. gamma = 1 - exp(-t/T1)"""
    _check(gamma, "gamma")
    _sqrt_1mg = np.sqrt(1 - gamma)
    def apply(state, qubit, n, rng):
        if gamma <= 0: return state
        i0, i1 = _get_1q_idx(n, qubit)
        st = state.reshape([2] * n)
        p1 = np.vdot(st[i1], st[i1]).real
        if rng.random() < p1 * gamma:
            st[i0] = st[i1]; st[i1] = 0.0
            norm = np.sqrt(p1)
        else:
            st[i1] *= _sqrt_1mg
            norm = np.sqrt(1 - p1 * gamma)
        if norm > 1e-10: state /= norm
        return state
    return apply

def phase_damping(lam: float) -> NoiseFn:
    """T2 dephasing: Z-basis measurement with probability lam. lam = 1 - exp(-t/T_phi)"""
    _check(lam, "lambda_")
    def apply(state, qubit, n, rng):
        if lam <= 0 or rng.random() >= lam: return state
        i0, i1 = _get_1q_idx(n, qubit)
        st = state.reshape([2] * n)
        p0 = np.vdot(st[i0], st[i0]).real
        if rng.random() < p0:
            st[i1] = 0.0; norm = np.sqrt(p0)
        else:
            st[i0] = 0.0; norm = np.sqrt(1 - p0)
        if norm > 1e-10: state /= norm
        return state
    return apply

def readout_error(p0_given_1: float = 0.0, p1_given_0: float = 0.0) -> Callable[[int, np.random.Generator], int]:
    """Bit-flip on measurement: p0_given_1 = P(read 0 | true 1), p1_given_0 = P(read 1 | true 0)."""
    _check(p0_given_1, "p0_given_1"); _check(p1_given_0, "p1_given_0")
    def apply(outcome, rng):
        if outcome == 0 and rng.random() < p1_given_0: return 1
        if outcome == 1 and rng.random() < p0_given_1: return 0
        return outcome
    return apply

def _depolarizing_kraus(p: float) -> list[np.ndarray]:
    s0, sp = np.sqrt(1 - p), np.sqrt(p / 3)
    return [s0 * np.eye(2, dtype=complex), sp * _get_gate_matrix(Gate.X, ()),
            sp * _get_gate_matrix(Gate.Y, ()), sp * _get_gate_matrix(Gate.Z, ())]

def _amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]

def _phase_damping_kraus(lam: float) -> list[np.ndarray]:
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]

@dataclass
class NoiseModel:
    """Noise configuration. Use add_* methods to configure, then pass to simulate()."""
    gate_noise: dict[Gate, list[NoiseFn]] = field(default_factory=dict)
    default_noise: list[NoiseFn] = field(default_factory=list)
    readout_error_fn: Callable[[int, np.random.Generator], int] | None = None
    gate_kraus: dict[Gate, list[list[np.ndarray]]] = field(default_factory=dict)
    default_kraus: list[list[np.ndarray]] = field(default_factory=list)

    def _add(self, noise: NoiseFn, gates: list[Gate] | None, kraus: list[np.ndarray] | None = None) -> "NoiseModel":
        if gates is None:
            self.default_noise.append(noise)
            if kraus is not None: self.default_kraus.append(kraus)
        else:
            for g in gates:
                self.gate_noise.setdefault(g, []).append(noise)
                if kraus is not None: self.gate_kraus.setdefault(g, []).append(kraus)
        return self

    def add_depolarizing(self, p: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(depolarizing(p), gates, _depolarizing_kraus(p))

    def add_amplitude_damping(self, gamma: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(amplitude_damping(gamma), gates, _amplitude_damping_kraus(gamma))

    def add_phase_damping(self, lam: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(phase_damping(lam), gates, _phase_damping_kraus(lam))

    def add_readout_error(self, p0_given_1: float = 0.0, p1_given_0: float = 0.0) -> "NoiseModel":
        self.readout_error_fn = readout_error(p0_given_1, p1_given_0)
        return self

def realistic_noise(t1=100e-6, t2=50e-6, gate_time_1q=50e-9, gate_time_2q=300e-9,
                    depolarizing_1q=0.001, depolarizing_2q=0.01, readout_err=0.02) -> NoiseModel:
    """Create noise model with typical superconducting qubit parameters."""
    gamma_1q = 1 - np.exp(-gate_time_1q / t1) if t1 > 0 else 0
    gamma_2q = 1 - np.exp(-gate_time_2q / t1) if t1 > 0 else 0
    t_phi = 1 / (1/t2 - 1/(2*t1)) if t2 > 0 and t1 > 0 and t2 < 2*t1 else float('inf')
    lam_1q = 1 - np.exp(-gate_time_1q / t_phi) if t_phi < float('inf') else 0
    lam_2q = 1 - np.exp(-gate_time_2q / t_phi) if t_phi < float('inf') else 0

    gates_1q = [Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RX, Gate.RY, Gate.RZ]
    gates_2q = [Gate.CX, Gate.CZ, Gate.SWAP, Gate.CP]
    gates_3q = [Gate.CCX, Gate.CCZ]
    noise = NoiseModel()
    for p, g, gates in [(depolarizing_1q, noise.add_depolarizing, gates_1q), (depolarizing_2q, noise.add_depolarizing, gates_2q),
                        (depolarizing_2q, noise.add_depolarizing, gates_3q),
                        (gamma_1q, noise.add_amplitude_damping, gates_1q), (gamma_2q, noise.add_amplitude_damping, gates_2q),
                        (gamma_2q, noise.add_amplitude_damping, gates_3q),
                        (lam_1q, noise.add_phase_damping, gates_1q), (lam_2q, noise.add_phase_damping, gates_2q),
                        (lam_2q, noise.add_phase_damping, gates_3q)]:
        if p > 0: g(p, gates)
    if readout_err > 0: noise.add_readout_error(readout_err, readout_err)
    return noise
