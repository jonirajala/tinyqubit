"""
Noise models for quantum simulation via Monte Carlo trajectories.

Channels: depolarizing, amplitude_damping, phase_damping, readout_error
Container: NoiseModel with add_* methods
Factory: realistic_noise() for typical hardware parameters
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from .ir import Gate
from .simulator import _get_gate_matrix, _apply_single_qubit

NoiseFn = Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray]

def _check(val: float, name: str) -> None:
    if not 0 <= val <= 1: raise ValueError(f"{name} must be in [0,1], got {val}")

def depolarizing(p: float) -> NoiseFn:
    """Apply random Pauli X/Y/Z with probability p."""
    _check(p, "probability")
    def apply(state, qubit, n, rng):
        if rng.random() < p:
            state = _apply_single_qubit(state, _get_gate_matrix(rng.choice([Gate.X, Gate.Y, Gate.Z]), ()), qubit, n)
        return state
    return apply

def amplitude_damping(gamma: float) -> NoiseFn:
    """T1 decay: |1⟩ → |0⟩ with probability gamma. gamma = 1 - exp(-t/T1)"""
    _check(gamma, "gamma")
    def apply(state, qubit, n, rng):
        if gamma <= 0: return state
        state = state.reshape([2] * n)
        idx = [slice(None)] * n
        idx[qubit] = 1
        idx1 = tuple(idx)
        idx[qubit] = 0
        idx0 = tuple(idx)
        p_jump = np.sum(np.abs(state[idx1]) ** 2) * gamma
        if rng.random() < p_jump:
            state[idx0], state[idx1] = state[idx1].copy(), 0.0
        else:
            state[idx1] *= np.sqrt(1 - gamma)
        norm = np.linalg.norm(state)
        return (state / norm if norm > 1e-10 else state).reshape(-1)
    return apply

def phase_damping(lam: float) -> NoiseFn:
    """T2 dephasing: Z-basis measurement with probability lam. lam = 1 - exp(-t/T_phi)"""
    _check(lam, "lambda_")
    def apply(state, qubit, n, rng):
        if lam <= 0 or rng.random() >= lam: return state
        state = state.reshape([2] * n)
        idx = [slice(None)] * n
        idx[qubit] = 0
        idx0 = tuple(idx)
        idx[qubit] = 1
        idx1 = tuple(idx)
        if rng.random() < np.sum(np.abs(state[idx0]) ** 2):
            state[idx1] = 0.0
        else:
            state[idx0] = 0.0
        norm = np.linalg.norm(state)
        return (state / norm if norm > 1e-10 else state).reshape(-1)
    return apply

def readout_error(p0_given_1: float = 0.0, p1_given_0: float = 0.0) -> Callable[[int, np.random.Generator], int]:
    """Bit-flip on measurement: p0_given_1 = P(read 0 | true 1), p1_given_0 = P(read 1 | true 0)."""
    _check(p0_given_1, "p0_given_1"); _check(p1_given_0, "p1_given_0")
    def apply(outcome, rng):
        if outcome == 0 and rng.random() < p1_given_0: return 1
        if outcome == 1 and rng.random() < p0_given_1: return 0
        return outcome
    return apply

@dataclass
class NoiseModel:
    """Noise configuration. Use add_* methods to configure, then pass to simulate()."""
    gate_noise: dict[Gate, list[NoiseFn]] = field(default_factory=dict)
    default_noise: list[NoiseFn] = field(default_factory=list)
    readout_error_fn: Callable[[int, np.random.Generator], int] | None = None

    def _add(self, noise: NoiseFn, gates: list[Gate] | None) -> "NoiseModel":
        if gates is None: self.default_noise.append(noise)
        else:
            for g in gates: self.gate_noise.setdefault(g, []).append(noise)
        return self

    def add_depolarizing(self, p: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(depolarizing(p), gates)

    def add_amplitude_damping(self, gamma: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(amplitude_damping(gamma), gates)

    def add_phase_damping(self, lam: float, gates: list[Gate] | None = None) -> "NoiseModel":
        return self._add(phase_damping(lam), gates)

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
