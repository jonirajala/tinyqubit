"""Feature map circuits — the bridge between classical data and quantum states."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate


# Mottonen state preparation -------

def _ry_angles(sv, n):
    angles = []
    for k in range(n):
        bs, h = 1 << (n - k), 1 << (n - k - 1)
        a = np.zeros(1 << k)
        for j in range(1 << k):
            a[j] = 2 * np.arctan2(np.linalg.norm(sv[j*bs+h:(j+1)*bs]), np.linalg.norm(sv[j*bs:j*bs+h]))
        angles.append(a)
    return angles


def _rz_angles(sv, n):
    phases = np.angle(sv)
    angles = []
    for k in range(n):
        bs, h = 1 << (n - k), 1 << (n - k - 1)
        a = np.zeros(1 << k)
        for j in range(1 << k):
            a[j] = np.mean(phases[j*bs+h:(j+1)*bs]) - np.mean(phases[j*bs:j*bs+h])
        angles.append(a)
    return angles


def _ucr(circuit, gate, angles, target, controls):
    if np.allclose(angles, 0): return
    if not controls:
        gate(target, float(angles[0]))
        return
    half = len(angles) // 2
    even, odd = (angles[:half] + angles[half:]) / 2, (angles[:half] - angles[half:]) / 2
    if np.allclose(odd, 0):
        _ucr(circuit, gate, even, target, controls[1:])
        return
    _ucr(circuit, gate, even, target, controls[1:])
    circuit.cx(controls[0], target)
    _ucr(circuit, gate, odd, target, controls[1:])
    circuit.cx(controls[0], target)


def _mottonen_prep(circuit, sv, wires):
    n = len(wires)
    ry = _ry_angles(sv, n)
    for k in range(n):
        _ucr(circuit, circuit.ry, ry[k], wires[k], wires[:k])
    # NOTE: skip RZ when all amplitudes are non-negative real (no phase correction needed)
    if not np.allclose(sv, np.abs(sv)):
        rz = _rz_angles(sv, n)
        for k in range(n):
            _ucr(circuit, circuit.rz, rz[k], wires[k], wires[:k])


def angle_feature_map(circuit: Circuit, features, wires: list[int], rotation: Gate = Gate.RY) -> None:
    """One rotation gate per wire. Simplest feature map."""
    rot = {Gate.RX: circuit.rx, Gate.RY: circuit.ry, Gate.RZ: circuit.rz}[rotation]
    for w, f in zip(wires, features):
        rot(w, f)


def basis_feature_map(circuit: Circuit, features, wires: list[int]) -> None:
    """X gate where feature bit is 1. Binary input only."""
    for w, f in zip(wires, features):
        if f: circuit.x(w)


def amplitude_feature_map(circuit: Circuit, features, wires: list[int], decompose: bool = False) -> None:
    """Encode features as amplitudes. decompose=True uses Mottonen RY+CX decomposition."""
    n = len(wires)
    if len(features) > 2 ** n:
        raise ValueError(f"Too many features ({len(features)}) for {n} wires (max {2 ** n})")
    if decompose:
        sv = np.zeros(2 ** n, dtype=complex)
        sv[:len(features)] = features
        norm = np.linalg.norm(sv)
        if norm > 1e-15: sv /= norm
        _mottonen_prep(circuit, sv, wires)
        return
    sv = np.zeros(2 ** circuit.n_qubits, dtype=complex)
    if list(wires) == list(range(circuit.n_qubits)):
        sv[:len(features)] = features
    else:
        # NOTE: non-wire qubits stay |0⟩, features fill the subspace spanned by wires
        for i, f in enumerate(features):
            idx = 0
            for bit_pos, w in enumerate(wires):
                if i & (1 << (n - 1 - bit_pos)):
                    idx |= 1 << (circuit.n_qubits - 1 - w)
            sv[idx] = f
    circuit.initialize(sv)


def zz_feature_map(circuit: Circuit, features, wires: list[int], reps: int = 2) -> None:
    """ZZ feature map: H + RZ(x_i) + CZ+RZ(x_i*x_j) per rep."""
    for _ in range(reps):
        for w in wires:
            circuit.h(w)
        for w, f in zip(wires, features):
            circuit.rz(w, f)
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                circuit.cz(wires[i], wires[j])
                circuit.rz(wires[j], features[i] * features[j])



def pauli_feature_map(circuit: Circuit, features, wires: list[int], paulis: str = "Z", reps: int = 2) -> None:
    """Generalized feature map with configurable Pauli rotations."""
    rot_map = {"X": circuit.rx, "Y": circuit.ry, "Z": circuit.rz}
    if len(paulis) == 1: paulis = paulis * len(wires)
    for _ in range(reps):
        for w in wires:
            circuit.h(w)
        for w, f, p in zip(wires, features, paulis):
            rot_map[p](w, f)
        for i in range(len(wires)):
            for j in range(i + 1, len(wires)):
                circuit.cz(wires[i], wires[j])
                circuit.rz(wires[j], features[i] * features[j])
