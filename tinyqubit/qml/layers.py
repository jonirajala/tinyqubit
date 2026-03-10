"""Composable circuit building blocks: feature maps and ansatze."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate, Parameter


def _parse_qubits(qubits) -> list[int]:
    return list(range(qubits)) if isinstance(qubits, int) else list(qubits)


def _auto_params(qubits: list[int], features) -> list:
    if features is None:
        return [Parameter(f"x{i}", trainable=False) for i in range(len(qubits))]
    return features


# Feature maps -------

def angle_feature_map(qubits, features=None, rotation: Gate = Gate.RY) -> Circuit:
    """One rotation gate per qubit. Simplest feature map."""
    qubits = _parse_qubits(qubits)
    features = _auto_params(qubits, features)
    c = Circuit(max(qubits) + 1)
    rot = {Gate.RX: c.rx, Gate.RY: c.ry, Gate.RZ: c.rz}[rotation]
    for q, f in zip(qubits, features):
        rot(q, f)
    return c


def basis_feature_map(qubits, features) -> Circuit:
    """X gate where feature bit is 1. Binary input only."""
    qubits = _parse_qubits(qubits)
    c = Circuit(max(qubits) + 1)
    for q, f in zip(qubits, features):
        if f: c.x(q)
    return c


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


def _mottonen_prep(circuit, sv, qubits):
    n = len(qubits)
    ry = _ry_angles(sv, n)
    for k in range(n):
        _ucr(circuit, circuit.ry, ry[k], qubits[k], qubits[:k])
    # NOTE: skip RZ when all amplitudes are non-negative real (no phase correction needed)
    if not np.allclose(sv, np.abs(sv)):
        rz = _rz_angles(sv, n)
        for k in range(n):
            _ucr(circuit, circuit.rz, rz[k], qubits[k], qubits[:k])


def amplitude_feature_map(qubits, features, decompose: bool = False) -> Circuit:
    """Encode features as amplitudes. decompose=True uses Mottonen RY+CX decomposition."""
    qubits = _parse_qubits(qubits)
    n = len(qubits)
    c = Circuit(max(qubits) + 1)
    if len(features) > 2 ** n:
        raise ValueError(f"Too many features ({len(features)}) for {n} qubits (max {2 ** n})")
    if decompose:
        sv = np.zeros(2 ** n, dtype=complex)
        sv[:len(features)] = features
        norm = np.linalg.norm(sv)
        if norm > 1e-15: sv /= norm
        _mottonen_prep(c, sv, qubits)
        return c
    sv = np.zeros(2 ** c.n_qubits, dtype=complex)
    if list(qubits) == list(range(c.n_qubits)):
        sv[:len(features)] = features
    else:
        # NOTE: non-qubit positions stay |0⟩, features fill the subspace spanned by qubits
        for i, f in enumerate(features):
            idx = 0
            for bit_pos, q in enumerate(qubits):
                if i & (1 << (n - 1 - bit_pos)):
                    idx |= 1 << (c.n_qubits - 1 - q)
            sv[idx] = f
    c.initialize(sv)
    return c


def zz_feature_map(qubits, features=None, reps: int = 2) -> Circuit:
    """ZZ feature map: H + RZ(x_i) + CZ+RZ(x_i*x_j) per rep."""
    qubits = _parse_qubits(qubits)
    features = _auto_params(qubits, features)
    c = Circuit(max(qubits) + 1)
    for _ in range(reps):
        for q in qubits:
            c.h(q)
        for q, f in zip(qubits, features):
            c.rz(q, f)
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                c.cz(qubits[i], qubits[j])
                c.rz(qubits[j], features[i] * features[j])
    return c


def pauli_feature_map(qubits, features=None, paulis: str = "Z", reps: int = 2) -> Circuit:
    """Generalized feature map with configurable Pauli rotations."""
    qubits = _parse_qubits(qubits)
    features = _auto_params(qubits, features)
    c = Circuit(max(qubits) + 1)
    rot_map = {"X": c.rx, "Y": c.ry, "Z": c.rz}
    if len(paulis) == 1: paulis = paulis * len(qubits)
    for _ in range(reps):
        for q in qubits:
            c.h(q)
        for q, f, p in zip(qubits, features, paulis):
            rot_map[p](q, f)
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                c.cz(qubits[i], qubits[j])
                c.rz(qubits[j], features[i] * features[j])
    return c


# Ansatze -------

def strongly_entangling_layers(qubits, n_layers: int, prefix: str = "sel") -> Circuit:
    """RY + RZ per qubit, CX with layer-dependent offset for full connectivity."""
    qubits = _parse_qubits(qubits)
    n = len(qubits)
    c = Circuit(max(qubits) + 1)
    for l in range(n_layers):
        for i, q in enumerate(qubits):
            c.ry(q, Parameter(f"{prefix}_{l}_{i}_y"))
            c.rz(q, Parameter(f"{prefix}_{l}_{i}_z"))
        for i in range(n):
            t = (i + l + 1) % n
            if t != i:
                c.cx(qubits[i], qubits[t])
    return c


def basic_entangler_layers(qubits, n_layers: int, prefix: str = "bel") -> Circuit:
    """RY only, linear CX ladder."""
    qubits = _parse_qubits(qubits)
    c = Circuit(max(qubits) + 1)
    for l in range(n_layers):
        for i, q in enumerate(qubits):
            c.ry(q, Parameter(f"{prefix}_{l}_{i}"))
        for i in range(len(qubits) - 1):
            c.cx(qubits[i], qubits[i + 1])
    return c


def hardware_efficient_ansatz(n_qubits: int, depth: int, circular: bool = False) -> Circuit:
    """Hardware-efficient ansatz: RY+RZ layers with linear (or circular) CX entanglement."""
    c = Circuit(n_qubits)
    for l in range(depth):
        for q in range(n_qubits):
            c.ry(q, Parameter(f"hea_{l}_{q}_y"))
            c.rz(q, Parameter(f"hea_{l}_{q}_z"))
        n_cx = n_qubits if circular else n_qubits - 1
        for q in range(n_cx):
            c.cx(q, (q + 1) % n_qubits)
    return c
