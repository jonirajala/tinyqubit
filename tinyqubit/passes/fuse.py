"""Gate fusion: merge consecutive 1Q gates into minimal RZ-RX-RZ rotations."""
from __future__ import annotations
import numpy as np
from math import pi, atan2, sqrt
from collections import defaultdict
from ..ir import Circuit, Operation, Gate
from ..simulator import _get_gate_matrix


def _is_identity(U: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if U is identity up to global phase."""
    # Factor out global phase: det(U) = e^{2iφ}, so φ = angle(det)/2
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    phase = np.exp(-1j * np.angle(det) / 2)
    U_su2 = U * phase
    return np.allclose(U_su2, np.eye(2), atol=tol) or np.allclose(U_su2, -np.eye(2), atol=tol)


def _decompose_zxz(U: np.ndarray, qubit: int, tol: float = 1e-9) -> list[Operation]:
    """Decompose 2x2 unitary to RZ(α) RX(β) RZ(γ). Returns [] if identity.

    Uses ZYZ Euler decomposition then converts to ZXZ.
    For U ∈ SU(2): U = RZ(α) RY(β) RZ(γ) where:
      β = 2 * atan2(|U[1,0]|, |U[0,0]|)
      α = angle(U[1,0]) - angle(U[0,0]) + π/2  (when β ≠ 0)
      γ = angle(U[1,0]) + angle(U[0,0]) - π/2  (when β ≠ 0)
    Then ZYZ → ZXZ: RZ(α) RY(β) RZ(γ) = RZ(α - π/2) RX(β) RZ(γ + π/2)
    """
    if _is_identity(U, tol):
        return []

    # Convert to SU(2) by removing global phase
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    phase = np.exp(-1j * np.angle(det) / 2)
    U = U * phase

    # Extract ZYZ angles
    # U = [[cos(β/2) e^{-i(α+γ)/2}, -sin(β/2) e^{-i(α-γ)/2}],
    #      [sin(β/2) e^{ i(α-γ)/2},  cos(β/2) e^{ i(α+γ)/2}]]
    cos_b2 = np.clip(np.abs(U[0, 0]), 0, 1)
    sin_b2 = np.clip(np.abs(U[1, 0]), 0, 1)
    beta = 2 * atan2(sin_b2, cos_b2)

    if sin_b2 < tol:
        # β ≈ 0: pure Z rotation, U ≈ diag(e^{-i(α+γ)/2}, e^{i(α+γ)/2})
        alpha_plus_gamma = -2 * np.angle(U[0, 0])
        alpha, gamma = alpha_plus_gamma / 2, alpha_plus_gamma / 2
    elif cos_b2 < tol:
        # β ≈ π: U[0,0] ≈ 0, use U[1,0] and U[0,1]
        alpha_minus_gamma = 2 * np.angle(U[1, 0])
        alpha, gamma = alpha_minus_gamma / 2, -alpha_minus_gamma / 2
    else:
        # General case
        alpha_plus_gamma = -2 * np.angle(U[0, 0])
        alpha_minus_gamma = 2 * np.angle(U[1, 0])
        alpha = (alpha_plus_gamma + alpha_minus_gamma) / 2
        gamma = (alpha_plus_gamma - alpha_minus_gamma) / 2

    # ZYZ → ZXZ conversion: RY(β) = RZ(π/2) RX(β) RZ(-π/2)
    # So: RZ(α) RY(β) RZ(γ) = RZ(α + π/2) RX(β) RZ(γ - π/2)
    alpha_x = alpha + pi / 2
    gamma_x = gamma - pi / 2

    # Normalize to [-π, π]
    def norm(a):
        return (a + pi) % (2 * pi) - pi

    alpha_x, beta, gamma_x = norm(alpha_x), norm(beta), norm(gamma_x)

    # Build ops in CIRCUIT order (reverse of matrix multiplication order)
    # U = RZ(α_x) @ RX(β) @ RZ(γ_x) means RZ(γ_x) applied first to state
    # So circuit order is: [RZ(γ_x), RX(β), RZ(α_x)]
    ops = []
    if abs(gamma_x) > tol:
        ops.append(Operation(Gate.RZ, (qubit,), (gamma_x,)))
    if abs(beta) > tol:
        ops.append(Operation(Gate.RX, (qubit,), (beta,)))
    if abs(alpha_x) > tol:
        ops.append(Operation(Gate.RZ, (qubit,), (alpha_x,)))

    return ops

_ROTATION_GATES = {Gate.RX, Gate.RY, Gate.RZ}


def _should_fuse(ops: list[Operation]) -> bool:
    """Decide if sequence should be fused. Fuse if:
    - Contains non-rotation gates (H, S, T, X, Y, Z, etc.) that may simplify
    - Has 4+ gates (likely to reduce)
    - Has 2-3 gates with at least one non-rotation
    """
    if len(ops) >= 4:
        return True
    has_non_rotation = any(op.gate not in _ROTATION_GATES for op in ops)
    return has_non_rotation and len(ops) >= 2


def fuse_1q_gates(circuit: Circuit) -> Circuit:
    """Merge consecutive 1Q gates on same qubit into ≤3 rotations."""
    pending: dict[int, list[Operation]] = defaultdict(list)
    result: list[Operation] = []

    def flush(q: int):
        if not pending[q]:
            return
        if not _should_fuse(pending[q]):
            result.extend(pending[q])
        else:
            U = np.eye(2, dtype=complex)
            for op in pending[q]:
                U = _get_gate_matrix(op.gate, op.params) @ U
            result.extend(_decompose_zxz(U, q))
        pending[q] = []

    for op in circuit.ops:
        if op.condition is not None:
            for q in op.qubits: flush(q)
            result.append(op)
        elif op.gate in (Gate.MEASURE, Gate.RESET):
            flush(op.qubits[0])
            result.append(op)
        elif op.gate.n_qubits == 2:
            flush(op.qubits[0]); flush(op.qubits[1])
            result.append(op)
        else:
            pending[op.qubits[0]].append(op)

    for q in list(pending.keys()): flush(q)
    out = Circuit(circuit.n_qubits, circuit.n_classical)
    out.ops = result

    return out

def fuse_2q_blocks(circuit: Circuit) -> Circuit:
    """Fuse consecutive gates on same qubit pair into single 4x4 unitary. (Placeholder - needs KAK decomposition)"""
    return circuit
