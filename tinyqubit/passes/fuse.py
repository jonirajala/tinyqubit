"""Gate fusion: merge consecutive 1Q/2Q gates into minimal rotations."""
from __future__ import annotations
import numpy as np
from math import pi, atan2, sqrt
from collections import defaultdict
from ..ir import Circuit, Operation, Gate, _has_parameter
from ..dag import DAGCircuit
from ..simulator import _get_gate_matrix
from ._kak import kak_decompose, cx_count, _extract_su2_pair


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


def _fuse_ops(ops: list[Operation], n_qubits: int) -> list[Operation]:
    """Core fusion logic on a flat op list."""
    pending: dict[int, list[Operation]] = defaultdict(list)
    result: list[Operation] = []

    def flush(q: int):
        if not pending[q]:
            return
        if any(_has_parameter(op.params) for op in pending[q]):
            result.extend(pending[q])
            pending[q] = []
            return
        U = np.eye(2, dtype=complex)
        for op in pending[q]:
            U = _get_gate_matrix(op.gate, op.params) @ U
        fused = _decompose_zxz(U, q)
        result.extend(fused if len(fused) < len(pending[q]) else pending[q])
        pending[q] = []

    for op in ops:
        if op.condition is not None:
            for q in op.qubits: flush(q)
            result.append(op)
        elif op.gate in (Gate.MEASURE, Gate.RESET):
            flush(op.qubits[0])
            result.append(op)
        elif op.gate.n_qubits >= 2:
            for q in op.qubits: flush(q)
            result.append(op)
        else:
            pending[op.qubits[0]].append(op)

    for q in list(pending.keys()): flush(q)
    return result


def fuse_1q_gates(inp):
    """Merge consecutive 1Q gates on same qubit into ≤3 rotations. Accepts Circuit or DAGCircuit."""
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp
    result_ops = _fuse_ops(list(dag.topological_ops()), dag.n_qubits)
    out = DAGCircuit(dag.n_qubits, dag.n_classical)
    for op in result_ops: out.add_op(op)
    return out.to_circuit() if from_circuit else out


# 2Q block fusion (KAK) -------

_I2 = np.eye(2, dtype=complex)
_CX_MATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CX_REV_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
_CZ_MATRIX = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _cp_matrix(theta: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def _gate_to_4x4(op: Operation, q0: int, q1: int) -> np.ndarray:
    """Convert gate to 4x4 unitary on (q0, q1). q0=MSB (left in kron)."""
    if op.gate.n_qubits == 1:
        M = _get_gate_matrix(op.gate, op.params)
        return np.kron(M, _I2) if op.qubits[0] == q0 else np.kron(_I2, M)
    if op.gate == Gate.CX:
        return _CX_MATRIX if op.qubits[0] == q0 else _CX_REV_MATRIX
    if op.gate == Gate.CZ:
        return _CZ_MATRIX
    if op.gate == Gate.SWAP:
        return _SWAP_MATRIX
    # CP
    return _cp_matrix(op.params[0])


def _synthesize_2q(A0, A1, xx, yy, zz, B0, B1, q0: int, q1: int, tol: float = 1e-9) -> list[Operation]:
    """Synthesize KAK decomposition into 3 CX + 1Q gates. A1/B1→q0(MSB), A0/B0→q1(LSB).

    Uses raw (non-canonical) interaction angles directly in the 3-CX Cirq formula.
    """
    ops: list[Operation] = []
    ops.extend(_decompose_zxz(B1, q0))
    ops.extend(_decompose_zxz(B0, q1))

    a = xx * (-2 / pi) + 0.5
    b = yy * (-2 / pi) + 0.5
    c = zz * (-2 / pi) + 0.5

    ops.append(Operation(Gate.RX, (q1,), (pi / 2,)))
    ops.append(Operation(Gate.CX, (q1, q0)))
    if abs(a * pi) > tol:
        ops.append(Operation(Gate.RX, (q1,), (a * pi,)))
    if abs(b * pi) > tol:
        ops.append(Operation(Gate.RY, (q0,), (b * pi,)))
    ops.append(Operation(Gate.CX, (q0, q1)))
    ops.append(Operation(Gate.RX, (q0,), (-pi / 2,)))
    if abs(c * pi) > tol:
        ops.append(Operation(Gate.RZ, (q0,), (c * pi,)))
    ops.append(Operation(Gate.CX, (q1, q0)))

    ops.extend(_decompose_zxz(A1, q0))
    ops.extend(_decompose_zxz(A0, q1))
    return ops


def _count_cx_in_block(ops: list[Operation]) -> int:
    """Count CX-equivalent gates in block (SWAP=3, other 2Q=1)."""
    return sum(3 if op.gate == Gate.SWAP else 1 for op in ops if op.gate in (Gate.CX, Gate.CZ, Gate.CP, Gate.SWAP))


def _fuse_2q_ops(ops: list[Operation], n_qubits: int) -> list[Operation]:
    """Core 2Q block fusion on a flat op list."""
    result: list[Operation] = []
    i = 0
    while i < len(ops):
        op = ops[i]
        if op.gate.n_qubits != 2 or op.condition is not None:
            result.append(op)
            i += 1
            continue

        pair = frozenset(op.qubits)
        q0, q1 = min(pair), max(pair)
        block = [op]
        j = i + 1
        while j < len(ops):
            nxt = ops[j]
            if nxt.condition is not None or nxt.gate in (Gate.MEASURE, Gate.RESET) or nxt.gate.n_qubits >= 3:
                break
            if nxt.gate.n_qubits == 2:
                if frozenset(nxt.qubits) != pair:
                    break
            elif nxt.qubits[0] not in pair:
                break
            block.append(nxt)
            j += 1

        if len(block) < 2:
            result.append(op)
            i += 1
            continue

        # Skip blocks with symbolic parameters
        if any(_has_parameter(bop.params) for bop in block):
            result.extend(block)
            i = j
            continue

        orig_cx = _count_cx_in_block(block)
        if orig_cx < 2:
            result.extend(block)
            i = j
            continue

        # Build 4x4 unitary
        U = np.eye(4, dtype=complex)
        for bop in block:
            U = _gate_to_4x4(bop, q0, q1) @ U

        try:
            A0, A1, xx, yy, zz, B0, B1 = kak_decompose(U)
        except Exception:
            result.extend(block)
            i = j
            continue

        ncx = cx_count(xx, yy, zz)
        if ncx == 0:
            # U is a tensor product — factor directly (raw KAK angles may be non-canonical)
            U0, U1 = _extract_su2_pair(U)
            synth = _decompose_zxz(U1, q0) + _decompose_zxz(U0, q1)
        else:
            synth = _synthesize_2q(A0, A1, xx, yy, zz, B0, B1, q0, q1)
        synth_cx = sum(1 for s in synth if s.gate == Gate.CX)

        if synth_cx < orig_cx:
            result.extend(synth)
        else:
            result.extend(block)

        i = j

    return result


def fuse_2q_blocks(inp):
    """Fuse consecutive gates on same qubit pair via KAK decomposition. Accepts Circuit or DAGCircuit."""
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp
    result_ops = _fuse_2q_ops(list(dag.topological_ops()), dag.n_qubits)
    out = DAGCircuit(dag.n_qubits, dag.n_classical)
    for op in result_ops:
        out.add_op(op)
    return out.to_circuit() if from_circuit else out
