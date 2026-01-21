"""
Gate decomposition to target basis.

Contains:
    - decompose(): Convert non-basis gates to basis gate sequences
    - DECOMPOSITIONS: Default rules (CX-native, for IBM-like backends)
    - DECOMPOSITIONS_CZ_NATIVE: Alternative rules for CZ-native backends

Standard decompositions (all correct up to global phase):
    - SWAP → CX CX CX
    - H → RZ(π/2) RX(π/2) RZ(π/2)
    - S → RZ(π/2), T → RZ(π/4)
    - X → RX(π), Y → RX(π) RZ(π), Z → RZ(π)
    - RX → H RZ(θ) H (when H is in basis but RX isn't)
    - RY → RX(π/2) RZ(θ) RX(-π/2)
    - CZ → H CX H (default) or CX → H CZ H (CZ-native)
    - CP → RZ(θ/2)_c · CX · RZ(-θ/2)_t · CX · RZ(θ/2)_c
"""

from math import pi
from ..ir import Circuit, Operation, Gate


def _decompose_swap(q0: int, q1: int) -> list[Operation]:
    """SWAP = CX(0,1) CX(1,0) CX(0,1)"""
    return [
        Operation(Gate.CX, (q0, q1)),
        Operation(Gate.CX, (q1, q0)),
        Operation(Gate.CX, (q0, q1)),
    ]


def _decompose_h(q: int) -> list[Operation]:
    """H = RZ(π/2) RX(π/2) RZ(π/2)"""
    return [
        Operation(Gate.RZ, (q,), (pi/2,)),
        Operation(Gate.RX, (q,), (pi/2,)),
        Operation(Gate.RZ, (q,), (pi/2,)),
    ]


def _decompose_s(q: int) -> list[Operation]:
    """S = RZ(π/2)"""
    return [Operation(Gate.RZ, (q,), (pi/2,))]


def _decompose_t(q: int) -> list[Operation]:
    """T = RZ(π/4)"""
    return [Operation(Gate.RZ, (q,), (pi/4,))]


def _decompose_sdg(q: int) -> list[Operation]:
    """S† = RZ(-π/2)"""
    return [Operation(Gate.RZ, (q,), (-pi/2,))]


def _decompose_tdg(q: int) -> list[Operation]:
    """T† = RZ(-π/4)"""
    return [Operation(Gate.RZ, (q,), (-pi/4,))]


def _decompose_x(q: int) -> list[Operation]:
    """X = RX(π)"""
    return [Operation(Gate.RX, (q,), (pi,))]


def _decompose_y(q: int) -> list[Operation]:
    """Y = RX(π) RZ(π)"""
    return [
        Operation(Gate.RX, (q,), (pi,)),
        Operation(Gate.RZ, (q,), (pi,)),
    ]


def _decompose_z(q: int) -> list[Operation]:
    """Z = RZ(π)"""
    return [Operation(Gate.RZ, (q,), (pi,))]


def _decompose_ry(q: int, theta: float) -> list[Operation]:
    """RY(θ) = RX(π/2) RZ(θ) RX(-π/2)"""
    return [
        Operation(Gate.RX, (q,), (pi/2,)),
        Operation(Gate.RZ, (q,), (theta,)),
        Operation(Gate.RX, (q,), (-pi/2,)),
    ]


def _decompose_rx(q: int, theta: float) -> list[Operation]:
    """RX(θ) = H RZ(θ) H (since H rotates Z-axis to X-axis)"""
    return [
        Operation(Gate.H, (q,)),
        Operation(Gate.RZ, (q,), (theta,)),
        Operation(Gate.H, (q,)),
    ]


def _decompose_cz(q0: int, q1: int) -> list[Operation]:
    """CZ = H(target) CX H(target)"""
    return [
        Operation(Gate.H, (q1,)),
        Operation(Gate.CX, (q0, q1)),
        Operation(Gate.H, (q1,)),
    ]


def _decompose_cx_to_cz(q0: int, q1: int) -> list[Operation]:
    """CX = H(target) CZ H(target) -- only used for CZ-native backends."""
    return [
        Operation(Gate.H, (q1,)),
        Operation(Gate.CZ, (q0, q1)),
        Operation(Gate.H, (q1,)),
    ]


def _decompose_cp(c: int, t: int, theta: float) -> list[Operation]:
    """CP(θ) = RZ(θ/2)_c · CX · RZ(-θ/2)_t · CX · RZ(θ/2)_c"""
    return [
        Operation(Gate.RZ, (c,), (theta/2,)),
        Operation(Gate.CX, (c, t)),
        Operation(Gate.RZ, (t,), (-theta/2,)),
        Operation(Gate.CX, (c, t)),
        Operation(Gate.RZ, (c,), (theta/2,)),
    ]


# Decomposition rules: gate -> function(qubits, params) -> list[Operation]
# Note: CX is treated as primitive. CZ decomposes to CX.
# For CZ-native backends (Rigetti, Google), use DECOMPOSITIONS_CZ_NATIVE instead.
DECOMPOSITIONS = {
    Gate.SWAP: lambda qs, ps: _decompose_swap(qs[0], qs[1]),
    Gate.H: lambda qs, ps: _decompose_h(qs[0]),
    Gate.S: lambda qs, ps: _decompose_s(qs[0]),
    Gate.T: lambda qs, ps: _decompose_t(qs[0]),
    Gate.SDG: lambda qs, ps: _decompose_sdg(qs[0]),
    Gate.TDG: lambda qs, ps: _decompose_tdg(qs[0]),
    Gate.X: lambda qs, ps: _decompose_x(qs[0]),
    Gate.Y: lambda qs, ps: _decompose_y(qs[0]),
    Gate.Z: lambda qs, ps: _decompose_z(qs[0]),
    Gate.RX: lambda qs, ps: _decompose_rx(qs[0], ps[0]),
    Gate.RY: lambda qs, ps: _decompose_ry(qs[0], ps[0]),
    Gate.CZ: lambda qs, ps: _decompose_cz(qs[0], qs[1]),
    Gate.CP: lambda qs, ps: _decompose_cp(qs[0], qs[1], ps[0]),
}

# Alternative decompositions for CZ-native backends (Rigetti, Google, IQM)
DECOMPOSITIONS_CZ_NATIVE = {
    **{k: v for k, v in DECOMPOSITIONS.items() if k != Gate.CZ},
    Gate.CX: lambda qs, ps: _decompose_cx_to_cz(qs[0], qs[1]),
}


def decompose(circuit: Circuit, basis: frozenset[Gate]) -> Circuit:
    """Decompose non-basis gates to target basis."""
    # Auto-detect CZ-native vs CX-native basis
    rules = DECOMPOSITIONS_CZ_NATIVE if (Gate.CZ in basis and Gate.CX not in basis) else DECOMPOSITIONS
    ops = list(circuit.ops)

    changed = True
    while changed:
        changed, new_ops = False, []
        for op in ops:
            if op.gate in basis or op.gate in (Gate.MEASURE, Gate.RESET):
                new_ops.append(op)
            elif op.gate in rules:
                # Propagate condition to all decomposed ops
                for new_op in rules[op.gate](op.qubits, op.params):
                    new_ops.append(Operation(new_op.gate, new_op.qubits, new_op.params,
                                             new_op.classical_bit, op.condition))
                changed = True
            else:
                raise NotImplementedError(f"No decomposition rule for {op.gate.name}")
        ops = new_ops

    result = Circuit(circuit.n_qubits)
    result.ops = ops
    return result
