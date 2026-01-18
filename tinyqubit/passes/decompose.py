"""
Gate decomposition to target basis.

Contains:
    - decompose(): Convert non-basis gates to basis gate sequences

Real Quantum Computer Native Gates:
    - IBM (superconducting):     RZ, SX (√X), X, ECR (or CX)
    - Rigetti (superconducting): RZ, RX(±π/2, ±π), CZ
    - IonQ (trapped ion):        GPI, GPI2, MS (all-to-all connectivity)
    - Google Sycamore:           √iSWAP, SYC (FSim), CZ, arbitrary XY
    - IQM (superconducting):     PRX (arbitrary XY rotation), CZ

Standard decompositions:
    - SWAP → CX CX CX
    - H → RZ(π/2) RX(π/2) RZ(π/2)
    - S → RZ(π/2)
    - T → RZ(π/4)
    - X → RX(π)
    - Y → RX(π) RZ(π)
    - Z → RZ(π)
    - RY → RX(π/2) RZ(θ) RX(-π/2)
    - CZ → H CX H
    - CX → H CZ H
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


def _decompose_cz(q0: int, q1: int) -> list[Operation]:
    """CZ = H(target) CX H(target)"""
    return [
        Operation(Gate.H, (q1,)),
        Operation(Gate.CX, (q0, q1)),
        Operation(Gate.H, (q1,)),
    ]


def _decompose_cx(q0: int, q1: int) -> list[Operation]:
    """CX = H(target) CZ H(target)"""
    return [
        Operation(Gate.H, (q1,)),
        Operation(Gate.CZ, (q0, q1)),
        Operation(Gate.H, (q1,)),
    ]


# Decomposition rules: gate -> function(qubits, params) -> list[Operation]
DECOMPOSITIONS = {
    Gate.SWAP: lambda qs, ps: _decompose_swap(qs[0], qs[1]),
    Gate.H: lambda qs, ps: _decompose_h(qs[0]),
    Gate.S: lambda qs, ps: _decompose_s(qs[0]),
    Gate.T: lambda qs, ps: _decompose_t(qs[0]),
    Gate.X: lambda qs, ps: _decompose_x(qs[0]),
    Gate.Y: lambda qs, ps: _decompose_y(qs[0]),
    Gate.Z: lambda qs, ps: _decompose_z(qs[0]),
    Gate.RY: lambda qs, ps: _decompose_ry(qs[0], ps[0]),
    Gate.CZ: lambda qs, ps: _decompose_cz(qs[0], qs[1]),
    Gate.CX: lambda qs, ps: _decompose_cx(qs[0], qs[1]),
}


def decompose(circuit: Circuit, basis: frozenset[Gate]) -> Circuit:
    """Decompose non-basis gates to target basis."""
    ops = list(circuit.ops)

    changed = True
    while changed:
        changed, new_ops = False, []
        for op in ops:
            if op.gate in basis or op.gate == Gate.MEASURE:
                new_ops.append(op)
            elif op.gate in DECOMPOSITIONS:
                new_ops.extend(DECOMPOSITIONS[op.gate](op.qubits, op.params))
                changed = True
            else:
                raise NotImplementedError(f"No decomposition rule for {op.gate.name}")
        ops = new_ops

    result = Circuit(circuit.n_qubits)
    result.ops = ops
    return result
