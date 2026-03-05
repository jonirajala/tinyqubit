"""Fault-tolerant quantum computing resource estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Circuit, Gate
from .dag import DAGCircuit


@dataclass(frozen=True)
class ResourceEstimate:
    logical_qubits: int
    t_count: int
    t_depth: int
    clifford_count: int
    code_distance: int
    physical_qubits: int
    error_rate: float


_T_GATES = frozenset({Gate.T, Gate.TDG})
_NON_LOGICAL = frozenset({Gate.MEASURE, Gate.RESET})


def resource_estimate(circuit: Circuit, code: str = "surface", error_rate: float = 1e-3, p_logical: float = 1e-10) -> ResourceEstimate:
    """Estimate fault-tolerant resources for a circuit using a simplified surface code model."""
    if code != "surface":
        raise ValueError(f"Unknown code: {code!r} (only 'surface' supported)")

    t_count = sum(1 for op in circuit.ops if op.gate in _T_GATES)
    clifford_count = sum(1 for op in circuit.ops if op.gate not in _T_GATES and op.gate not in _NON_LOGICAL)

    # T-depth: number of DAG layers containing at least one T/TDG
    layers = DAGCircuit.from_circuit(circuit).layers()
    t_depth = sum(1 for layer in layers if any(op.gate in _T_GATES for op in layer))

    # Surface code distance from target logical error rate (simplified Fowler et al.)
    d = max(3, math.ceil(math.log(p_logical / 0.1) / math.log(10 * error_rate)))
    if d % 2 == 0:
        d += 1

    # Physical qubits: data+syndrome per logical qubit + distillation factories
    qubits_per_logical = 2 * d * d
    physical_qubits = circuit.n_qubits * qubits_per_logical + t_count * 15 * qubits_per_logical

    return ResourceEstimate(
        logical_qubits=circuit.n_qubits,
        t_count=t_count,
        t_depth=t_depth,
        clifford_count=clifford_count,
        code_distance=d,
        physical_qubits=physical_qubits,
        error_rate=error_rate,
    )
