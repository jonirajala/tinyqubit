"""
Qubit routing for hardware connectivity.

Contains:
    - route(): Insert SWAPs to satisfy connectivity constraints
    - SWAPs are deferred and materialized at the end (with cancellation)
"""

from ..ir import Circuit, Operation, Gate
from ..target import Target
from ..tracker import QubitTracker, PendingSwap, PendingGate


def route(circuit: Circuit, target: Target) -> Circuit:
    """Route circuit to satisfy target connectivity with automatic SWAP insertion."""
    tracker = QubitTracker(circuit.n_qubits)

    if target.is_all_to_all():
        result = Circuit(circuit.n_qubits)
        result.ops = circuit.ops.copy()
        result._tracker = tracker
        return result

    for op_idx, op in enumerate(circuit.ops):
        if op.gate.n_qubits == 1 or op.gate == Gate.MEASURE:
            tracker.add_gate(op.gate, tracker.get_physical_qubits(op.qubits), op.params)
        else:
            log_a, log_b = op.qubits
            phys_a, phys_b = tracker.logical_to_phys(log_a), tracker.logical_to_phys(log_b)

            if not target.are_connected(phys_a, phys_b):
                path = target.shortest_path(phys_a, phys_b)
                for i in range(len(path) - 2):
                    tracker.record_swap(path[i], path[i + 1], op_idx)
                phys_a, phys_b = tracker.logical_to_phys(log_a), tracker.logical_to_phys(log_b)

            tracker.add_gate(op.gate, (phys_a, phys_b), op.params)

    # Materialize with SWAP cancellation
    result = Circuit(target.n_qubits)
    for op in tracker.materialize():
        if isinstance(op, PendingSwap):
            result.ops.append(Operation(Gate.SWAP, (op.phys_a, op.phys_b)))
        else:
            result.ops.append(Operation(op.gate, op.phys_qubits, op.params))
    result._tracker = tracker
    return result
