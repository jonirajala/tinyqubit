"""
Qubit routing for hardware connectivity.

Contains:
    - route(): Insert SWAPs to satisfy connectivity constraints
    - SWAPs are deferred and materialized at the end (with cancellation)
    - MEASURE/RESET/conditional ops are barriers (flush pending SWAPs first)
"""

from ..ir import Circuit, Operation, Gate
from ..target import Target
from ..tracker import QubitTracker, PendingSwap, PendingGate


def _is_barrier(op: Operation) -> bool:
    """Check if op is a barrier that requires flushing pending SWAPs."""
    return op.gate in (Gate.MEASURE, Gate.RESET) or op.condition is not None


def _emit_pending(tracker: QubitTracker, result: Circuit):
    """Flush tracker and append materialized ops to result."""
    for pending in tracker.flush():
        if isinstance(pending, PendingSwap):
            result.ops.append(Operation(Gate.SWAP, (pending.phys_a, pending.phys_b)))
        else:
            result.ops.append(Operation(pending.gate, pending.phys_qubits, pending.params))


def route(circuit: Circuit, target: Target) -> Circuit:
    """Route circuit to satisfy target connectivity with automatic SWAP insertion.

    Routing strategy: greedy shortest-path. For each 2Q gate on non-adjacent qubits,
    move the first qubit along the shortest path until adjacent. This is simple and
    deterministic but not always optimal (see SABRE for lookahead-based routing).
    """
    # Validate qubit counts
    if target.n_qubits < circuit.n_qubits:
        raise ValueError(f"Target has {target.n_qubits} qubits but circuit needs {circuit.n_qubits}")

    tracker = QubitTracker(circuit.n_qubits)

    if target.is_all_to_all():
        result = Circuit(circuit.n_qubits)
        result.ops = circuit.ops.copy()
        result._tracker = tracker
        return result

    result = Circuit(target.n_qubits)

    for op_idx, op in enumerate(circuit.ops):
        # Flush pending SWAPs before barriers
        if _is_barrier(op):
            _emit_pending(tracker, result)
            # Barriers use current physical mapping, emit directly
            phys_q = tracker.get_physical_qubits(op.qubits)
            result.ops.append(Operation(op.gate, phys_q, op.params, op.classical_bit, op.condition))
            continue

        if op.gate.n_qubits == 1:
            tracker.add_logical_gate(op.gate, op.qubits, op.params)
        else:
            log_a, log_b = op.qubits
            phys_a, phys_b = tracker.logical_to_phys(log_a), tracker.logical_to_phys(log_b)

            if not target.are_connected(phys_a, phys_b):
                path = target.shortest_path(phys_a, phys_b)
                if not path:
                    raise ValueError(f"No path between physical qubits {phys_a} and {phys_b} - "
                                     f"target topology is disconnected")
                for i in range(len(path) - 2):
                    tracker.record_swap(path[i], path[i + 1], op_idx)
                phys_a, phys_b = tracker.logical_to_phys(log_a), tracker.logical_to_phys(log_b)

            tracker.add_gate(op.gate, (phys_a, phys_b), op.params)

    # Final flush
    _emit_pending(tracker, result)
    result._tracker = tracker
    return result
