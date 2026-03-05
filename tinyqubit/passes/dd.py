"""Dynamic decoupling — insert identity-equivalent pulse sequences into idle periods."""
from __future__ import annotations

from ..ir import Circuit, Gate, Operation
from ..target import Target
from ..schedule import idle_periods, asap_times
from ..passes.decompose import decompose


_DD_SEQUENCES = {
    "XX": [Gate.X, Gate.X],
    "XY4": [Gate.X, Gate.Y, Gate.X, Gate.Y],
}


def _dd_native_duration(sequence: list[Gate], target: Target) -> int:
    c = Circuit(1)
    for g in sequence:
        c.ops.append(Operation(g, (0,)))
    c = decompose(c, target.basis_gates)
    return sum(target.duration.get(op.gate, 0) for op in c.ops)


def dynamic_decoupling(circuit: Circuit, target: Target, sequence: str = "XX") -> Circuit:
    """Insert DD sequences into idle qubit periods to suppress decoherence."""
    if target.duration is None:
        return circuit
    seq_gates = _DD_SEQUENCES[sequence]
    dd_dur = _dd_native_duration(seq_gates, target)
    if dd_dur == 0:
        return circuit

    active_qubits = {q for op in circuit.ops for q in op.qubits}
    # Find the last non-MEASURE/RESET end time per qubit — no DD after final measurement
    times = asap_times(circuit, target.duration)
    last_useful = {}
    for t, op in zip(times, circuit.ops):
        if op.gate not in (Gate.MEASURE, Gate.RESET):
            for q in op.qubits:
                last_useful[q] = t + target.duration.get(op.gate, 0)
    gaps = idle_periods(circuit, target)
    viable = [(q, s, e) for q, s, e in gaps if q in active_qubits and (e - s) >= dd_dur and s < last_useful.get(q, 0)]
    if not viable:
        return circuit

    # Map each gap to an insertion index: after the last op on that qubit before the gap
    insertions = []  # (index, qubit)
    for q, gap_start, _ in viable:
        insert_idx = 0
        for i, (op, t) in enumerate(zip(circuit.ops, times)):
            if q in op.qubits and t + target.duration.get(op.gate, 0) <= gap_start:
                insert_idx = i + 1
        insertions.append((insert_idx, q))

    # Sort descending by index so insertions don't shift earlier indices
    insertions.sort(key=lambda x: x[0], reverse=True)
    result = Circuit(circuit.n_qubits, circuit.n_classical)
    result.ops = list(circuit.ops)
    for idx, q in insertions:
        dd_ops = [Operation(g, (q,)) for g in seq_gates]
        result.ops[idx:idx] = dd_ops

    result = decompose(result, target.basis_gates)
    result._tracker = getattr(circuit, '_tracker', None)
    return result
