"""ASAP scheduling and idle-period detection."""
from __future__ import annotations

from .ir import Gate, Circuit
from .target import Target


def asap_times(circuit: Circuit, durations: dict[Gate, int]) -> list[int]:
    """ASAP start time (in dt) for each op. Ops already in dependency order."""
    avail = [0] * circuit.n_qubits
    times = []
    for op in circuit.ops:
        start = max(avail[q] for q in op.qubits)
        times.append(start)
        dur = durations.get(op.gate, 0)
        for q in op.qubits:
            avail[q] = start + dur
    return times


def circuit_duration(circuit: Circuit, target: Target) -> int:
    """Total circuit duration in dt. Returns 0 if target has no durations."""
    if target.duration is None:
        return 0
    times = asap_times(circuit, target.duration)
    if not times:
        return 0
    return max(t + target.duration.get(op.gate, 0) for t, op in zip(times, circuit.ops))


def idle_periods(circuit: Circuit, target: Target) -> list[tuple[int, int, int]]:
    """Return (qubit, start_dt, end_dt) for every idle gap, including trailing idles."""
    if target.duration is None:
        return []
    times = asap_times(circuit, target.duration)
    total = circuit_duration(circuit, target)
    if total == 0:
        return []

    busy: dict[int, list[tuple[int, int]]] = {q: [] for q in range(circuit.n_qubits)}
    for t, op in zip(times, circuit.ops):
        dur = target.duration.get(op.gate, 0)
        for q in op.qubits:
            busy[q].append((t, t + dur))

    gaps = []
    for q in range(circuit.n_qubits):
        intervals = sorted(busy[q])
        cursor = 0
        for start, end in intervals:
            if start > cursor:
                gaps.append((q, cursor, start))
            cursor = max(cursor, end)
        if cursor < total:
            gaps.append((q, cursor, total))
    return gaps
