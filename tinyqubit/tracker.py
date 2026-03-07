"""Symbolic qubit mapping tracker with deferred SWAP materialization."""

from dataclasses import dataclass
from .ir import Gate


@dataclass
class PendingSwap:
    phys_a: int
    phys_b: int
    triggered_by: int = -1


@dataclass
class PendingGate:
    gate: Gate
    phys_qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


class QubitTracker:
    """Track logical→physical qubit mapping with deferred SWAP materialization."""

    def __init__(self, n_qubits: int, initial_layout: list[int] | None = None):
        self.n_qubits = n_qubits
        self.initial_layout = initial_layout
        self.logical_to_physical = list(range(n_qubits))
        self.physical_to_logical = list(range(n_qubits))
        if initial_layout is not None:
            used = set(initial_layout)
            remaining = [p for p in range(n_qubits) if p not in used]
            for lq, pq in enumerate(initial_layout):
                self.logical_to_physical[lq] = pq
                self.physical_to_logical[pq] = lq
            for i, lq in enumerate(range(len(initial_layout), n_qubits)):
                self.logical_to_physical[lq] = remaining[i]
                self.physical_to_logical[remaining[i]] = lq
        self.pending: list[PendingSwap | PendingGate] = []
        self.swap_log: list[tuple[int, int, int]] = []
        self.swap_cancel_log: list[tuple[int, int]] = []

    def logical_to_phys(self, logical: int) -> int:
        return self.logical_to_physical[logical]

    def phys_to_logical(self, physical: int) -> int:
        return self.physical_to_logical[physical]

    def record_swap(self, phys_a: int, phys_b: int, triggered_by: int = -1):
        if not (0 <= phys_a < self.n_qubits and 0 <= phys_b < self.n_qubits):
            raise ValueError(f"Invalid physical qubit index: ({phys_a}, {phys_b}) for {self.n_qubits}-qubit tracker")
        if phys_a == phys_b:
            return  # No-op swap
        # Swap mappings
        log_a, log_b = self.physical_to_logical[phys_a], self.physical_to_logical[phys_b]
        self.logical_to_physical[log_a], self.logical_to_physical[log_b] = phys_b, phys_a
        self.physical_to_logical[phys_a], self.physical_to_logical[phys_b] = log_b, log_a
        # Record
        self.pending.append(PendingSwap(phys_a, phys_b, triggered_by))
        self.swap_log.append((phys_a, phys_b, triggered_by))

    def add_gate(self, gate: 'Gate', phys_qubits: tuple[int, ...], params: tuple[float, ...] = ()):
        """Caller must translate qubits to physical before calling."""
        self.pending.append(PendingGate(gate, phys_qubits, params))


    def flush(self) -> list[PendingSwap | PendingGate]:
        """Use before barriers (MEASURE/RESET/conditional) to preserve semantics."""
        ops = self.materialize()
        self.pending.clear()
        return ops

    def materialize(self) -> list[PendingSwap | PendingGate]:
        """Return pending ops, cancelling consecutive SWAP-SWAP pairs."""
        result = []
        for op in self.pending:
            if (isinstance(op, PendingSwap) and result and
                isinstance(result[-1], PendingSwap) and
                {result[-1].phys_a, result[-1].phys_b} == {op.phys_a, op.phys_b}):
                # Log cancellation for reports
                self.swap_cancel_log.append((result[-1].triggered_by, op.triggered_by))
                result.pop()
            else:
                result.append(op)
        return result
