"""
Symbolic qubit mapping tracker.

Contains:
    - QubitTracker: Track logical→physical qubit permutations
    - Deferred SWAP materialization with automatic cancellation

Note: SWAP cancellation is local (consecutive pairs only). Commutation-aware
cancellation across intervening gates is not implemented.
"""

from dataclasses import dataclass
from .ir import Gate


@dataclass
class PendingSwap:
    """A SWAP that hasn't been materialized yet."""
    phys_a: int
    phys_b: int
    triggered_by: int = -1


@dataclass
class PendingGate:
    """A gate with its physical qubits."""
    gate: Gate
    phys_qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


class QubitTracker:
    """
    Track logical→physical qubit mapping symbolically.

    SWAPs are not materialized immediately - they're recorded as pending.
    At export time, consecutive SWAP-SWAP pairs are cancelled automatically.

    For circuits with MEASURE/RESET/conditionals, caller must flush() before
    these barriers to preserve correct semantics.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.logical_to_physical = list(range(n_qubits))
        self.physical_to_logical = list(range(n_qubits))
        self.pending: list[PendingSwap | PendingGate] = []
        self.swap_log: list[tuple[int, int, int]] = []
        self.swap_cancel_log: list[tuple[int, int]] = []  # (cancelled_trigger1, cancelled_trigger2)

    def logical_to_phys(self, logical: int) -> int:
        """Get physical location of a logical qubit."""
        return self.logical_to_physical[logical]

    def phys_to_logical(self, physical: int) -> int:
        """Get logical qubit at a physical location."""
        return self.physical_to_logical[physical]

    def record_swap(self, phys_a: int, phys_b: int, triggered_by: int = -1):
        """Update mapping and record pending SWAP for later materialization."""
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
        """Record a gate at current physical positions (caller must translate qubits)."""
        self.pending.append(PendingGate(gate, phys_qubits, params))

    def add_logical_gate(self, gate: 'Gate', logical_qubits: tuple[int, ...], params: tuple[float, ...] = ()):
        """Record a gate, auto-translating logical to physical qubits."""
        self.add_gate(gate, self.get_physical_qubits(logical_qubits), params)

    def get_physical_qubits(self, logical_qubits: tuple[int, ...]) -> tuple[int, ...]:
        """Convert logical qubits to their current physical locations."""
        return tuple(self.logical_to_physical[q] for q in logical_qubits)

    def flush(self) -> list[PendingSwap | PendingGate]:
        """Materialize and clear pending ops. Use before barriers (MEASURE/RESET/conditional)."""
        ops = self.materialize()
        self.pending.clear()
        return ops

    def materialize(self) -> list[PendingSwap | PendingGate]:
        """Return pending ops with consecutive SWAP-SWAP pairs cancelled."""
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
