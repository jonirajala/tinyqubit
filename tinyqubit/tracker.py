"""
Symbolic qubit mapping tracker.

Contains:
    - QubitTracker: Track logical→physical qubit permutations
    - Deferred SWAP materialization with automatic cancellation
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
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # logical_to_physical[i] = physical qubit where logical qubit i currently lives
        self.logical_to_physical = list(range(n_qubits))
        # physical_to_logical[i] = logical qubit currently at physical location i
        self.physical_to_logical = list(range(n_qubits))
        # Pending operations (SWAPs and gates, in order)
        self.pending: list[PendingSwap | PendingGate] = []
        # Log of swaps for debugging: [(physical_a, physical_b, triggered_by), ...]
        self.swap_log: list[tuple[int, int, int]] = []

    def logical_to_phys(self, logical: int) -> int:
        """Get physical location of a logical qubit."""
        return self.logical_to_physical[logical]

    def phys_to_logical(self, physical: int) -> int:
        """Get logical qubit at a physical location."""
        return self.physical_to_logical[physical]

    def record_swap(self, phys_a: int, phys_b: int, triggered_by: int = -1):
        """Update mapping and record pending SWAP for later materialization."""
        # Swap mappings
        log_a, log_b = self.physical_to_logical[phys_a], self.physical_to_logical[phys_b]
        self.logical_to_physical[log_a], self.logical_to_physical[log_b] = phys_b, phys_a
        self.physical_to_logical[phys_a], self.physical_to_logical[phys_b] = log_b, log_a
        # Record
        self.pending.append(PendingSwap(phys_a, phys_b, triggered_by))
        self.swap_log.append((phys_a, phys_b, triggered_by))


    def add_gate(self, gate: 'Gate', phys_qubits: tuple[int, ...], params: tuple[float, ...] = ()):
        """Record a gate at current physical positions."""
        self.pending.append(PendingGate(gate, phys_qubits, params))

    def get_physical_qubits(self, logical_qubits: tuple[int, ...]) -> tuple[int, ...]:
        """Convert logical qubits to their current physical locations."""
        return tuple(self.logical_to_physical[q] for q in logical_qubits)

    def materialize(self) -> list[PendingSwap | PendingGate]:
        """Return pending ops with consecutive SWAP-SWAP pairs cancelled."""
        result = []
        for op in self.pending:
            if (isinstance(op, PendingSwap) and result and
                isinstance(result[-1], PendingSwap) and
                {result[-1].phys_a, result[-1].phys_b} == {op.phys_a, op.phys_b}):
                result.pop()
            else:
                result.append(op)
        return result
