"""Hardware backend adapters: IBM Quantum, AWS Braket."""
from __future__ import annotations

from .ibm_native import submit_ibm, wait_ibm, list_ibm_backends, ibm_target, IBMJob
from .braket import submit_to_braket, get_braket_results

__all__ = [
    "submit_ibm", "wait_ibm", "list_ibm_backends", "ibm_target", "IBMJob",
    "submit_to_braket", "get_braket_results",
]


def _normalize_counts(counts: dict[str, int], n_qubits: int, reverse_bits: bool = False, tracker=None) -> dict[str, int]:
    if not reverse_bits and tracker is None:
        return counts
    result = {}
    for bitstring, count in counts.items():
        bits = list(bitstring[::-1] if reverse_bits else bitstring)
        if tracker is not None:
            phys = list(bits)
            for p in range(n_qubits):
                bits[tracker.phys_to_logical(p)] = phys[p]
        key = ''.join(bits)
        result[key] = result.get(key, 0) + count
    return result
