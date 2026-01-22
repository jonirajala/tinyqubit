"""
Additional circuit metrics for benchmarks.

Provides specialized gate counting:
- count_swaps: Count SWAP gates specifically
- count_t_gates: Count T and TDG gates (fault-tolerant cost metric)
- count_2q_gates: Count all two-qubit gates
- count_1q_gates: Count all single-qubit gates
- circuit_depth: Calculate circuit depth
- gate_breakdown: Full breakdown of all gate types
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit, Gate


def count_swaps(circuit: Circuit) -> int:
    """
    Count SWAP gates in circuit.

    Args:
        circuit: TinyQubit circuit

    Returns:
        Number of SWAP gates
    """
    return sum(1 for op in circuit.ops if op.gate == Gate.SWAP)


def count_t_gates(circuit: Circuit) -> int:
    """
    Count T and TDG gates (fault-tolerant metric).

    In fault-tolerant quantum computing, T gates are expensive as they
    require magic state distillation. This metric is important for
    assessing the cost of running on error-corrected hardware.

    Args:
        circuit: TinyQubit circuit

    Returns:
        Total count of T and TDG gates
    """
    return sum(1 for op in circuit.ops if op.gate in {Gate.T, Gate.TDG})


def count_2q_gates(circuit: Circuit) -> int:
    """
    Count all two-qubit gates (CX, CZ, SWAP, CP).

    Args:
        circuit: TinyQubit circuit

    Returns:
        Number of two-qubit gates
    """
    two_qubit = {Gate.CX, Gate.CZ, Gate.SWAP, Gate.CP}
    return sum(1 for op in circuit.ops if op.gate in two_qubit)


def count_1q_gates(circuit: Circuit) -> int:
    """
    Count all single-qubit gates.

    Args:
        circuit: TinyQubit circuit

    Returns:
        Number of single-qubit gates
    """
    return sum(1 for op in circuit.ops if op.gate.n_qubits == 1 and op.gate not in {Gate.MEASURE, Gate.RESET})


def count_cx_equivalent(circuit: Circuit) -> int:
    """
    Count CX-equivalent gates (SWAP = 3 CX).

    This is the standard metric for comparing routing quality,
    as SWAP gates decompose to 3 CX gates.

    Args:
        circuit: TinyQubit circuit

    Returns:
        CX-equivalent count
    """
    count = 0
    for op in circuit.ops:
        if op.gate == Gate.SWAP:
            count += 3
        elif op.gate in {Gate.CX, Gate.CZ, Gate.CP}:
            count += 1
    return count


def circuit_depth(circuit: Circuit) -> int:
    """
    Calculate circuit depth (longest path through circuit).

    Args:
        circuit: TinyQubit circuit

    Returns:
        Circuit depth
    """
    if not circuit.ops:
        return 0

    # Track when each qubit becomes free
    qubit_time = {}

    for op in circuit.ops:
        # Gate starts when all its qubits are free
        start_time = max((qubit_time.get(q, 0) for q in op.qubits), default=0)
        end_time = start_time + 1

        # Update qubit availability
        for q in op.qubits:
            qubit_time[q] = end_time

    return max(qubit_time.values()) if qubit_time else 0


def gate_breakdown(circuit: Circuit) -> dict[str, int]:
    """
    Full breakdown of gate counts by type.

    Args:
        circuit: TinyQubit circuit

    Returns:
        Dictionary mapping gate name to count
    """
    counts: dict[str, int] = {}
    for op in circuit.ops:
        name = op.gate.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def compare_circuits(original: Circuit, optimized: Circuit) -> dict:
    """
    Compare metrics between original and optimized circuits.

    Args:
        original: Original circuit
        optimized: Optimized circuit

    Returns:
        Dictionary with comparison metrics
    """
    return {
        "original": {
            "total_gates": len(original.ops),
            "1q_gates": count_1q_gates(original),
            "2q_gates": count_2q_gates(original),
            "t_count": count_t_gates(original),
            "depth": circuit_depth(original),
        },
        "optimized": {
            "total_gates": len(optimized.ops),
            "1q_gates": count_1q_gates(optimized),
            "2q_gates": count_2q_gates(optimized),
            "t_count": count_t_gates(optimized),
            "depth": circuit_depth(optimized),
        },
        "reduction": {
            "total_gates": len(original.ops) - len(optimized.ops),
            "1q_gates": count_1q_gates(original) - count_1q_gates(optimized),
            "2q_gates": count_2q_gates(original) - count_2q_gates(optimized),
            "t_count": count_t_gates(original) - count_t_gates(optimized),
            "depth": circuit_depth(original) - circuit_depth(optimized),
        }
    }
