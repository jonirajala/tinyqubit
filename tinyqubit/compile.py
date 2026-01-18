"""
Main compilation entry point.

Contains:
    - transpile(circuit, target) → Circuit
    - Orchestrates: route → decompose → optimize
"""

from .ir import Circuit
from .target import Target
from .passes.route import route
from .passes.decompose import decompose
from .passes.optimize import optimize


def transpile(circuit: Circuit, target: Target) -> Circuit:
    """
    Transpile circuit for target hardware.

    Pipeline: route → decompose → optimize

    Args:
        circuit: Input circuit with logical qubit indices
        target: Hardware target (connectivity + basis gates)

    Returns:
        Transpiled circuit ready for execution on target
    """
    routed = route(circuit, target)
    decomposed = decompose(routed, target.basis_gates)
    optimized = optimize(decomposed)
    optimized._tracker = routed._tracker
    return optimized
