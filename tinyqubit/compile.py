"""
Main compilation entry point.
"""

from .ir import Circuit
from .target import Target
from .passes.route import route
from .passes.decompose import decompose
from .passes.optimize import optimize
from .report import collect_metrics, build_report


def transpile(circuit: Circuit, target: Target, verbosity: int = 0) -> Circuit:
    """
    Transpile circuit for target hardware.

    Args:
        circuit: Input circuit with logical qubit indices
        target: Hardware target (connectivity + basis gates)
        verbosity: 0=silent, 1=summary, 2=normal, 3=verbose
    """
    routed = route(circuit, target)
    decomposed = decompose(routed, target.basis_gates)
    optimized = optimize(decomposed)
    optimized._tracker = routed._tracker

    if verbosity > 0:
        passes = [collect_metrics(c, n) for c, n in
                  [(circuit, "input"), (routed, "route"), (decomposed, "decompose"), (optimized, "optimize")]]
        print(build_report(circuit, optimized, passes, routed._tracker, target).to_text(verbosity))

    return optimized
