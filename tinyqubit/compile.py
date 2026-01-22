"""
Main compilation entry point.

Pipeline order (optimized for gate count):
1. decompose1: Decompose to routing primitives (keep SWAP for routing)
2. fuse1 + opt1: Pre-routing optimization
3. route: Insert SWAPs for connectivity
4. decompose2: Expand SWAP to target basis (e.g., 3×CX)
5. fuse2 + opt2: Post-routing optimization

This order ensures:
- Routing sees true 2Q structure (not high-level gates)
- Optimization works on both sides of routing
- SWAP patterns are cleaned up after expansion
"""

from .ir import Circuit, Gate
from .target import Target
from .passes.route import route
from .passes.decompose import decompose
from .passes.fuse import fuse_1q_gates
from .passes.optimize import optimize
from .passes.push_diagonals import push_diagonals
from .report import collect_metrics, build_report

# Routing basis: universal set that router understands
# Keep SWAP so router can reason about it before expansion
_ROUTING_BASIS = frozenset({Gate.RX, Gate.RZ, Gate.CX, Gate.CZ, Gate.SWAP, Gate.H, Gate.MEASURE, Gate.RESET})


def transpile(circuit: Circuit, target: Target, verbosity: int = 0) -> Circuit:
    """
    Transpile circuit for target hardware.

    Args:
        circuit: Input circuit with logical qubit indices
        target: Hardware target (connectivity + basis gates)
        verbosity: 0=silent, 1=summary, 2=normal, 3=verbose
    """
    # Phase 1: Pre-routing - decompose to routing primitives, optimize
    decomposed1 = decompose(circuit, _ROUTING_BASIS)
    pushed1 = push_diagonals(decomposed1)
    fused1 = fuse_1q_gates(pushed1)
    opt1 = optimize(fused1)

    # Phase 2: Route for target connectivity
    routed = route(opt1, target)

    # Phase 3: Post-routing - decompose to target basis, optimize
    decomposed2 = decompose(routed, target.basis_gates)
    pushed2 = push_diagonals(decomposed2)
    fused2 = fuse_1q_gates(pushed2)
    optimized = optimize(fused2)
    optimized._tracker = routed._tracker

    if verbosity > 0:
        passes = [collect_metrics(c, n) for c, n in [
            (circuit, "input"), (decomposed1, "decompose1"), (opt1, "opt1"),
            (routed, "route"), (decomposed2, "decompose2"), (optimized, "output")]]
        print(build_report(circuit, optimized, passes, routed._tracker, target).to_text(verbosity))

    return optimized
