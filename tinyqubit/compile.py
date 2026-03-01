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

Internally uses DAGCircuit as the IR — built once from Circuit, passed through
all passes, and converted back to Circuit at the end.
"""

from .ir import Circuit, Gate
from .dag import DAGCircuit
from .target import Target
from .passes.route import route
from .passes.layout import select_layout
from .passes.decompose import decompose
from .passes.fuse import fuse_1q_gates, fuse_2q_blocks
from .passes.optimize import optimize
from .passes.push_diagonals import push_diagonals
from .report import collect_metrics, build_report

# Routing basis: universal set that router understands
# Keep SWAP so router can reason about it before expansion
_ROUTING_BASIS = frozenset({Gate.RX, Gate.RZ, Gate.CX, Gate.CZ, Gate.SWAP, Gate.H, Gate.MEASURE, Gate.RESET})


def transpile(circuit: Circuit, target: Target, verbosity: int = 0, cache: dict | None = None,
              verify: bool = False, t_optimal: bool = False) -> Circuit:
    """
    Transpile circuit for target hardware.

    Args:
        circuit: Input circuit with logical qubit indices
        target: Hardware target (connectivity + basis gates)
        verbosity: 0=silent, 1=summary, 2=normal, 3=verbose
        cache: Optional dict for caching compiled circuits by structure.
            NOTE: cache returns the same mutable object on hits, so bind_params
            on a cached result mutates the cached entry. Intended pattern is
            "transpile once, rebind many times" on the same circuit object.
        verify: If True, check equivalence between input and output circuits
        t_optimal: If True, use relative-phase Toffoli (4T instead of 7T).
            Safe for compute-uncompute patterns; not exact for bare CCX/CCZ.
    """
    if cache is not None:
        key = (circuit._structure_key(), target.n_qubits, target.edges, target.basis_gates, t_optimal)
        if key in cache:
            return cache[key]
    stages = [] if verbosity > 0 else None

    def track(dag: DAGCircuit, name: str) -> DAGCircuit:
        if stages is not None: stages.append(collect_metrics(dag, name))
        return dag

    dag = DAGCircuit.from_circuit(circuit)
    track(dag, "input")

    # Phase 1: Pre-routing - decompose to routing primitives, optimize
    dag = decompose(dag, _ROUTING_BASIS, t_optimal=t_optimal)
    track(dag, "decompose1")
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = optimize(dag)
    track(dag, "opt1")

    # Phase 1.5: Select initial layout
    layout = select_layout(dag, target)

    # Phase 2: Route for target connectivity
    dag = route(dag, target, initial_layout=layout)
    tracker = getattr(dag, '_tracker', None)
    track(dag, "route")

    # Phase 3: Post-routing - decompose to target basis, optimize
    dag = decompose(dag, target.basis_gates, t_optimal=t_optimal)
    track(dag, "decompose2")
    dag = fuse_2q_blocks(dag)
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = optimize(dag, basis=target.basis_gates)
    track(dag, "output")

    result = dag.to_circuit()
    result._tracker = tracker

    verified = None
    if verify:
        from .simulator import verify as _verify
        verified = _verify(circuit, result, tracker=tracker)
        if not verified:
            import warnings
            warnings.warn("Compiled circuit failed equivalence check")

    if verbosity > 0:
        report = build_report(circuit, result, stages, tracker, target)
        report.verified = verified
        print(report.to_text(verbosity))

    if cache is not None:
        cache[key] = result

    return result
