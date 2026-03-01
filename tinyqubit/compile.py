"""
Main compilation entry point.

Two-phase pipeline:
1. precompile: Target-independent decompose + fuse + optimize
2. realize: Layout, route, decompose to native basis, re-optimize

transpile() runs both phases. precompile() and realize() can be called
separately to enable multi-target compilation from a single precompiled circuit.
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


def _precompile_dag(dag: DAGCircuit, t_optimal: bool = False) -> DAGCircuit:
    """Target-independent: decompose to routing primitives, fuse, optimize."""
    dag = decompose(dag, _ROUTING_BASIS, t_optimal=t_optimal)
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = optimize(dag)
    return dag


def _realize_dag(dag: DAGCircuit, target: Target, t_optimal: bool = False, objective: str | None = None) -> DAGCircuit:
    """Target-specific: layout, route, decompose to native basis, re-optimize."""
    layout = select_layout(dag, target, objective=objective)
    dag = route(dag, target, initial_layout=layout, objective=objective)
    tracker = getattr(dag, '_tracker', None)
    dag = decompose(dag, target.basis_gates, t_optimal=t_optimal)
    dag = fuse_2q_blocks(dag)
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = optimize(dag, basis=target.basis_gates)
    dag._tracker = tracker
    return dag


def precompile(circuit: Circuit, t_optimal: bool = False) -> Circuit:
    """Target-independent optimization. Result can be realize()'d for any target."""
    dag = DAGCircuit.from_circuit(circuit)
    return _precompile_dag(dag, t_optimal).to_circuit()


def realize(circuit: Circuit, target: Target, t_optimal: bool = False, objective: str | None = None) -> Circuit:
    """Target-specific lowering: route + decompose to native basis."""
    dag = DAGCircuit.from_circuit(circuit)
    dag = _realize_dag(dag, target, t_optimal, objective=objective)
    result = dag.to_circuit()
    result._tracker = getattr(dag, '_tracker', None)
    return result


def transpile(circuit: Circuit, target: Target, verbosity: int = 0, cache: dict | None = None,
              verify: bool = False, t_optimal: bool = False, objective: str | None = None) -> Circuit:
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
        key = (circuit._structure_key(), target.n_qubits, target.edges, target.basis_gates, t_optimal, objective)
        if key in cache:
            return cache[key]
    stages = [] if verbosity > 0 else None

    def track(dag: DAGCircuit, name: str) -> DAGCircuit:
        if stages is not None: stages.append(collect_metrics(dag, name))
        return dag

    dag = DAGCircuit.from_circuit(circuit)
    track(dag, "input")

    dag = _precompile_dag(dag, t_optimal)
    track(dag, "precompiled")

    dag = _realize_dag(dag, target, t_optimal, objective=objective)
    tracker = getattr(dag, '_tracker', None)
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
