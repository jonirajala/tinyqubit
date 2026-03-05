"""
Main compilation entry point.

Two-phase pipeline:
1. precompile: Target-independent decompose + fuse + optimize
2. realize: Layout, route, decompose to native basis, re-optimize

transpile() runs both phases. precompile() and realize() can be called
separately to enable multi-target compilation from a single precompiled circuit.
"""

from dataclasses import dataclass

from .ir import Circuit, Gate
from .dag import DAGCircuit
from .target import Target
from .passes.route import route
from .passes.layout import select_layout
from .passes.decompose import decompose
from .passes.fuse import fuse_1q_gates, fuse_2q_blocks
from .passes.optimize import optimize
from .passes.push_diagonals import push_diagonals
from .passes.direction import fix_direction_dag
from .report import collect_metrics, build_report

# Routing basis: universal set that router understands
# Keep SWAP so router can reason about it before expansion
_ROUTING_BASIS = frozenset({Gate.RX, Gate.RZ, Gate.SX, Gate.CX, Gate.CZ, Gate.SWAP, Gate.H, Gate.MEASURE, Gate.RESET})


@dataclass(frozen=True)
class CompileConfig:
    sabre_trials: int = 5
    lookahead_depth: int = 20
    max_opt_iterations: int = 1000
    fuse_2q: bool = True
    t_optimal: bool = False
    objective: str = "2q"       # "2q" | "error" | "depth"
    dd: bool = False
    multi_trials: int = 1


PRESET_FAST = CompileConfig(sabre_trials=2, lookahead_depth=5, max_opt_iterations=100, fuse_2q=False)
PRESET_DEFAULT = CompileConfig()
PRESET_QUALITY = CompileConfig(sabre_trials=10, lookahead_depth=40, max_opt_iterations=2000, multi_trials=3)
PRESET_FT = CompileConfig(t_optimal=True, fuse_2q=False)
_PRESETS = {"fast": PRESET_FAST, "default": PRESET_DEFAULT, "quality": PRESET_QUALITY, "ft": PRESET_FT}


def _resolve_config(preset: str | CompileConfig) -> CompileConfig:
    if isinstance(preset, CompileConfig):
        return preset
    return _PRESETS[preset]


def _obj_to_route(objective: str) -> str | None:
    return None if objective == "2q" else objective


def _precompile_dag(dag: DAGCircuit, cfg: CompileConfig) -> DAGCircuit:
    """Target-independent: decompose to routing primitives, fuse, optimize."""
    dag = decompose(dag, _ROUTING_BASIS, t_optimal=cfg.t_optimal)
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = optimize(dag, max_iterations=cfg.max_opt_iterations)
    return dag


def _realize_dag(dag: DAGCircuit, target: Target, cfg: CompileConfig, _seed_offset: int = 0) -> DAGCircuit:
    """Target-specific: layout, route, decompose to native basis, re-optimize."""
    obj = _obj_to_route(cfg.objective)
    layout = select_layout(dag, target, objective=obj, sabre_trials=cfg.sabre_trials, seed_offset=_seed_offset)
    dag = route(dag, target, initial_layout=layout, objective=obj, lookahead_depth=cfg.lookahead_depth)
    tracker = getattr(dag, '_tracker', None)
    dag = decompose(dag, target.basis_gates, t_optimal=cfg.t_optimal)
    if cfg.fuse_2q:
        dag = fuse_2q_blocks(dag)
        dag = decompose(dag, target.basis_gates)
    dag = fix_direction_dag(dag, target)
    dag = push_diagonals(dag)
    dag = fuse_1q_gates(dag)
    dag = decompose(dag, target.basis_gates)  # Re-lower fused RX when basis uses SX
    dag = optimize(dag, max_iterations=cfg.max_opt_iterations, basis=target.basis_gates)
    dag._tracker = tracker
    return dag


def precompile(circuit: Circuit, preset: str | CompileConfig = "default") -> Circuit:
    """Target-independent optimization. Result can be realize()'d for any target."""
    cfg = _resolve_config(preset)
    dag = DAGCircuit.from_circuit(circuit)
    return _precompile_dag(dag, cfg).to_circuit()


def realize(circuit: Circuit, target: Target, preset: str | CompileConfig = "default") -> Circuit:
    """Target-specific lowering: route + decompose to native basis."""
    cfg = _resolve_config(preset)
    dag = DAGCircuit.from_circuit(circuit)
    dag = _realize_dag(dag, target, cfg)
    result = dag.to_circuit()
    result._tracker = getattr(dag, '_tracker', None)
    if cfg.dd and target.duration is not None:
        from .passes.dd import dynamic_decoupling
        result = dynamic_decoupling(result, target)
    return result


def transpile(circuit: Circuit, target: Target, *, preset: str | CompileConfig = "default",
              verbosity: int = 0, cache: dict | None = None, verify: bool = False) -> Circuit:
    """Transpile circuit for target hardware."""
    cfg = _resolve_config(preset)
    if cache is not None:
        key = (circuit._structure_key(), target.n_qubits, target.edges, target.basis_gates, target.directed, cfg)
        if key in cache:
            return cache[key]
    stages = [] if verbosity > 0 else None

    def track(dag: DAGCircuit, name: str) -> DAGCircuit:
        if stages is not None: stages.append(collect_metrics(dag, name))
        return dag

    dag = DAGCircuit.from_circuit(circuit)
    track(dag, "input")

    dag = _precompile_dag(dag, cfg)
    track(dag, "precompiled")

    # Multi-trial: run realize N times, pick lowest 2Q count
    if cfg.multi_trials > 1:
        best_dag, best_2q = None, float('inf')
        for trial in range(cfg.multi_trials):
            candidate = _realize_dag(dag, target, cfg, _seed_offset=trial * cfg.sabre_trials)
            n_2q = sum(1 for op in candidate.topological_ops() if op.gate.n_qubits == 2)
            if n_2q < best_2q:
                best_2q, best_dag = n_2q, candidate
        dag = best_dag
    else:
        dag = _realize_dag(dag, target, cfg)

    tracker = getattr(dag, '_tracker', None)
    track(dag, "output")

    result = dag.to_circuit()
    result._tracker = tracker

    if cfg.dd and target.duration is not None:
        from .passes.dd import dynamic_decoupling
        result = dynamic_decoupling(result, target)

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
