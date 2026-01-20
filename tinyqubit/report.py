"""Compilation report for explainability."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .tracker import PendingSwap

if TYPE_CHECKING:
    from .ir import Circuit, Operation
    from .tracker import QubitTracker
    from .target import Target


@dataclass
class PassMetrics:
    """Metrics captured after a single pass."""
    name: str
    gates: int
    two_q: int
    depth: int
    ops: list[str] | None = None


@dataclass
class CompileReport:
    """Compilation report with pass-by-pass metrics."""
    n_qubits: int
    passes: list[PassMetrics]
    swaps: tuple[int, int, int]  # (inserted, cancelled, final)
    swap_details: list[tuple[tuple[int, int], int, str, str]] = field(default_factory=list)
    qubit_map: dict[int, int] = field(default_factory=dict)
    target: str = ""
    basis: list[str] = field(default_factory=list)

    def to_text(self, verbosity: int = 2) -> str:
        lines = [
            "=" * 40, "  TinyQubit Compilation Report", "=" * 40, "",
            "SUMMARY",
            f"  Qubits: {self.n_qubits}",
            f"  Input:  {self.passes[0].gates} gates  Output: {self.passes[-1].gates} gates",
            f"  SWAPs:  {self.swaps[0]} inserted, {self.swaps[1]} cancelled, {self.swaps[2]} final",
        ]
        if self.target: lines.append(f"  Target: {self.target}")
        if self.basis: lines.append(f"  Basis:  {', '.join(self.basis)}")
        if verbosity < 2: return "\n".join(lines)

        lines += ["", "PASSES", "  Name            Gates  2Q  Depth", "  " + "-" * 34]
        lines += [f"  {m.name:<14} {m.gates:>5} {m.two_q:>4} {m.depth:>5}" for m in self.passes]

        if self.swap_details:
            lines += ["", "SWAPS"]
            for (a, b), idx, gate, reason in self.swap_details:
                lines += [f"  SWAP({a},{b}) by {gate} @{idx}: {reason}"]

        if self.qubit_map:
            lines += ["", "MAPPING"] + [f"  q{l}->p{p}" for l, p in sorted(self.qubit_map.items())]

        if verbosity >= 3:
            lines += ["", "OPS"]
            for m in self.passes:
                if m.ops:
                    lines += [f"  [{m.name}] {', '.join(m.ops)}"]
        return "\n".join(lines)


def _fmt(op: Operation) -> str:
    q = ",".join(map(str, op.qubits))
    return f"{op.gate.name}({q},{','.join(f'{p:.2f}' for p in op.params)})" if op.params else f"{op.gate.name}({q})"


def _depth(ops: list[Operation]) -> int:
    free: dict[int, int] = {}
    for op in ops:
        t = max((free.get(q, 0) for q in op.qubits), default=0) + 1
        for q in op.qubits: free[q] = t
    return max(free.values(), default=0)


def collect_metrics(circuit: Circuit, name: str) -> PassMetrics:
    return PassMetrics(name, len(circuit.ops), sum(op.gate.n_qubits == 2 for op in circuit.ops),
                       _depth(circuit.ops), [_fmt(op) for op in circuit.ops])


def build_report(input_circ: Circuit, output_circ: Circuit, passes: list[PassMetrics],
                 tracker: QubitTracker | None, target: Target) -> CompileReport:
    inserted = len(tracker.swap_log) if tracker else 0
    final = sum(isinstance(op, PendingSwap) for op in (tracker.materialize() if tracker else []))

    details = []
    for a, b, idx in (tracker.swap_log if tracker else []):
        if 0 <= idx < len(input_circ.ops):
            op = input_circ.ops[idx]
            details.append(((a, b), idx, _fmt(op), f"q{op.qubits[0]},q{op.qubits[1]} not connected"))
        else:
            details.append(((a, b), idx, "?", "manual"))

    return CompileReport(
        input_circ.n_qubits, passes, (inserted, inserted - final, final), details,
        {i: tracker.logical_to_physical[i] for i in range(tracker.n_qubits)} if tracker else {},
        getattr(target, 'name', ''), [g.name for g in target.basis_gates] if target.basis_gates else [])
