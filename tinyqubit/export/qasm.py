"""OpenQASM export: to_openqasm2(), to_openqasm3()"""
from __future__ import annotations

from ..ir import Circuit, Gate


class UnsupportedGateError(Exception):
    """Gate not supported in target format."""


def _format_gate(op) -> str:
    params = f'({op.params[0]})' if op.params else ''
    qubits = ', '.join(f'q[{q}]' for q in op.qubits)
    return f'{op.gate.name.lower()}{params} {qubits};'


def _add_mapping(lines, circuit):
    if hasattr(circuit, '_tracker') and circuit._tracker is not None:
        lines.extend(['', '// Qubit mapping (logical -> physical):'])
        lines.extend(f"  // logical q{i} -> physical q{circuit._tracker.logical_to_phys(i)}"
                     for i in range(circuit.n_qubits))


def to_openqasm2(circuit: Circuit, include_mapping: bool = True) -> str:
    """Export circuit to OpenQASM 2.0 format."""
    has_measure = any(op.gate == Gate.MEASURE for op in circuit.ops)
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{circuit.n_qubits}];']
    if has_measure: lines.append(f'creg c[{circuit.n_qubits}];')
    if include_mapping: _add_mapping(lines, circuit)

    for op in circuit.ops:
        if op.gate == Gate.CP:
            raise UnsupportedGateError("CP gate not in qelib1.inc. Use to_openqasm3() or decompose.")
        elif op.gate == Gate.MEASURE: lines.append(f'measure q[{op.qubits[0]}] -> c[{op.qubits[0]}];')
        else: lines.append(_format_gate(op))

    return '\n'.join(lines)


def to_openqasm3(circuit: Circuit, include_mapping: bool = True) -> str:
    """Export circuit to OpenQASM 3.0 format."""
    has_measure = any(op.gate == Gate.MEASURE for op in circuit.ops)
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', '', f'qubit[{circuit.n_qubits}] q;']
    if has_measure: lines.append(f'bit[{circuit.n_qubits}] c;')
    if include_mapping: _add_mapping(lines, circuit)
    lines.append('')

    for op in circuit.ops:
        if op.gate == Gate.MEASURE: lines.append(f'c[{op.qubits[0]}] = measure q[{op.qubits[0]}];')
        else: lines.append(_format_gate(op))

    return '\n'.join(lines)
