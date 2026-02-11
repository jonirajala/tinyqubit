"""OpenQASM export: to_openqasm2(), to_openqasm3()"""
from __future__ import annotations

from ..ir import Circuit, Gate, Operation


class UnsupportedGateError(Exception):
    """Gate not supported in target format."""


def _format_params(op: Operation) -> str:
    return f"({', '.join(str(p) for p in op.params)})" if op.params else ''


def _format_gate(op: Operation) -> str:
    qubits = ', '.join(f'q[{q}]' for q in op.qubits)
    return f'{op.gate.name.lower()}{_format_params(op)} {qubits};'


def _needs_classical(circuit: Circuit) -> bool:
    """Check if circuit needs classical register (measure or conditional)."""
    return any(op.gate == Gate.MEASURE or op.condition is not None for op in circuit.ops)


def _add_mapping(lines: list[str], circuit: Circuit):
    if hasattr(circuit, '_tracker') and circuit._tracker is not None:
        lines.extend(['', '// Qubit mapping (logical -> physical):'])
        lines.extend(f"// q{i} -> p{circuit._tracker.logical_to_phys(i)}"
                     for i in range(circuit.n_qubits))


def to_openqasm2(circuit: Circuit, include_mapping: bool = True) -> str:
    """Export circuit to OpenQASM 2.0 format."""
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{circuit.n_qubits}];']
    if _needs_classical(circuit):
        lines.append(f'creg c[{circuit.n_classical}];')
    if include_mapping:
        _add_mapping(lines, circuit)

    for op in circuit.ops:
        if op.gate == Gate.CP:
            raise UnsupportedGateError("CP gate not in qelib1.inc. Use to_openqasm3() or decompose.")
        if op.gate == Gate.CCZ:
            raise UnsupportedGateError("CCZ gate not in qelib1.inc. Use to_openqasm3() or decompose.")

        if op.gate == Gate.MEASURE:
            cb = op.classical_bit if op.classical_bit is not None else op.qubits[0]
            stmt = f'measure q[{op.qubits[0]}] -> c[{cb}];'
        elif op.gate == Gate.RESET:
            stmt = f'reset q[{op.qubits[0]}];'
        else:
            stmt = _format_gate(op)

        # Wrap with conditional if present
        if op.condition is not None:
            bit, val = op.condition
            lines.append(f'if (c[{bit}] == {val}) {stmt}')
        else:
            lines.append(stmt)

    return '\n'.join(lines)


def to_openqasm3(circuit: Circuit, include_mapping: bool = True) -> str:
    """Export circuit to OpenQASM 3.0 format."""
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', '', f'qubit[{circuit.n_qubits}] q;']
    if _needs_classical(circuit):
        lines.append(f'bit[{circuit.n_classical}] c;')
    if include_mapping:
        _add_mapping(lines, circuit)
    lines.append('')

    for op in circuit.ops:
        if op.gate == Gate.MEASURE:
            cb = op.classical_bit if op.classical_bit is not None else op.qubits[0]
            stmt = f'c[{cb}] = measure q[{op.qubits[0]}];'
        elif op.gate == Gate.RESET:
            stmt = f'reset q[{op.qubits[0]}];'
        else:
            stmt = _format_gate(op)

        # Wrap with conditional if present
        if op.condition is not None:
            bit, val = op.condition
            lines.append(f'if (c[{bit}] == {val}) {{ {stmt} }}')
        else:
            lines.append(stmt)

    return '\n'.join(lines)
