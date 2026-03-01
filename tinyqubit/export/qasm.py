"""OpenQASM export: to_openqasm2(), to_openqasm3()"""
from __future__ import annotations

from ..ir import Circuit, Gate, Operation, Parameter


class UnsupportedGateError(Exception):
    """Gate not supported in target format."""


# Gates NOT in stdgates.inc that need explicit definitions in QASM3
_QASM3_GATE_DEFS = {
    Gate.ECR: "gate ecr a, b { rz(1.5707963267948966) a; sx a; rz(3.141592653589793) a; cx a, b; x b; }",
    Gate.RZZ: "gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }",
}


def _format_params(op: Operation) -> str:
    if not op.params:
        return ''
    parts = [p.name if isinstance(p, Parameter) else str(p) for p in op.params]
    return f"({', '.join(parts)})"


def _format_gate(op: Operation, physical_qubits: bool = False, tracker=None) -> str:
    fmt = (lambda q: f'${tracker.logical_to_phys(q)}') if physical_qubits else (lambda q: f'q[{q}]')
    return f'{op.gate.name.lower()}{_format_params(op)} {", ".join(fmt(q) for q in op.qubits)};'


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
    if circuit.is_parameterized:
        raise ValueError("Cannot export parameterized circuit to OpenQASM 2.0. Call circuit.bind() first.")
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{circuit.n_qubits}];']
    if _needs_classical(circuit):
        lines.append(f'creg c[{circuit.n_classical}];')
    if include_mapping:
        _add_mapping(lines, circuit)

    _QASM2_UNSUPPORTED = {
        Gate.CP: "CP", Gate.CCZ: "CCZ", Gate.ECR: "ECR", Gate.RZZ: "RZZ",
    }
    for op in circuit.ops:
        if op.gate in _QASM2_UNSUPPORTED:
            raise UnsupportedGateError(f"{_QASM2_UNSUPPORTED[op.gate]} gate not in qelib1.inc. Use to_openqasm3() or decompose.")

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


def to_openqasm3(circuit: Circuit, include_mapping: bool = True, physical_qubits: bool = False) -> str:
    """Export circuit to OpenQASM 3.0. physical_qubits=True emits $N references for ISA circuits."""
    tracker = None
    if physical_qubits:
        if not hasattr(circuit, '_tracker') or circuit._tracker is None:
            raise ValueError("physical_qubits=True requires a compiled circuit (use transpile() first)")
        tracker = circuit._tracker

    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";']
    # Emit definitions for gates not in stdgates.inc
    needed = {op.gate for op in circuit.ops} & _QASM3_GATE_DEFS.keys()
    for gate in sorted(needed, key=lambda g: g.name):
        lines.append(_QASM3_GATE_DEFS[gate])
    lines.append('')
    if not physical_qubits:
        lines.append(f'qubit[{circuit.n_qubits}] q;')
    if _needs_classical(circuit):
        lines.append(f'bit[{circuit.n_classical}] c;')
    for p in sorted(circuit.parameters, key=lambda p: p.name):
        lines.append(f'input float {p.name};')
    if include_mapping and not physical_qubits:
        _add_mapping(lines, circuit)
    lines.append('')

    qref = (lambda q: f'${tracker.logical_to_phys(q)}') if physical_qubits else (lambda q: f'q[{q}]')
    for op in circuit.ops:
        if op.gate == Gate.MEASURE:
            cb = op.classical_bit if op.classical_bit is not None else op.qubits[0]
            stmt = f'c[{cb}] = measure {qref(op.qubits[0])};'
        elif op.gate == Gate.RESET:
            stmt = f'reset {qref(op.qubits[0])};'
        else:
            stmt = _format_gate(op, physical_qubits, tracker)

        if op.condition is not None:
            bit, val = op.condition
            lines.append(f'if (c[{bit}] == {val}) {{ {stmt} }}')
        else:
            lines.append(stmt)

    return '\n'.join(lines)
