"""OpenQASM export/import: to_openqasm2(), to_openqasm3(), from_openqasm2(), from_openqasm3()"""
from __future__ import annotations

import re, ast, operator
import numpy as np

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


def _format_gate(op: Operation, physical_qubits: bool = False) -> str:
    fmt = (lambda q: f'${q}') if physical_qubits else (lambda q: f'q[{q}]')
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
    if physical_qubits and (not hasattr(circuit, '_tracker') or circuit._tracker is None):
        raise ValueError("physical_qubits=True requires a compiled circuit (use transpile() first)")

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

    qref = (lambda q: f'${q}') if physical_qubits else (lambda q: f'q[{q}]')
    for op in circuit.ops:
        if op.gate == Gate.MEASURE:
            cb = op.classical_bit if op.classical_bit is not None else op.qubits[0]
            stmt = f'c[{cb}] = measure {qref(op.qubits[0])};'
        elif op.gate == Gate.RESET:
            stmt = f'reset {qref(op.qubits[0])};'
        else:
            stmt = _format_gate(op, physical_qubits)

        if op.condition is not None:
            bit, val = op.condition
            lines.append(f'if (c[{bit}] == {val}) {{ {stmt} }}')
        else:
            lines.append(stmt)

    return '\n'.join(lines)


# --- QASM Import ---

_QASM_GATE_MAP = {g.name.lower(): g for g in Gate}
_SYMMETRIC_2Q = frozenset({Gate.CZ, Gate.SWAP})
_PARAM_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval_param(node) -> float:
    if isinstance(node, ast.Constant): return float(node.value)
    if isinstance(node, ast.Name) and node.id == 'pi': return np.pi
    if isinstance(node, ast.UnaryOp): return _PARAM_OPS[type(node.op)](_eval_param(node.operand))
    if isinstance(node, ast.BinOp): return _PARAM_OPS[type(node.op)](_eval_param(node.left), _eval_param(node.right))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _parse_param(s: str) -> float:
    try: return float(s)
    except ValueError: return _eval_param(ast.parse(s.strip(), mode='eval').body)


def _from_openqasm(qasm: str, v3: bool = False) -> Circuit:
    """Unified OpenQASM 2.0/3.0 parser. v3=True enables QASM3 syntax."""
    n_qubits = n_classical = None
    ops = []
    params_decl: list[str] = []  # QASM3 `input float name;` declarations
    max_phys_qubit = -1  # track physical qubit refs for QASM3
    in_gate_def = False
    for line in qasm.split('\n'):
        line = line.split('//')[0].strip()
        if not line or line.startswith('OPENQASM') or line.startswith('include') or line.startswith('barrier'):
            continue
        # Skip custom gate definitions (single-line or multi-line)
        if line.startswith('gate '):
            if '}' not in line: in_gate_def = True
            continue
        if in_gate_def:
            if '}' in line: in_gate_def = False
            continue

        # Register declarations — v2 vs v3 syntax
        if v3:
            m = re.match(r'qubit\[(\d+)\]\s+\w+;', line)
            if m: n_qubits = int(m.group(1)); continue
            m = re.match(r'bit\[(\d+)\]\s+\w+;', line)
            if m: n_classical = (n_classical or 0) + int(m.group(1)); continue
            m = re.match(r'input\s+float\s+(\w+);', line)
            if m: params_decl.append(m.group(1)); continue
        else:
            m = re.match(r'qreg\s+\w+\[(\d+)\];', line)
            if m: n_qubits = int(m.group(1)); continue
            m = re.match(r'creg\s+\w+\[(\d+)\];', line)
            if m: n_classical = (n_classical or 0) + int(m.group(1)); continue

        # Peel off conditional prefix
        condition = None
        if v3:
            m = re.match(r'if\s*\(\s*\w+\[(\d+)\]\s*==\s*(\d+)\s*\)\s*\{\s*(.*?)\s*\}', line)
        else:
            m = re.match(r'if\s*\(\s*\w+\[(\d+)\]\s*==\s*(\d+)\s*\)\s*(.*)', line)
        if m:
            condition = (int(m.group(1)), int(m.group(2)))
            line = m.group(3)

        # Measure — v3 uses `c[dst] = measure qref;`
        if v3:
            m = re.match(r'\w+\[(\d+)\]\s*=\s*measure\s+(?:\$(\d+)|\w+\[(\d+)\]);', line)
            if m:
                q = int(m.group(2) if m.group(2) is not None else m.group(3))
                if m.group(2) is not None: max_phys_qubit = max(max_phys_qubit, q)
                ops.append(Operation(Gate.MEASURE, (q,), (), int(m.group(1)), condition))
                continue
        else:
            m = re.match(r'measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\];', line)
            if m:
                ops.append(Operation(Gate.MEASURE, (int(m.group(1)),), (), int(m.group(2)), condition))
                continue

        # Reset — v3 regex handles both $N and q[N]
        m = re.match(r'reset\s+(?:\$(\d+)|\w+\[(\d+)\]);', line)
        if m:
            q = int(m.group(1) if m.group(1) is not None else m.group(2))
            if v3 and m.group(1) is not None: max_phys_qubit = max(max_phys_qubit, q)
            ops.append(Operation(Gate.RESET, (q,), (), None, condition))
            continue

        # Gate: name(params) qubits;
        m = re.match(r'(\w+)\s*(?:\(([^)]*)\))?\s+(.+);', line)
        if not m: continue
        name, param_str, qubit_str = m.group(1), m.group(2), m.group(3)

        # Parse params — in v3, tokens matching params_decl become Parameter objects
        if param_str:
            parts = []
            for p in param_str.split(','):
                p = p.strip()
                if v3 and p in params_decl:
                    parts.append(Parameter(p))
                else:
                    parts.append(_parse_param(p))
            params = tuple(parts)
        else:
            params = ()

        # Extract qubit indices — regex handles both $N and q[N]
        qubits = tuple(int(x) for x in re.findall(r'(?:\$|[\w]+\[)(\d+)\]?', qubit_str))
        if v3:
            for tok in re.findall(r'\$(\d+)', qubit_str):
                max_phys_qubit = max(max_phys_qubit, int(tok))

        nl = name.lower()
        if nl == 'id': continue  # identity gate — skip

        # u1/u2/u3 → decompose into RZ/RY sequence (global phase ignored)
        if nl == 'u1':
            ops.append(Operation(Gate.RZ, qubits, (params[0],), None, condition)); continue
        if nl == 'u2':
            phi, lam = params
            for g, p in [(Gate.RZ, lam), (Gate.RY, np.pi / 2), (Gate.RZ, phi)]:
                ops.append(Operation(g, qubits, (p,), None, condition))
            continue
        if nl == 'u3':
            theta, phi, lam = params
            for g, p in [(Gate.RZ, lam), (Gate.RY, theta), (Gate.RZ, phi)]:
                ops.append(Operation(g, qubits, (p,), None, condition))
            continue

        gate = _QASM_GATE_MAP.get(nl)
        if gate is None: raise ValueError(f"Unknown gate: {name}")

        # Canonicalize symmetric gates
        if gate in _SYMMETRIC_2Q: qubits = (min(qubits), max(qubits))
        elif gate == Gate.CCZ: qubits = tuple(sorted(qubits))

        ops.append(Operation(gate, qubits, params, None, condition))

    # Infer n_qubits from physical qubit refs if no qubit[] declaration (v3 only)
    if n_qubits is None and v3 and max_phys_qubit >= 0:
        n_qubits = max_phys_qubit + 1
    if n_qubits is None:
        raise ValueError("No qreg declaration found" if not v3 else "No qubit[] declaration or physical qubit ($N) references found")
    c = Circuit(n_qubits, n_classical if n_classical is not None else n_qubits)
    c.ops = ops
    return c


def from_openqasm2(qasm: str) -> Circuit:
    """Parse an OpenQASM 2.0 string into a Circuit."""
    return _from_openqasm(qasm, v3=False)


def from_openqasm3(qasm: str) -> Circuit:
    """Parse an OpenQASM 3.0 string into a Circuit."""
    return _from_openqasm(qasm, v3=True)
