"""OpenQASM export/import: to_openqasm2(), to_openqasm3(), from_openqasm2(), from_openqasm3()"""
from __future__ import annotations

import json, re, ast, operator
import numpy as np

from .ir import Circuit, Gate, Operation, Parameter


class UnsupportedGateError(Exception):
    """Gate not supported in target format."""


# Gates NOT in stdgates.inc that need explicit definitions in QASM3
_QASM3_GATE_DEFS = {
    Gate.ECR: "gate ecr a, b { rz(1.5707963267948966) a; sx a; rz(3.141592653589793) a; cx a, b; x b; }",
    Gate.RZZ: "gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }",
}
_QASM2_UNSUPPORTED = {Gate.CP: "CP", Gate.CCZ: "CCZ", Gate.ECR: "ECR", Gate.RZZ: "RZZ"}


def _format_params(op: Operation) -> str:
    if not op.params:
        return ''
    parts = [p.name if isinstance(p, Parameter) else str(p) for p in op.params]
    return f"({', '.join(parts)})"


def _format_gate(op: Operation, physical_qubits: bool = False) -> str:
    fmt = (lambda q: f'${q}') if physical_qubits else (lambda q: f'q[{q}]')
    return f'{op.gate.name.lower()}{_format_params(op)} {", ".join(fmt(q) for q in op.qubits)};'


def _needs_classical(circuit: Circuit) -> bool:
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


_QASM_GATE_MAP = {g.name.lower(): g for g in Gate}
_SYMMETRIC_2Q = frozenset({Gate.CZ, Gate.SWAP})
_PARAM_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.USub: operator.neg}
_U_GATES = {
    'u1': lambda p: [(Gate.RZ, p[0])],
    'u2': lambda p: [(Gate.RZ, p[1]), (Gate.RY, np.pi / 2), (Gate.RZ, p[0])],
    'u3': lambda p: [(Gate.RZ, p[2]), (Gate.RY, p[0]), (Gate.RZ, p[1])],
}

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
    params_decl: list[str] = []
    in_gate_def = False
    for line in qasm.split('\n'):
        line = line.split('//')[0].strip()
        if not line or line.startswith('OPENQASM') or line.startswith('include') or line.startswith('barrier'):
            continue
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
        cond_pat = r'if\s*\(\s*\w+\[(\d+)\]\s*==\s*(\d+)\s*\)\s*\{\s*(.*?)\s*\}' if v3 else r'if\s*\(\s*\w+\[(\d+)\]\s*==\s*(\d+)\s*\)\s*(.*)'
        m = re.match(cond_pat, line)
        if m:
            condition = (int(m.group(1)), int(m.group(2)))
            line = m.group(3)

        # Measure — v3 uses `c[dst] = measure qref;`
        if v3:
            m = re.match(r'\w+\[(\d+)\]\s*=\s*measure\s+(?:\$(\d+)|\w+\[(\d+)\]);', line)
            if m:
                q = int(m.group(2) if m.group(2) is not None else m.group(3))
                ops.append(Operation(Gate.MEASURE, (q,), (), int(m.group(1)), condition))
                continue
        else:
            m = re.match(r'measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\];', line)
            if m:
                ops.append(Operation(Gate.MEASURE, (int(m.group(1)),), (), int(m.group(2)), condition))
                continue

        # Reset
        m = re.match(r'reset\s+(?:\$(\d+)|\w+\[(\d+)\]);', line)
        if m:
            q = int(m.group(1) if m.group(1) is not None else m.group(2))
            ops.append(Operation(Gate.RESET, (q,), (), None, condition))
            continue

        # Gate: name(params) qubits;
        m = re.match(r'(\w+)\s*(?:\(([^)]*)\))?\s+(.+);', line)
        if not m: continue
        name, param_str, qubit_str = m.group(1), m.group(2), m.group(3)
        params = tuple(Parameter(p.strip()) if v3 and p.strip() in params_decl else _parse_param(p.strip())
                       for p in param_str.split(',')) if param_str else ()
        qubits = tuple(int(x) for x in re.findall(r'(?:\$|[\w]+\[)(\d+)\]?', qubit_str))

        nl = name.lower()
        if nl == 'id': continue

        # u1/u2/u3 → RZ/RY decomposition (global phase ignored)
        if nl in _U_GATES:
            for g, p in _U_GATES[nl](params):
                ops.append(Operation(g, qubits, (p,), None, condition))
            continue

        gate = _QASM_GATE_MAP.get(nl)
        if gate is None: raise ValueError(f"Unknown gate: {name}")
        if gate in _SYMMETRIC_2Q: qubits = (min(qubits), max(qubits))
        elif gate == Gate.CCZ: qubits = tuple(sorted(qubits))
        ops.append(Operation(gate, qubits, params, None, condition))

    # Infer n_qubits from $N physical qubit refs if no declaration (v3 only)
    if n_qubits is None and v3:
        phys_refs = re.findall(r'\$(\d+)', qasm)
        if phys_refs: n_qubits = max(int(q) for q in phys_refs) + 1
    if n_qubits is None:
        raise ValueError("No qreg declaration found" if not v3 else "No qubit[] or $N references found")
    c = Circuit(n_qubits, n_classical if n_classical is not None else n_qubits)
    c.ops = ops
    return c


def from_openqasm2(qasm: str) -> Circuit: return _from_openqasm(qasm, v3=False)
def from_openqasm3(qasm: str) -> Circuit: return _from_openqasm(qasm, v3=True)


def _op_to_dict(op: Operation) -> dict:
    d = {"gate": op.gate.name, "qubits": list(op.qubits)}
    if op.params:
        d["params"] = [{"_param": p.name, "trainable": p.trainable} if isinstance(p, Parameter) else p for p in op.params]
    if op.classical_bit is not None:
        d["classical_bit"] = op.classical_bit
    if op.condition is not None:
        d["condition"] = list(op.condition)
    return d


def _op_from_dict(d: dict) -> Operation:
    gate = Gate[d["gate"]]
    qubits = tuple(d["qubits"])
    params = tuple(Parameter(p["_param"], p.get("trainable", True)) if isinstance(p, dict) else p for p in d.get("params", ()))
    classical_bit = d.get("classical_bit")
    condition = tuple(d["condition"]) if "condition" in d else None
    return Operation(gate, qubits, params, classical_bit, condition)


def circuit_to_json(circuit: Circuit) -> str:
    data = {"n_qubits": circuit.n_qubits, "n_classical": circuit.n_classical,
            "ops": [_op_to_dict(op) for op in circuit.ops]}
    if circuit._initial_state is not None:
        data["initial_state"] = [[z.real, z.imag] for z in circuit._initial_state]
    if hasattr(circuit, '_tracker') and circuit._tracker is not None:
        t = circuit._tracker
        data["tracker"] = {"n_qubits": t.n_qubits, "initial_layout": t.initial_layout,
                           "logical_to_physical": t.logical_to_physical,
                           "physical_to_logical": t.physical_to_logical}
    return json.dumps(data)


def circuit_from_json(s: str) -> Circuit:
    data = json.loads(s)
    c = Circuit(data["n_qubits"], data.get("n_classical"))
    c.ops = [_op_from_dict(d) for d in data["ops"]]
    if "initial_state" in data:
        c._initial_state = np.array([complex(r, i) for r, i in data["initial_state"]])
    if "tracker" in data:
        from .tracker import QubitTracker
        td = data["tracker"]
        t = QubitTracker(td["n_qubits"], td.get("initial_layout"))
        t.logical_to_physical = td["logical_to_physical"]
        t.physical_to_logical = td["physical_to_logical"]
        c._tracker = t
    return c
