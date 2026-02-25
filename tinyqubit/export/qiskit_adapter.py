"""Qiskit adapter - converts tinyqubit Circuit to Qiskit QuantumCircuit."""
from __future__ import annotations


def _add_gate(qc, g, q, p, Gate):
    """Add gate to circuit, return the instruction for potential condition."""
    if g == Gate.X: return qc.x(q[0])
    elif g == Gate.Y: return qc.y(q[0])
    elif g == Gate.Z: return qc.z(q[0])
    elif g == Gate.H: return qc.h(q[0])
    elif g == Gate.S: return qc.s(q[0])
    elif g == Gate.T: return qc.t(q[0])
    elif g == Gate.SDG: return qc.sdg(q[0])
    elif g == Gate.TDG: return qc.tdg(q[0])
    elif g == Gate.RX: return qc.rx(p[0], q[0])
    elif g == Gate.RY: return qc.ry(p[0], q[0])
    elif g == Gate.RZ: return qc.rz(p[0], q[0])
    elif g == Gate.CX: return qc.cx(q[0], q[1])
    elif g == Gate.CZ: return qc.cz(q[0], q[1])
    elif g == Gate.CP: return qc.cp(p[0], q[0], q[1])
    elif g == Gate.SWAP: return qc.swap(q[0], q[1])
    elif g == Gate.CCX: return qc.ccx(q[0], q[1], q[2])
    elif g == Gate.CCZ: return qc.ccz(q[0], q[1], q[2])
    elif g == Gate.RESET: return qc.reset(q[0])
    else: raise NotImplementedError(f"Unsupported gate for Qiskit export: {g.name}")


def to_qiskit(circuit) -> "QuantumCircuit":
    """Convert a tinyqubit Circuit to a Qiskit QuantumCircuit."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter as QiskitParameter
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    from ..ir import Gate, Parameter

    # Map tinyqubit Parameters to Qiskit Parameters
    _param_map = {}
    for p in circuit.parameters:
        _param_map[p.name] = QiskitParameter(p.name)

    needs_clbits = any(op.gate == Gate.MEASURE or op.condition is not None for op in circuit.ops)
    n_clbits = circuit.n_classical if needs_clbits else 0
    qc = QuantumCircuit(circuit.n_qubits, n_clbits)

    for op in circuit.ops:
        g, q = op.gate, op.qubits
        p = tuple(_param_map[v.name] if isinstance(v, Parameter) else v for v in op.params)

        if g == Gate.MEASURE:
            cbit = op.classical_bit if op.classical_bit is not None else q[0]
            qc.measure(q[0], cbit)
        elif op.condition is not None:
            # Use if_test context manager for conditional ops (Qiskit 1.0+ API)
            bit, val = op.condition
            with qc.if_test((qc.clbits[bit], val)):
                _add_gate(qc, g, q, p, Gate)
        else:
            _add_gate(qc, g, q, p, Gate)

    return qc
