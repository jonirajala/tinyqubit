"""Qiskit adapter - converts tinyqubit Circuit to Qiskit QuantumCircuit."""
from __future__ import annotations


def to_qiskit(circuit) -> "QuantumCircuit":
    """Convert a tinyqubit Circuit to a Qiskit QuantumCircuit."""
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit required: pip install qiskit")

    from ..ir import Gate

    has_measure = any(op.gate == Gate.MEASURE for op in circuit.ops)
    qc = QuantumCircuit(circuit.n_qubits, circuit.n_qubits if has_measure else 0)

    for op in circuit.ops:
        g, q, p = op.gate, op.qubits, op.params
        if g == Gate.X: qc.x(q[0])
        elif g == Gate.Y: qc.y(q[0])
        elif g == Gate.Z: qc.z(q[0])
        elif g == Gate.H: qc.h(q[0])
        elif g == Gate.S: qc.s(q[0])
        elif g == Gate.T: qc.t(q[0])
        elif g == Gate.SDG: qc.sdg(q[0])
        elif g == Gate.TDG: qc.tdg(q[0])
        elif g == Gate.RX: qc.rx(p[0], q[0])
        elif g == Gate.RY: qc.ry(p[0], q[0])
        elif g == Gate.RZ: qc.rz(p[0], q[0])
        elif g == Gate.CX: qc.cx(q[0], q[1])
        elif g == Gate.CZ: qc.cz(q[0], q[1])
        elif g == Gate.CP: qc.cp(p[0], q[0], q[1])
        elif g == Gate.SWAP: qc.swap(q[0], q[1])
        elif g == Gate.MEASURE: qc.measure(q[0], q[0])

    return qc
