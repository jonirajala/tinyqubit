"""
CX direction correction for directed-coupler hardware.

CX reversal identity: CX(a,b) = H(a) H(b) CX(b,a) H(a) H(b)
Inserted H gates get cleaned up by downstream fuse_1q + optimize passes.
"""

from ..ir import Circuit, Operation, Gate
from ..dag import DAGCircuit


def _fix_cx_direction(op: Operation, allowed: frozenset[tuple[int, int]]) -> list[Operation]:
    if op.gate != Gate.CX or op.qubits in allowed:
        return [op]
    q0, q1 = op.qubits
    h0 = Operation(Gate.H, (q0,), condition=op.condition)
    h1 = Operation(Gate.H, (q1,), condition=op.condition)
    cx = Operation(Gate.CX, (q1, q0), condition=op.condition)
    return [h0, h1, cx, h0, h1]


def fix_direction_dag(dag: DAGCircuit, target) -> DAGCircuit:
    """Reverse CX gates not matching target edge direction. No-op if target.directed is False."""
    if not target.directed:
        return dag
    result = DAGCircuit(dag.n_qubits, dag.n_classical)
    for op in dag.topological_ops():
        for new_op in _fix_cx_direction(op, target.edges):
            result.add_op(new_op)
    return result


def fix_direction(circuit: Circuit, target) -> Circuit:
    dag = DAGCircuit.from_circuit(circuit)
    return fix_direction_dag(dag, target).to_circuit()
