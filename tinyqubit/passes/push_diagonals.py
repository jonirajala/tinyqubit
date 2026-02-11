"""
Diagonal gate pushing pass.

Moves diagonal gates backward through commuting gates for merge opportunities.
Uses centralized commutation rules from dag.py.
"""
from ..ir import Circuit, Operation, Gate
from ..dag import DAGCircuit, commutes

DIAGONAL_1Q = {Gate.Z, Gate.S, Gate.T, Gate.SDG, Gate.TDG, Gate.RZ}


def push_diagonals(inp):
    """Push diagonal gates backward to gather for merging. Accepts Circuit or DAGCircuit."""
    from_circuit = isinstance(inp, Circuit)
    dag = DAGCircuit.from_circuit(inp) if from_circuit else inp
    ops = list(dag.topological_ops())
    changed = True
    while changed:
        changed = False
        for i in range(len(ops) - 1, -1, -1):
            if i >= len(ops) or ops[i].gate not in DIAGONAL_1Q: continue
            # Find target position, only push if passing non-diagonal
            target, passed_non_diag = i, False
            for j in range(i - 1, -1, -1):
                if not commutes(ops[i], ops[j]): break
                target = j
                if ops[j].gate not in DIAGONAL_1Q: passed_non_diag = True
            if passed_non_diag and target != i:
                ops.insert(target, ops.pop(i))
                changed = True
    result = DAGCircuit(dag.n_qubits, dag.n_classical)
    for op in ops: result.add_op(op)
    return result.to_circuit() if from_circuit else result
