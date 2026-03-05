"""Lossless JSON serialization for Circuit objects."""
from __future__ import annotations

import json
import numpy as np
from .ir import Circuit, Gate, Operation, Parameter


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
