"""AWS Braket backend adapter.
    Example device ARNs:
        - IonQ Harmony: "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"
        - IonQ Aria: "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
        - Rigetti Ankaa: "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2"
        - IQM Garnet: "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
        - Local simulator: "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
"""
from __future__ import annotations

import numpy as np

from ..qasm import to_openqasm3


def submit_to_braket(circuit, device_arn: str, s3_bucket: str, s3_prefix: str = "tinyqubit-results", shots: int = 1024, target=None):
    """Submit circuit to AWS Braket hardware, returns AwsQuantumTask."""
    if target is not None:
        errors = target.validate(circuit)
        if errors:
            raise ValueError("Circuit validation failed:\n" + "\n".join(errors))
    try:
        from braket.aws import AwsDevice
        from braket.ir.openqasm import Program
    except ImportError:
        raise ImportError("amazon-braket-sdk required: pip install amazon-braket-sdk")

    program = Program(source=to_openqasm3(circuit, include_mapping=False))
    return AwsDevice(device_arn).run(program, (s3_bucket, s3_prefix), shots=shots)


def get_braket_results(task, timeout: float = 600, n_qubits: int | None = None, tracker=None) -> dict[str, int]:
    """Get measurement counts from Braket task. Returns {"00": 512, "11": 512}."""
    counts = dict(task.result().measurement_counts)
    if n_qubits is None or tracker is None:
        return counts
    result = {}
    for bitstring, count in counts.items():
        bits = list(bitstring)
        phys = list(bits)
        for p in range(n_qubits):
            bits[tracker.phys_to_logical(p)] = phys[p]
        key = ''.join(bits)
        result[key] = result.get(key, 0) + count
    return result


class BraketBackend:
    """AWS Braket backend. Use as circuit.backend = BraketBackend(...)."""
    def __init__(self, device_arn: str, s3_bucket: str, s3_prefix: str = "tinyqubit-results",
                 target=None, shots: int = 4096, preset: str = "fast"):
        self.device_arn, self.s3_bucket, self.s3_prefix = device_arn, s3_bucket, s3_prefix
        self.target, self.shots, self.preset = target, shots, preset

    def __call__(self, circuit, observable) -> float:
        from ..ir import Circuit
        from ..compile import transpile
        result = 0.0
        for coeff, paulis in observable.terms:
            if not paulis:
                result += coeff
                continue
            rc = Circuit(circuit.n_qubits)
            rc._initial_state, rc.ops = circuit._initial_state, list(circuit.ops)
            for q, p in paulis.items():
                if p == 'X': rc.h(q)
                elif p == 'Y': rc.rz(q, -np.pi / 2); rc.h(q)
            rc.measure_all()
            compiled = transpile(rc, self.target, preset=self.preset) if self.target else rc
            task = submit_to_braket(compiled, self.device_arn, self.s3_bucket, self.s3_prefix, self.shots)
            counts = get_braket_results(task, n_qubits=circuit.n_qubits)
            ev = 0.0
            for bs, cnt in counts.items():
                parity = sum(int(bs[q]) for q in paulis)
                ev += (-1) ** (parity % 2) * cnt
            result += coeff * ev / self.shots
        return result
