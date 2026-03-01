"""AWS Braket backend adapter.
    Example device ARNs:
        - IonQ Harmony: "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"
        - IonQ Aria: "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
        - Rigetti Ankaa: "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2"
        - IQM Garnet: "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
        - Local simulator: "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
"""
from __future__ import annotations

from ..qasm import to_openqasm3


def submit_to_braket(circuit, device_arn: str, s3_bucket: str, s3_prefix: str = "tinyqubit-results", shots: int = 1024, target=None):
    """Submit circuit to AWS Braket hardware, returns AwsQuantumTask."""
    if target is not None:
        from ...target import validate
        errors = validate(circuit, target)
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
    if n_qubits is None:
        return counts
    from . import _normalize_counts
    return _normalize_counts(counts, n_qubits, reverse_bits=False, tracker=tracker)
