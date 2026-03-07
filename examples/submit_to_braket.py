"""
Build circuit → transpile with tinyqubit → submit to AWS Braket.

Setup:
    pip install amazon-braket-sdk
    AWS credentials configured (aws configure or env vars)

Run:   python examples/submit_to_braket.py
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, transpile, to_openqasm3, IQM_GARNET
from tinyqubit.hardware import submit_to_braket, get_braket_results

# Config — change these for your setup
DEVICE_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
S3_BUCKET = "my-braket-results"  # your S3 bucket name
S3_PREFIX = "tinyqubit-results"
SHOTS = 1024

# Build GHZ circuit
circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure(0).measure(1).measure(2)
print("=== Logical Circuit ===")
print(to_openqasm3(circuit))

# Transpile for IQM Garnet (20Q square lattice, CZ basis)
target = IQM_GARNET
print(f"\n=== Compiling for {target.name} ===")
compiled = transpile(circuit, target, preset="quality", verbosity=1)

print("\n=== Compiled Circuit (OpenQASM 3.0) ===")
print(to_openqasm3(compiled, include_mapping=False))

gate_counts = Counter(op.gate.name for op in compiled.ops if op.gate.name != "MEASURE")
print(f"=== {sum(gate_counts.values())} gates: {', '.join(f'{g}={n}' for g, n in sorted(gate_counts.items()))} ===")

# Submit via Braket SDK
print("\n=== Submitting Job ===")
task = submit_to_braket(compiled, device_arn=DEVICE_ARN, s3_bucket=S3_BUCKET, s3_prefix=S3_PREFIX, shots=SHOTS)
print(f"Task ARN: {task.id}\nWaiting for results...")

counts = get_braket_results(task, n_qubits=compiled.n_qubits)
print("\n=== Results ===")
total = sum(counts.values())
for bitstring, count in sorted(counts.items()):
    print(f"  |{bitstring}⟩: {count} ({100 * count / total:.1f}%)")
print("\nExpected: ~50% |000⟩, ~50% |111⟩ (GHZ state)")
