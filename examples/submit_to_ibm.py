"""
Build circuit → transpile with tinyqubit → submit to IBM Quantum.

Setup: Create .env file with IBM_API_KEY="your-key-here"
Run:   python examples/submit_to_ibm.py
"""
import os, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from tinyqubit import Circuit, Gate, Target, transpile, to_openqasm2
from tinyqubit.export import to_qiskit

_GATE_MAP = {"sx": Gate.SX, "rz": Gate.RZ, "cx": Gate.CX, "cz": Gate.CZ, "ecr": Gate.ECR}
_2Q_GATES = ("cx", "cz", "ecr")


def _load_api_key() -> str:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("IBM_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"\'')
    return os.environ.get("IBM_API_KEY", "")


def _target_from_backend(backend) -> Target:
    """Build a tinyqubit Target from a Qiskit backend."""
    basis = frozenset(_GATE_MAP[n] for n in backend.target.operation_names if n in _GATE_MAP)
    edges = set()
    for name in _2Q_GATES:
        if name in backend.target.operation_names:
            for qargs in backend.target.qargs_for_operation_name(name):
                edges.add(tuple(qargs))
    return Target(n_qubits=backend.num_qubits, edges=frozenset(edges),
                  basis_gates=basis, name=backend.name, directed=True)


api_key = _load_api_key()
if not api_key:
    print("ERROR: Set IBM_API_KEY in .env file or environment")
    sys.exit(1)

# Build GHZ circuit
circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure(0).measure(1).measure(2)
print("=== Logical Circuit ===")
print(to_openqasm2(circuit))

# Connect to IBM, build target from assigned backend
print("=== Connecting to IBM Quantum ===")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key)
backend = service.backends()[0]
target = _target_from_backend(backend)
print(f"Backend: {target.name} ({target.n_qubits}Q, basis: {', '.join(g.name for g in target.basis_gates)})")

# Transpile with tinyqubit
print(f"\n=== Compiling for {target.name} ===")
compiled = transpile(circuit, target, verbosity=1)

print("\n=== Compiled Circuit (OpenQASM 2.0) ===")
print(to_openqasm2(compiled, include_mapping=False))

counts = Counter(op.gate.name for op in compiled.ops)
print(f"=== {sum(v for k, v in counts.items() if k != 'MEASURE')} gates: {', '.join(f'{g}={n}' for g, n in sorted(counts.items()))} ===")

# Submit
print("\n=== Submitting Job ===")
job = SamplerV2(backend).run([to_qiskit(compiled)], shots=1024)
print(f"Job ID: {job.job_id()}\nWaiting for results...")

result_counts = job.result()[0].data.c.get_counts()
print("\n=== Results ===")
for bitstring, count in sorted(result_counts.items()):
    print(f"  |{bitstring}⟩: {count} ({100 * count / 1024:.1f}%)")
print("\nExpected: ~50% |000⟩, ~50% |111⟩ (GHZ state)")
