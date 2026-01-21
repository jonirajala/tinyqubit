"""
Run a Bell state on IBM Quantum hardware.

STATUS: TinyQubit builds the circuit, Qiskit handles transpilation.
        Future (Phase 9+9.5): TinyQubit will transpile directly for IBM.

Setup: Create .env file with IBM_API_KEY="your-key-here"
Run:   python examples/run_on_ibm.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tinyqubit import Circuit, to_openqasm2
from tinyqubit.export import to_qiskit

# Load API key from .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("IBM_API_KEY"):
            IBM_API_KEY = line.split("=", 1)[1].strip().strip('"\'')
            break
else:
    IBM_API_KEY = os.environ.get("IBM_API_KEY", "")

if not IBM_API_KEY:
    print("ERROR: Set IBM_API_KEY in .env file or environment")
    sys.exit(1)

# Build Bell state circuit
print("=== TinyQubit Circuit ===")
circuit = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
print(to_openqasm2(circuit))

# Connect to IBM
print("=== Connecting to IBM Quantum ===")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_API_KEY)
backend = service.backends()[0]
print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

# Submit job
print("\n=== Submitting Job ===")
qc = to_qiskit(circuit)
qc_transpiled = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
print(f"Transpiled to {len(qc_transpiled.data)} gates")

job = SamplerV2(backend).run([qc_transpiled], shots=1024)
print(f"Job ID: {job.job_id()}")
print("Waiting for results...")

# Results
counts = job.result()[0].data.c.get_counts()
print("\n=== Results ===")
for bitstring, count in sorted(counts.items()):
    print(f"  |{bitstring}⟩: {count} ({100*count/1024:.1f}%)")
print("\nExpected: ~50% |00⟩, ~50% |11⟩ (Bell state)")
