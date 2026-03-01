"""
Build circuit → transpile with tinyqubit → submit to IBM Quantum (zero vendor SDK).

Setup: Create .env file with:
    IBM_API_KEY="your-key-here"
    IBM_CRN="crn:v1:bluemix:public:quantum-computing:..."  (optional, auto-discovered)

Run:   python examples/submit_to_ibm.py
"""
import os, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, transpile, to_openqasm3, submit_ibm, wait_ibm
from tinyqubit.export.backends.ibm_native import list_ibm_backends, ibm_target


def _load_env() -> dict[str, str]:
    """Load key=value pairs from .env file into environment."""
    env_file = Path(__file__).parent.parent / ".env"
    vals = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"\'')
    return vals


env = _load_env()
api_key = env.get("IBM_API_KEY", "") or os.environ.get("IBM_API_KEY", "")
crn = env.get("IBM_CRN") or os.environ.get("IBM_CRN")
if not api_key:
    print("ERROR: Set IBM_API_KEY in .env file or environment")
    sys.exit(1)

# Discover available backends
print("=== Discovering Backends ===")
backends = list_ibm_backends(api_key=api_key, crn=crn)
for b in backends:
    print(f"  {b['name']:20s} {b['n_qubits']}Q  ({b['status']})")
if not backends:
    print("ERROR: No backends available")
    sys.exit(1)

# Query real coupling map + basis gates from IBM
backend_name = backends[0]["name"]
print(f"\n=== Fetching target config for {backend_name} ===")
target = ibm_target(backend_name, api_key=api_key, crn=crn)
print(f"  {target.n_qubits}Q, basis: {', '.join(g.name for g in target.basis_gates)}, {len(target.edges)} edges")

# Build GHZ circuit
circuit = Circuit(3).h(0).cx(0, 1).cx(1, 2).measure(0).measure(1).measure(2)
print("\n=== Logical Circuit ===")
print(to_openqasm3(circuit))

# Transpile with tinyqubit
print(f"\n=== Compiling for {target.name} ===")
compiled = transpile(circuit, target, verbosity=1)

print("\n=== Compiled Circuit (OpenQASM 3.0 — physical qubits) ===")
print(to_openqasm3(compiled, include_mapping=False, physical_qubits=True))

gate_counts = Counter(op.gate.name for op in compiled.ops if op.gate.name != "MEASURE")
print(f"=== {sum(gate_counts.values())} gates: {', '.join(f'{g}={n}' for g, n in sorted(gate_counts.items()))} ===")

# Submit via native REST
print("\n=== Submitting Job ===")
job = submit_ibm(compiled, backend=backend_name, shots=1024, api_key=api_key, crn=crn)
print(f"Job ID: {job.job_id}\nWaiting for results...")

result_counts = wait_ibm(job, timeout=600)
print("\n=== Results ===")
total = sum(result_counts.values())
for bitstring, count in sorted(result_counts.items()):
    print(f"  |{bitstring}⟩: {count} ({100 * count / total:.1f}%)")
print("\nExpected: ~50% |000⟩, ~50% |111⟩ (GHZ state)")
