"""
VQE on real hardware — H₂ ground state via IBM Quantum.

Same Hamiltonian as qml_vqe_h2.py, but expectation values estimated from
hardware measurement counts. Parameter-shift gradients, Adam optimizer.

Setup: Create .env file with:
    IBM_API_KEY="your-key-here"
    IBM_CRN="crn:v1:bluemix:public:quantum-computing:..."  (optional, auto-discovered)

Run:   python examples/qml_vqe_hardware.py

NOTE: Each VQE step submits 2 * n_params * n_pauli_terms jobs to the queue.
      With 4 parameters and 4 Hamiltonian terms, that's ~32 jobs/step.
      Expect ~5-10 min per step depending on queue wait times.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, expectation, IBMBackend
from tinyqubit.measurement.observable import I, X, Y, Z
from tinyqubit.qml.optim import Adam, parameter_shift_gradient, SPSA
from tinyqubit.qml.layers import strongly_entangling_layers
from tinyqubit.hardware.ibm_native import ibm_target, list_ibm_backends


def _load_env() -> dict[str, str]:
    env_file = Path(__file__).parent.parent / ".env"
    vals = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"\'')
    return vals


# --- Credentials ---
env = _load_env()
api_key = env.get("IBM_API_KEY", "") or os.environ.get("IBM_API_KEY", "")
crn = env.get("IBM_CRN") or os.environ.get("IBM_CRN")
if not api_key:
    print("ERROR: Set IBM_API_KEY in .env file or environment")
    sys.exit(1)

# --- H₂ Hamiltonian (STO-3G, Jordan-Wigner, R=0.735 Å) ---
H = (-1.0523 * I() + 0.3979 * Z(0) + -0.3979 * Z(1)
     + -0.0112 * (Z(0) @ Z(1)) + 0.1809 * (X(0) @ X(1)))
exact_gs = -1.8572

# --- Backend ---
backends = list_ibm_backends(api_key=api_key, crn=crn)
if not backends: print("ERROR: No backends available"); sys.exit(1)
backend_name = backends[0]["name"]
print(f"=== VQE on Hardware: H₂ Ground State ({backend_name}) ===\n")

target = ibm_target(backend_name, api_key=api_key, crn=crn, calibration=True)
print(f"  target: {target.n_qubits}Q, basis: {', '.join(g.name for g in target.basis_gates)}")

# --- Ansatz: 2 qubits, 1 layer (4 parameters — keeps job count manageable) ---
ansatz = strongly_entangling_layers(2, n_layers=1, prefix="w")
ansatz.init_params(seed=0)
ansatz.backend = IBMBackend(backend_name, target=target, shots=4096, api_key=api_key, crn=crn)
opt = SPSA(stepsize=0.3, perturbation=0.2, seed=0)
n_steps = 10

print(f"  ansatz: {len(ansatz.parameters)} parameters, {n_steps} steps, 4096 shots/circuit")
print(f"  exact ground state: {exact_gs:+.4f} Ha\n")

# --- VQE loop ---
for step in range(n_steps):
    _, e = opt.step_and_cost(ansatz, H)
    print(f"  step {step:2d}: E = {e:+.6f} Ha  (error: {abs(e - exact_gs):.4f})")

print(f"\n  exact: {exact_gs:+.6f} Ha")
