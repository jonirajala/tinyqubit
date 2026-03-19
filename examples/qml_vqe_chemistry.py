"""VQE for H₂ + LiH using molecular_hamiltonian + UCCSD ansatz."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import molecular_hamiltonian, uccsd_ansatz, expectation
from tinyqubit.qml.optim import Adam

# --- Single-point VQE: H₂ at equilibrium ---
H, n_qubits, n_electrons = molecular_hamiltonian('h2')
ansatz = uccsd_ansatz(n_qubits, n_electrons)
ansatz.init_params(seed=42)
opt = Adam(stepsize=0.1)

print("=== VQE: H₂ at R=0.735 Å ===\n")
for step in range(200):
    _, e = opt.step_and_cost(ansatz, H)
    if step % 20 == 0:
        print(f"  step {step:3d}: E = {e:+.6f} Ha")

print(f"\n  final:  {e:+.6f} Ha")
print(f"  exact:  -1.1386 Ha (FCI)")
print(f"  error:  {abs(e - (-1.1386)):.2e} Ha")

# --- PES scan: H₂ dissociation curve ---
print("\n=== H₂ PES scan (HF energy) ===\n")
from tinyqubit import hf_state
for R in [0.5, 0.6, 0.735, 0.8, 1.0, 1.5, 2.0, 2.5]:
    H_R, nq, ne = molecular_hamiltonian('h2', bond_length=R)
    e_hf = expectation(hf_state(nq, ne), H_R)
    print(f"  R={R:.3f} Å: E_HF = {e_hf:+.6f} Ha")

# --- LiH at equilibrium ---
print("\n=== LiH CAS(2,2) at R=1.546 Å ===\n")
H_lih, nq, ne = molecular_hamiltonian('lih')
ansatz_lih = uccsd_ansatz(nq, ne)
ansatz_lih.init_params(seed=42)
opt_lih = Adam(stepsize=0.1)
for step in range(200):
    _, e = opt_lih.step_and_cost(ansatz_lih, H_lih)
    if step % 50 == 0:
        print(f"  step {step:3d}: E = {e:+.6f} Ha")
print(f"\n  final:  {e:+.6f} Ha")
print(f"  exact:  -7.8634 Ha (FCI/CAS)")
