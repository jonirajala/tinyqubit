"""H₂ VQE: ADAPT-VQE, noise, and comparison with exact diagonalization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import (molecular_hamiltonian, hf_state, expectation, exact_diag,
                        adapt_vqe, taper, uccsd_ansatz, NoiseModel, zne)

# --- H₂: the full workflow ---
print("=== H₂ STO-3G CAS(2,2) ===\n")
H, nq, ne = molecular_hamiltonian('h2')
e_exact = exact_diag(H, nq)
e_hf = expectation(hf_state(nq, ne), H)

# ADAPT-VQE
circuit, e_vqe, history = adapt_vqe(H, nq, ne, max_iters=10, opt_steps=200, stepsize=0.1)
print("ADAPT-VQE convergence:")
for i, (e, g) in enumerate(history):
    print(f"  iter {i}: E={e:+.6f} Ha, max|grad|={g:.4f}")

# Noisy evaluation
bound = circuit.bind()
noise = NoiseModel().add_depolarizing(0.01)
e_noisy = expectation(bound, H)  # noiseless reference
e_zne = zne(bound, H, noise, scale_factors=[1, 3, 5], seed=42)

print(f"\n{'Method':<20} {'Energy (Ha)':>12} {'Error (mHa)':>12}")
print(f"{'-'*20} {'-'*12} {'-'*12}")
for name, e in [('HF', e_hf), ('ADAPT-VQE', e_vqe),
                ('ZNE (p=0.01)', e_zne), ('Exact diag', e_exact)]:
    print(f"{name:<20} {e:>+12.6f} {abs(e - e_exact)*1000:>12.2f}")

# --- LiH: CAS(4,4) with tapering ---
print("\n=== LiH STO-3G CAS(4,4) + tapering ===\n")
H_lih, nq, ne = molecular_hamiltonian('lih')
H_tap, nq_tap = taper(H_lih, nq, ne)
print(f"Before tapering: {nq} qubits, {len(H_lih.terms)} terms")
print(f"After tapering:  {nq_tap} qubits, {len(H_tap.terms)} terms")
e_exact_lih = exact_diag(H_lih, nq)
print(f"Exact energy: {e_exact_lih:+.6f} Ha")

# --- Spin-adapted UCCSD comparison ---
print("\n=== Spin-adapted UCCSD parameter reduction ===\n")
for mol in ['h2', 'lih', 'beh2', 'h2o']:
    H, nq, ne = molecular_hamiltonian(mol)
    n_full = len(uccsd_ansatz(nq, ne).parameters)
    n_spin = len(uccsd_ansatz(nq, ne, spin_adapted=True).parameters)
    print(f"  {mol:4s} ({nq}q): UCCSD={n_full:3d} params, spin-adapted={n_spin:3d} params")
