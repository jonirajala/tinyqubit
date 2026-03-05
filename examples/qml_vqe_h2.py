"""
VQE for the H₂ molecule ground state energy.

Hamiltonian: 2-qubit STO-3G Jordan-Wigner at equilibrium bond length (0.735 Å).
Exact ground state energy ≈ -1.8572 Ha.
Coefficients from Kandala et al., Nature 549, 242 (2017).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, NoiseModel, expectation, resource_estimate, zne
from tinyqubit.observable import X, Y, Z, Observable
from tinyqubit.qml import Adam
from tinyqubit.qml.ansatz import strongly_entangling_layers

# H₂ Hamiltonian (STO-3G, Jordan-Wigner, R=0.735 Å)
H = (-1.0523 * Observable([(1.0, {})]) + 0.3979 * Z(0) + -0.3979 * Z(1)
     + -0.0112 * (Z(0) @ Z(1)) + 0.1809 * (X(0) @ X(1)))
exact_gs = -1.8572

# Ansatz: strongly entangling layers (2 layers, 2 qubits → 8 parameters)
ansatz = Circuit(2)
strongly_entangling_layers(ansatz, n_layers=2, prefix="w")

# --- VQE optimization ---
print("=== VQE: H₂ Ground State ===\n")
params = {p.name: 0.1 * (i - 4) for i, p in enumerate(ansatz.parameters)}
opt = Adam(stepsize=0.15)

for step in range(100):
    params = opt.step(params, ansatz, H)
    if step % 10 == 0:
        e = expectation(ansatz.bind(params), H)
        print(f"  step {step:3d}: E = {e:+.6f} Ha")

e_final = expectation(ansatz.bind(params), H)
print(f"\n  converged: E = {e_final:+.6f} Ha")
print(f"  exact:     E = {exact_gs:+.6f} Ha")
print(f"  error:     {abs(e_final - exact_gs):.2e} Ha")

# --- Error mitigation ---
bound = ansatz.bind(params)
noise = NoiseModel().add_depolarizing(0.01)
noisy = zne(bound, H, noise, scale_factors=[1], seed=42)
mitigated = zne(bound, H, noise, scale_factors=[1, 3, 5], seed=42)
print(f"\n=== Error Mitigation (ZNE) ===")
print(f"  ideal:     {e_final:+.6f} Ha")
print(f"  noisy:     {noisy:+.6f} Ha")
print(f"  ZNE:       {mitigated:+.6f} Ha")

# FTQC resource estimate for the converged ansatz
r = resource_estimate(ansatz.bind(params))
print(f"\n=== FTQC Resource Estimate ===")
print(f"  T-count: {r.t_count}, T-depth: {r.t_depth}, Cliffords: {r.clifford_count}")
print(f"  Code distance: {r.code_distance}, Physical qubits: {r.physical_qubits}")
