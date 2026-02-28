"""
VQE for 2-qubit transverse-field Ising model with landscape analysis.

Hamiltonian: H = -Z₀Z₁ + 0.5(X₀ + X₁)
Exact ground state energy: -√2 ≈ -1.4142

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import (
    Circuit, expectation, Adam,
    expectation_sweep, gradient_landscape,
)
from tinyqubit.observable import X, Y, Z
from tinyqubit.ansatz import basic_entangler_layers

# Transverse-field Ising model on 2 qubits
H = -1.0 * (Z(0) @ Z(1)) + 0.5 * X(0) + 0.5 * X(1)
exact_gs = -np.sqrt(2)

# Hardware-efficient ansatz: RY layer → CX entangler → RY layer
ansatz = Circuit(2)
basic_entangler_layers(ansatz, n_layers=2, prefix="w")

ansatz.draw()

# --- VQE optimization via Adam ---
print("\n=== VQE Optimization ===")
params = {"w_0_0": 0.5, "w_0_1": -0.3, "w_1_0": 0.8, "w_1_1": -0.2}
opt = Adam(stepsize=0.15)

for step in range(80):
    params = opt.step(params, ansatz, H)
    if step % 10 == 0:
        e = expectation(ansatz.bind(params), H)
        print(f"  step {step:2d}: E = {e:+.6f}")

e_final = expectation(ansatz.bind(params), H)
print(f"\n  converged: E = {e_final:+.6f}")
print(f"  exact:     E = {exact_gs:+.6f}")
print(f"  error:     {abs(e_final - exact_gs):.2e}")

# --- Sweep parameter 'w_0_0' around optimum ---
print("\n=== Energy vs parameter 'w_0_0' ===")
sweep_vals = np.linspace(params["w_0_0"] - np.pi, params["w_0_0"] + np.pi, 21)
energies = expectation_sweep(ansatz, "w_0_0", sweep_vals, H, base_values=params)

lo, hi = energies.min(), energies.max()
for v, e in zip(sweep_vals, energies):
    bar_len = int((e - lo) / (hi - lo + 1e-10) * 30)
    print(f"  w_0_0={v:+5.2f}  E={e:+.3f}  {'█' * bar_len}")

# --- 2D energy landscape over (w_0_0, w_1_0) ---
print("\n=== Energy Landscape (w_0_0 x w_1_0) ===")
n = 20
landscape = gradient_landscape(
    ansatz, ["w_0_0", "w_1_0"], H, params, n_points=n,
    ranges=[(params["w_0_0"] - np.pi, params["w_0_0"] + np.pi),
            (params["w_1_0"] - np.pi, params["w_1_0"] + np.pi)],
)

levels = " ░▒▓█"
lo, hi = landscape.min(), landscape.max()
for row in landscape:
    line = ""
    for val in row:
        idx = min(int((val - lo) / (hi - lo + 1e-10) * len(levels)), len(levels) - 1)
        line += levels[idx] * 2
    print(f"  {line}")
print(f"  (' '=E_min={lo:+.3f},  '█'=E_max={hi:+.3f})")
print(f"  rows=w_0_0, cols=w_1_0")
