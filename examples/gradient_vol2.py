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
    Circuit, Parameter, expectation, adjoint_gradient,
    expectation_sweep, gradient_landscape,
)
from tinyqubit.observable import X, Y, Z

# Transverse-field Ising model on 2 qubits
H = -1.0 * (Z(0) @ Z(1)) + 0.5 * X(0) + 0.5 * X(1)
exact_gs = -np.sqrt(2)

# Hardware-efficient ansatz: RY layer → CX entangler → RY layer
a, b, c, d = Parameter("a"), Parameter("b"), Parameter("c"), Parameter("d")
ansatz = Circuit(2)
ansatz.ry(0, a).ry(1, b)
ansatz.cx(0, 1)
ansatz.ry(0, c).ry(1, d)

ansatz.draw()

# --- VQE optimization via adjoint gradient descent ---
print("\n=== VQE Optimization ===")
params = {"a": 0.5, "b": -0.3, "c": 0.8, "d": -0.2}
lr = 0.4

for step in range(50):
    grad = adjoint_gradient(ansatz, H, params)
    for k in params:
        params[k] -= lr * grad[k]
    if step % 10 == 0:
        e = expectation(ansatz.bind(params), H)
        print(f"  step {step:2d}: E = {e:+.6f}")

e_final = expectation(ansatz.bind(params), H)
print(f"\n  converged: E = {e_final:+.6f}")
print(f"  exact:     E = {exact_gs:+.6f}")
print(f"  error:     {abs(e_final - exact_gs):.2e}")

# --- Sweep parameter 'a' around optimum ---
print("\n=== Energy vs parameter 'a' ===")
sweep_vals = np.linspace(params["a"] - np.pi, params["a"] + np.pi, 21)
energies = expectation_sweep(ansatz, "a", sweep_vals, H, base_values=params)

lo, hi = energies.min(), energies.max()
for v, e in zip(sweep_vals, energies):
    bar_len = int((e - lo) / (hi - lo + 1e-10) * 30)
    print(f"  a={v:+5.2f}  E={e:+.3f}  {'█' * bar_len}")

# --- 2D energy landscape over (a, c) ---
print("\n=== Energy Landscape (a × c) ===")
n = 20
landscape = gradient_landscape(
    ansatz, ["a", "c"], H, params, n_points=n,
    ranges=[(params["a"] - np.pi, params["a"] + np.pi),
            (params["c"] - np.pi, params["c"] + np.pi)],
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
print(f"  rows=a, cols=c")
