"""Train a 1-qubit circuit to produce |1⟩ using parameter-shift gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, Parameter, Z, expectation
import numpy as np

# Parameterized circuit: RY(θ)|0⟩
theta = Parameter("theta")
c = Circuit(1)
c.ry(0, theta)

# Train θ so ⟨Z⟩ → -1 (i.e. |1⟩)
# Loss = ⟨Z⟩, gradient via parameter-shift rule
lr = 0.3
theta_val = 0.1

for step in range(30):
    shift = np.pi / 2
    e_plus = expectation(c.bind({"theta": theta_val + shift}), Z(0))
    e_minus = expectation(c.bind({"theta": theta_val - shift}), Z(0))
    grad = (e_plus - e_minus) / 2

    theta_val -= lr * grad

    e = expectation(c.bind({"theta": theta_val}), Z(0))
    if step % 5 == 0:
        print(f"step {step:2d}: θ={theta_val:.4f}, ⟨Z⟩={e:.4f}")

# θ converges to π, where RY(π)|0⟩ = |1⟩ and ⟨Z⟩ = -1
print(f"\nfinal: θ={theta_val:.4f} (π={np.pi:.4f}), ⟨Z⟩={e:.4f}")
