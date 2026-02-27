"""Train a 1-qubit circuit to produce |1⟩ using parameter-shift gradients."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, Parameter
from tinyqubit.simulator import probabilities
import numpy as np

# Parameterized circuit: RY(θ)|0⟩
theta = Parameter("theta")
c = Circuit(1)
c.ry(0, theta)

# Train θ so P(|1⟩) → 1.0
# Loss = -P(|1⟩), gradient via parameter-shift rule
lr = 0.3
theta_val = 0.1

for step in range(30):
    shift = np.pi / 2
    p_plus = probabilities(c.bind({"theta": theta_val + shift}))
    p_minus = probabilities(c.bind({"theta": theta_val - shift}))
    grad = (-p_plus[1] + p_minus[1]) / 2

    theta_val -= lr * grad

    p = probabilities(c.bind({"theta": theta_val}))
    if step % 5 == 0:
        print(f"step {step:2d}: θ={theta_val:.4f}, P(|1⟩)={p[1]:.4f}")

# θ converges to π, where RY(π)|0⟩ = |1⟩
print(f"\nfinal: θ={theta_val:.4f} (π={np.pi:.4f}), P(|1⟩)={p[1]:.4f}")
