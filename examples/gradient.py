"""Train a 1-qubit circuit to produce |1⟩ using GradientDescent optimizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit import Circuit, Parameter, Z, expectation, GradientDescent, fidelity_cost
import numpy as np

# Parameterized circuit: RY(θ)|0⟩
theta = Parameter("theta")
c = Circuit(1)
c.ry(0, theta)

# Train θ so ⟨Z⟩ → -1 (i.e. |1⟩)
opt = GradientDescent(stepsize=0.3)
params = {"theta": 0.1}

for step in range(30):
    params = opt.step(params, c, Z(0))
    if step % 5 == 0:
        e = expectation(c.bind(params), Z(0))
        print(f"step {step:2d}: θ={params['theta']:.4f}, ⟨Z⟩={e:.4f}")

# θ converges to π, where RY(π)|0⟩ = |1⟩ and ⟨Z⟩ = -1
e = expectation(c.bind(params), Z(0))
target_one = np.array([0.0, 1.0])
print(f"\nfinal: θ={params['theta']:.4f} (π={np.pi:.4f}), ⟨Z⟩={e:.4f}, fidelity_cost={fidelity_cost(c.bind(params), target_one):.6f}")
