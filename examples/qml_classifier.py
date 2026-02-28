"""
2-class classifier on half-moon data using data reuploading.

Circuit: 3 layers of [angle_feature_map → CX → trainable RY]
Decision: sign(⟨Z₀⟩) → class label (+1 or -1)
Training: online Adam, minimize -yᵢ·⟨Z₀⟩ per sample
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, Parameter, expectation, Adam, cross_entropy_cost
from tinyqubit.observable import Z
from tinyqubit.feature_map import angle_feature_map
from tinyqubit.ansatz import basic_entangler_layers


# --- Half-moon dataset ---
def make_moons(n, noise=0.1, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, np.pi, n)
    x1 = np.column_stack([np.cos(t), np.sin(t)]) + rng.randn(n, 2) * noise
    x2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5]) + rng.randn(n, 2) * noise
    return np.vstack([x1, x2]), np.concatenate([np.ones(n), -np.ones(n)])

X, y = make_moons(20)
X = (X - X.min(0)) / (X.max(0) - X.min(0)) * np.pi
y_01 = ((y + 1) / 2).astype(int)  # {-1,+1} → {0,1} for cross-entropy

# --- Data-reuploading circuit: encode data 3 times interleaved with trainable layers ---
x0, x1 = Parameter("x0"), Parameter("x1")
qc = Circuit(2)
for layer in range(3):
    angle_feature_map(qc, [x0, x1], wires=[0, 1])
    basic_entangler_layers(qc, n_layers=1, prefix=f"l{layer}")

obs = Z(0)
params = {f"l{l}_{0}_{w}": 0.5 * (-1) ** (l * 2 + w) for l in range(3) for w in range(2)}
opt = Adam(stepsize=0.05)

# --- Train: online Adam, one step per sample ---
print("=== Quantum Classifier (half-moons, data reuploading) ===\n")
for epoch in range(20):
    order = np.random.RandomState(epoch).permutation(len(y))
    for i in order:
        data_bound = qc.bind({"x0": X[i, 0], "x1": X[i, 1]})
        params = opt.step(params, data_bound, -y[i] * obs)

    trained = qc.bind(params)
    preds = [np.sign(expectation(trained.bind({"x0": xi[0], "x1": xi[1]}), obs)) for xi in X]
    acc = np.mean([p == yi for p, yi in zip(preds, y)]) * 100
    if epoch % 4 == 0 or epoch == 19:
        loss = cross_entropy_cost(trained, X, y_01, obs)
        print(f"  epoch {epoch:2d}: acc={acc:.0f}%  loss={loss:.4f}")

print(f"\n  final: {int(acc)}% ({sum(p == yi for p, yi in zip(preds, y))}/{len(y)})")
