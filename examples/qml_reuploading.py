"""
Data re-uploading classifier on breast cancer data (Perez-Salinas et al., 2020).

Circuit: 3 layers of [angle_feature_map → CX → trainable RY]
Decision: sign(⟨Z₀⟩) → class label (+1 or -1)
Training: batch Adam on cross-entropy loss

Ref: Perez-Salinas et al., "Data re-uploading for a universal quantum classifier",
     Quantum 4, 226 (2020). arXiv:1907.02085
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import urllib.request
import numpy as np
from tinyqubit import Circuit, Parameter, cost_gradient
from tinyqubit.qml import Adam, cross_entropy_cost, predict
from tinyqubit.qml.feature_map import angle_feature_map
from tinyqubit.qml.ansatz import basic_entangler_layers

# --- Fetch breast cancer Wisconsin dataset from UCI ---
_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
_raw = urllib.request.urlopen(_UCI_URL).read().decode()
_rows = [line.split(",") for line in _raw.strip().split("\n") if line.strip()]
_features = np.array([[float(v) for v in r[2:]] for r in _rows])
_labels = np.array([1 if r[1] == "M" else -1 for r in _rows])  # malignant=+1, benign=-1

# --- Config ---
n_features = 4
n_layers = 3

# Select features, scale to [0, π]
X_raw = _features[:, :n_features]
X = (X_raw - X_raw.min(0)) / (X_raw.max(0) - X_raw.min(0)) * np.pi
y = _labels

# --- Data-reuploading circuit: encode data n_layers times interleaved with trainable layers ---
feature_params = [Parameter(f"x{i}", trainable=False) for i in range(n_features)]
qc = Circuit(n_features)
for layer in range(n_layers):
    angle_feature_map(qc, feature_params, wires=list(range(n_features)))
    basic_entangler_layers(qc, n_layers=1, prefix=f"l{layer}")

params = {p.name: 0.1 for p in qc.trainable_parameters}
opt = Adam(stepsize=0.05)

# --- Train: batch Adam on cross-entropy ---
print("=== Quantum Classifier (breast cancer, data reuploading) ===\n")
for epoch in range(80):
    grad = cost_gradient(qc, cross_entropy_cost, params, X, y)
    params = opt.step(params, grad=grad)
    if epoch % 10 == 0 or epoch == 79:
        trained = qc.bind(params)
        acc = np.mean(np.sign(predict(trained, X)) == y) * 100
        loss = cross_entropy_cost(trained, X, y)
        print(f"  epoch {epoch:2d}: acc={acc:.0f}%  loss={loss:.4f}")

trained = qc.bind(params)
preds = np.sign(predict(trained, X))
acc = np.mean(preds == y) * 100
print(f"\n  final: {int(acc)}% ({int(np.sum(preds == y))}/{len(y)})")
