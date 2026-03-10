"""
Variational Quantum Classifier (VQC) on Iris (setosa vs versicolor).

Circuit: angle encoding → strongly entangling layers (2 layers, 4 qubits).
Decision: sign(⟨Z₀⟩) → class label.
Training: Adam on MSE loss with parameter-shift gradients.

Ref: Schuld, Bocharov, Svore, Wiebe, "Circuit-centric quantum classifiers",
     Phys. Rev. A 101, 032308 (2020). arXiv:1804.00633
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import urllib.request
import numpy as np
from tinyqubit import Circuit
from tinyqubit.qml.optim import Adam, cost_gradient
from tinyqubit.qml.loss import mse_cost, predict
from tinyqubit.qml.layers import angle_feature_map, strongly_entangling_layers


# --- Fetch iris dataset from UCI ML Repository ---
_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
_raw = urllib.request.urlopen(_UCI_URL).read().decode()
_rows = [line.split(",") for line in _raw.strip().split("\n") if line.strip()]
_all_features = np.array([[float(v) for v in r[:4]] for r in _rows])

# Setosa (0→-1) vs versicolor (1→+1), 4 features
X_raw = _all_features[:100]
y = np.array([-1]*50 + [1]*50, dtype=float)
X = (X_raw - X_raw.min(0)) / (X_raw.max(0) - X_raw.min(0)) * np.pi

# Train/test split (80/20)
rng = np.random.RandomState(42)
idx = rng.permutation(100)
X_train, y_train = X[idx[:80]], y[idx[:80]]
X_test, y_test = X[idx[80:]], y[idx[80:]]

# --- Build VQC: feature encoding + trainable ansatz ---
n_qubits = 4
qc = Circuit(n_qubits)
qc.compose(
    angle_feature_map(n_qubits),
    strongly_entangling_layers(n_qubits, n_layers=2, prefix="w"),
)

qc.init_params(0.1)
opt = Adam(stepsize=0.05)

# --- Train ---
print("=== Variational Quantum Classifier (Iris) ===\n")
print(f"  train: {len(X_train)} samples, test: {len(X_test)} samples")
print(f"  circuit: {n_qubits} qubits, {len(qc.trainable_parameters)} trainable params\n")

for epoch in range(60):
    grad = cost_gradient(qc, mse_cost, X_train, y_train)
    opt.step(qc, grad=grad)
    if epoch % 10 == 0 or epoch == 59:
        trained = qc.bind()
        preds = np.sign(predict(trained, X_train))
        acc = np.mean(preds == y_train) * 100
        loss = mse_cost(trained, X_train, y_train)
        print(f"  epoch {epoch:2d}: train_acc={acc:.0f}%  mse={loss:.4f}")

# --- Evaluate on test set ---
trained = qc.bind()
test_preds = np.sign(predict(trained, X_test))
test_acc = np.mean(test_preds == y_test) * 100
print(f"\n  test accuracy: {test_acc:.0f}% ({int(test_acc * len(y_test) / 100)}/{len(y_test)})")
