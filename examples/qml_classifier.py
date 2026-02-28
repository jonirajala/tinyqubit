"""
Iris classification with a quantum kernel nearest-centroid classifier.

Data: UCI Iris (setosa vs versicolor, 4 features, 4 qubits).
Kernel: ZZ feature map → Gram matrix via |⟨φ(xᵢ)|φ(xⱼ)⟩|².
Classifier: for each test point, predict class with highest mean kernel similarity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import urllib.request
import numpy as np
from tinyqubit import kernel_matrix
from tinyqubit.feature_map import zz_feature_map

# --- Fetch iris dataset from UCI ML Repository ---
_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
_raw = urllib.request.urlopen(_UCI_URL).read().decode()
_rows = [line.split(",") for line in _raw.strip().split("\n") if line.strip()]
_all_features = np.array([[float(v) for v in r[:4]] for r in _rows])
_all_labels = np.array([r[4].strip() for r in _rows])

# First 100 samples = setosa (0) + versicolor (1), all 4 features
_IRIS_RAW = _all_features[:100]
_IRIS_LABELS = np.array([0]*50 + [1]*50)

# Scale to [0, π]
X = (_IRIS_RAW - _IRIS_RAW.min(0)) / (_IRIS_RAW.max(0) - _IRIS_RAW.min(0)) * np.pi

# Train/test split (80/20, deterministic)
rng = np.random.RandomState(42)
idx = rng.permutation(100)
X_train, y_train = X[idx[:80]], _IRIS_LABELS[idx[:80]]
X_test, y_test = X[idx[80:]], _IRIS_LABELS[idx[80:]]

# --- Quantum kernel Gram matrices (4 qubits = 4 features) ---
print("=== Iris Quantum Kernel Classifier ===\n")
print(f"  train: {len(X_train)} samples, test: {len(X_test)} samples")
print("  computing kernel matrices...")

K_train = kernel_matrix(zz_feature_map, X_train, n_qubits=4, wires=[0, 1, 2, 3])
K_test = kernel_matrix(zz_feature_map, X_test, X_train, n_qubits=4, wires=[0, 1, 2, 3])

# --- Kernel nearest-centroid: predict class with highest mean kernel similarity ---
preds = []
for i in range(len(X_test)):
    sim = [K_test[i, y_train == c].mean() for c in [0, 1]]
    preds.append(np.argmax(sim))

acc = np.mean(np.array(preds) == y_test) * 100
print(f"\n  accuracy: {acc:.0f}% ({int(acc * len(y_test) / 100)}/{len(y_test)})")
