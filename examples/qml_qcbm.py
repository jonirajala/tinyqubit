"""
Quantum Circuit Born Machine (QCBM) for learning a probability distribution.

Target: bars-and-stripes patterns on a 2x2 grid (4 qubits).
Valid patterns: 0000, 0101, 1010, 1111, 0011, 1100 (6 of 16).
Cost: KL divergence between circuit output probabilities and target distribution.
Training: Adam with backprop gradient on KL divergence.

Ref: Benedetti et al., "A generative modeling approach for benchmarking and
     training shallow quantum circuits", npj Quantum Inf. 5, 45 (2019).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, probabilities, kl_divergence
from tinyqubit.qml.optim import Adam
from tinyqubit.qml.layers import hardware_efficient_ansatz

# --- Target distribution: bars-and-stripes 2x2 ---
# Grid layout:  q0 q1
#               q2 q3
# Valid: all-0, all-1, cols (01,01), cols (10,10), rows (00,11), rows (11,00)
n_qubits = 4
BAS_STATES = [0b0000, 0b0101, 0b1010, 0b1111, 0b0011, 0b1100]
target = np.zeros(2 ** n_qubits)
for s in BAS_STATES:
    target[s] = 1.0 / len(BAS_STATES)

# --- Train ---
print("=== Quantum Circuit Born Machine (bars-and-stripes 2x2) ===\n")
print(f"  target: {len(BAS_STATES)} valid patterns out of {2**n_qubits}")

qc = hardware_efficient_ansatz(n_qubits, depth=5, circular=True)
print(f"  circuit: {n_qubits} qubits, {len(qc.parameters)} parameters, 5 layers\n")

qc.init_params(seed=42, trainable_only=False)
opt = Adam(stepsize=0.05)
loss = kl_divergence(target)

for step in range(200):
    opt.step(qc, loss)
    if step % 25 == 0 or step == 199:
        kl = loss(probabilities(qc.bind()))
        print(f"  step {step:3d}: KL = {kl:.4f}")

# --- Results ---
probs = probabilities(qc.bind())
print(f"\nLearned distribution:")
print(f"  {'State':>6}  {'Target':>8}  {'Learned':>8}")
for i in range(2 ** n_qubits):
    if target[i] > 0 or probs[i] > 0.01:
        bs = format(i, f'0{n_qubits}b')
        marker = " *" if target[i] > 0 else ""
        print(f"  |{bs}⟩  {target[i]:8.4f}  {probs[i]:8.4f}{marker}")

total_on_target = sum(probs[s] for s in BAS_STATES)
print(f"\n  probability mass on valid patterns: {total_on_target:.1%}")
