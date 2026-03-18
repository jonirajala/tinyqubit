"""
QAOA for MaxCut on a small graph.

Graph: 6 nodes, 7 edges (irregular). QAOA depth p=2.
Cost: C = Σ ½(1 - Z_i Z_j) counts edges cut.
Exact MaxCut = 7 (partition {0,2,4} vs {1,3,5}).

Ref: Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm",
     arXiv:1411.4028 (2014).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tinyqubit import Circuit, simulate, sample, expectation
from tinyqubit.qml.optim import Adam
from tinyqubit.qml.circuits import maxcut_hamiltonian, qaoa_mixer

# --- Problem graph ---
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
n_qubits = 6

H = maxcut_hamiltonian(edges)
circuit = qaoa_mixer(edges, p=2)

# --- QAOA optimization (negate H: Adam minimizes, we want to maximize cuts) ---
print("=== QAOA: MaxCut ===\n")
print(f"  graph: {n_qubits} nodes, {len(edges)} edges")
print(f"  QAOA depth: p=2 ({len(circuit.parameters)} parameters)\n")

circuit.init_params(seed=42, trainable_only=False)
opt = Adam(stepsize=0.1)

for step in range(150):
    _, neg_cost = opt.step_and_cost(circuit, -H)
    if step % 25 == 0:
        print(f"  step {step:3d}: ⟨C⟩ = {-neg_cost:.4f}")

cost_final = -neg_cost
print(f"\n  converged: ⟨C⟩ = {cost_final:.4f}")
print(f"  exact max:       {len(edges):.4f}")

# --- Sample solution ---
state, _ = simulate(circuit.bind())
counts = sample(state, shots=1000, seed=0)
best = max(counts, key=counts.get)
cut = sum(1 for i, j in edges if best[i] != best[j])
print(f"\n  most frequent bitstring: |{best}⟩ ({counts[best]}/1000)")
print(f"  edges cut: {cut}/{len(edges)}")
