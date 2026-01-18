"""
Grover's Search Algorithm (2-qubit)

Searches for |11⟩ in a 4-item database with 1 iteration.

Run: python examples/grover.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit
from tinyqubit.simulator import simulate, sample

# 2-qubit Grover searching for |11⟩
grover = Circuit(2)

# Initialize superposition
grover.h(0).h(1)

# Oracle: mark |11⟩ with phase flip
grover.cz(0, 1)

# Diffusion operator
grover.h(0).h(1)
grover.x(0).x(1)
grover.cz(0, 1)
grover.x(0).x(1)
grover.h(0).h(1)

# Measure
grover.measure(0).measure(1)

print("=== 2-Qubit Grover (searching for |11⟩) ===")
print(f"Operations: {len(grover.ops)}")
print()

# Simulate
state = simulate(grover)
print("Statevector:")
print(f"  |00⟩: {state[0]:.4f}")
print(f"  |01⟩: {state[1]:.4f}")
print(f"  |10⟩: {state[2]:.4f}")
print(f"  |11⟩: {state[3]:.4f}")
print()

# Sample
counts = sample(state, 1000, seed=42)
print(f"Samples (1000 shots): {counts}")
print()

print("OpenQASM:")
print(grover.to_openqasm())
