"""
Grover's Search Algorithm (2-qubit)

Searches for |11⟩ in a 4-item database with 1 iteration.

Run: python examples/grover.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit

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
print(grover.to_openqasm())
