"""
Basic TinyQubit circuit examples.

Run: python examples/circuit.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit, Gate
from tinyqubit.export import to_openqasm2

# =============================================================================
# Example 1: Bell State
# =============================================================================
print("=== Bell State ===")
bell = Circuit(2).h(0).cx(0, 1)

print(f"Operations: {len(bell.ops)}")
for op in bell.ops:
    print(f"  {op.gate.name} on qubits {op.qubits}")

print("\nOpenQASM:")
print(to_openqasm2(bell))

# =============================================================================
# Example 2: GHZ State (3-qubit entanglement)
# =============================================================================
print("\n=== GHZ State ===")
ghz = Circuit(3).h(0).cx(0, 1).cx(1, 2)

print(f"Operations: {len(ghz.ops)}")
print("\nOpenQASM:")
print(to_openqasm2(ghz))

# =============================================================================
# Example 3: Circuit with rotations
# =============================================================================
print("\n=== Rotation Circuit ===")
rot = Circuit(2)
rot.h(0)
rot.rx(0, 1.57)  # ~pi/2
rot.ry(1, 3.14)  # ~pi
rot.rz(0, 0.78)  # ~pi/4
rot.cx(0, 1)

print("OpenQASM:")
print(to_openqasm2(rot))

# =============================================================================
# Example 4: Circuit with measurements
# =============================================================================
print("\n=== Measured Circuit ===")
measured = Circuit(2)
measured.h(0).cx(0, 1)
measured.measure(0).measure(1)

print("OpenQASM:")
print(to_openqasm2(measured))

# =============================================================================
# Example 5: Method chaining
# =============================================================================
print("\n=== Method Chaining ===")
chain = (
    Circuit(3)
    .h(0)
    .h(1)
    .h(2)
    .cx(0, 1)
    .cx(1, 2)
    .t(0)
    .s(1)
    .z(2)
)

print(f"Built circuit with {len(chain.ops)} operations using chaining")
