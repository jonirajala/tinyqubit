"""
Minimal statevector simulator for testing.

Contains:
    - Statevector: State representation and gate application
    - simulate(circuit) → Statevector
    - states_equal(a, b) → bool (up to global phase)
    - sample(statevector, shots, seed) → counts

No external dependencies beyond numpy.
"""
