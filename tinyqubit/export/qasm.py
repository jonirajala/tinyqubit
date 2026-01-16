"""
OpenQASM export.

Contains:
    - to_openqasm2(circuit, tracker) → str
    - to_openqasm3(circuit, tracker) → str
    
QubitTracker SWAPs are materialized here, not before.
"""
