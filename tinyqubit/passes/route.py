"""
Qubit routing for hardware connectivity.

Uses QubitTracker for symbolic routing:
    - Track logical→physical mapping without inserting SWAPs
    - Shortest-path routing with deterministic tie-breaking
    - SWAPs materialized only at export time
"""
