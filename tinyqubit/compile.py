"""
Main compilation entry point.

Contains:
    - compile(circuit, target) → (CompiledCircuit, CompileReport)
    - Orchestrates: decompose → route → optimize
    - All work happens here, not during circuit construction (lazy)
"""
