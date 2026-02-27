"""
TinyQubit - A tiny quantum circuit compiler
"""

from .ir import Circuit, Gate, Operation, Parameter
from .dag import DAGCircuit, commutes
from .target import Target
from .compile import transpile
from .simulator import simulate, states_equal, sample, to_unitary, probabilities, marginal_counts
from .export import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError
from .noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise
)
from .passes.fuse import fuse_1q_gates

__all__ = [
    # Core IR
    "Circuit",
    "Gate",
    "Operation",
    "Parameter",
    "DAGCircuit",
    "commutes",
    # Compilation
    "Target",
    "transpile",
    # Simulation
    "simulate",
    "states_equal",
    "sample",
    "to_unitary",
    "probabilities",
    "marginal_counts",
    # Noise
    "NoiseModel",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "readout_error",
    "realistic_noise",
    # Optimization
    "fuse_1q_gates",
    # Export
    "to_openqasm2",
    "to_openqasm3",
    "to_qiskit",
    "UnsupportedGateError",
]
