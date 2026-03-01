"""
TinyQubit - A tiny quantum circuit compiler
"""

from .ir import Circuit, Gate, Operation, Parameter
from .dag import DAGCircuit, commutes
from .target import Target
from .compile import transpile
from .simulator import simulate, simulate_batch, states_equal, sample, to_unitary, probabilities, marginal_counts, verify
from .export import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError
from .noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise
)
from .observable import Observable, X, Y, Z, expectation, expectation_batch, expectation_sweep
from .gradient import parameter_shift_gradient, finite_difference_gradient, adjoint_gradient, gradient_landscape, quantum_fisher_information, cost_gradient
from .info import state_fidelity, partial_trace, entanglement_entropy, concurrence, mutual_information
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
    "simulate_batch",
    "states_equal",
    "sample",
    "to_unitary",
    "probabilities",
    "marginal_counts",
    "verify",
    # Noise
    "NoiseModel",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "readout_error",
    "realistic_noise",
    # Observables
    "Observable",
    "X",
    "Y",
    "Z",
    "expectation",
    "expectation_batch",
    "expectation_sweep",
    # Gradients
    "parameter_shift_gradient",
    "finite_difference_gradient",
    "adjoint_gradient",
    "gradient_landscape",
    "quantum_fisher_information",
    "cost_gradient",
    # Quantum info
    "state_fidelity",
    "partial_trace",
    "entanglement_entropy",
    "concurrence",
    "mutual_information",
    # Optimization
    "fuse_1q_gates",
    # Export
    "to_openqasm2",
    "to_openqasm3",
    "to_qiskit",
    "UnsupportedGateError",
]
