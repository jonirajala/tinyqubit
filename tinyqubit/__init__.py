"""
TinyQubit - A tiny quantum circuit compiler
"""

from .ir import Circuit, Gate, Operation, Parameter
from .dag import DAGCircuit, commutes
from .target import Target
from .compile import transpile
from .simulator import simulate, simulate_batch, states_equal, sample, to_unitary, probabilities, marginal_counts
from .export import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError
from .noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise
)
from .observable import Observable, X, Y, Z, expectation, expectation_batch, expectation_sweep
from .gradient import parameter_shift_gradient, finite_difference_gradient, adjoint_gradient, gradient_landscape, quantum_fisher_information, cost_gradient
from .optimize import GradientDescent, Adam, SPSA, QNG
from .info import state_fidelity, partial_trace, entanglement_entropy, concurrence, mutual_information
from .feature_map import angle_feature_map, basis_feature_map, amplitude_feature_map, zz_feature_map, pauli_feature_map
from .kernel import quantum_kernel, kernel_matrix
from .ansatz import strongly_entangling_layers, basic_entangler_layers
from .cost import predict, cross_entropy_cost, mse_cost, fidelity_cost
from .hamiltonian import maxcut_hamiltonian
from .trainability import gradient_variance, expressibility
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
    # Optimizers
    "GradientDescent",
    "Adam",
    "SPSA",
    "QNG",
    # Quantum info
    "state_fidelity",
    "partial_trace",
    "entanglement_entropy",
    "concurrence",
    "mutual_information",
    # Feature maps
    "angle_feature_map",
    "basis_feature_map",
    "amplitude_feature_map",
    "zz_feature_map",
    "pauli_feature_map",
    # Quantum kernels
    "quantum_kernel",
    "kernel_matrix",
    # Ansatze
    "strongly_entangling_layers",
    "basic_entangler_layers",
    # Cost functions
    "predict",
    "cross_entropy_cost",
    "mse_cost",
    "fidelity_cost",
    # Hamiltonians
    "maxcut_hamiltonian",
    # Trainability
    "gradient_variance",
    "expressibility",
    # Optimization
    "fuse_1q_gates",
    # Export
    "to_openqasm2",
    "to_openqasm3",
    "to_qiskit",
    "UnsupportedGateError",
]
