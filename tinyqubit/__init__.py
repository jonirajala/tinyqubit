"""
TinyQubit - A tiny quantum circuit compiler
"""

from .ir import Circuit, Gate, Operation, Parameter
from .target import Target
from .compile import transpile, precompile, realize, CompileConfig
from .simulator import simulate, simulate_statevector, simulate_density, simulate_stabilizer, states_equal, sample, to_unitary, probabilities, marginal_counts, simulate_mps, mps_to_statevector, is_clifford
from .qasm import to_openqasm2, to_openqasm3, from_openqasm2, from_openqasm3, UnsupportedGateError
from .hardware import submit_ibm, wait_ibm, IBMBackend, BraketBackend
from .simulator.noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise
)
from .measurement.observable import Observable, I, X, Y, Z, expectation, expectation_batch, expectation_sweep, state_fidelity, partial_trace, entanglement_entropy, concurrence, mutual_information
from .qml.optim import parameter_shift_gradient, finite_difference_gradient, adjoint_gradient, backprop_gradient, gradient_landscape, quantum_fisher_information, cost_gradient
from .qml.loss import kl_divergence, mse
from .qml.circuits import qft, ghz, grover_oracle, qaoa_mixer
from .qml.layers import hardware_efficient_ansatz
from .measurement.ftqc import resource_estimate, ResourceEstimate
from .measurement.mitigation import zne, calibration_matrix, mitigate_readout

__all__ = [
    # Core IR
    "Circuit", "Gate", "Operation", "Parameter",
    # Compilation
    "Target", "transpile", "precompile", "realize", "CompileConfig",
    # Simulation
    "simulate", "simulate_statevector", "simulate_density",
    "simulate_stabilizer", "simulate_mps", "mps_to_statevector",
    "states_equal", "sample", "to_unitary", "probabilities",
    "marginal_counts", "is_clifford",
    # Noise
    "NoiseModel", "depolarizing", "amplitude_damping", "phase_damping",
    "readout_error", "realistic_noise",
    # Observables & measurement
    "Observable", "I", "X", "Y", "Z",
    "expectation", "expectation_batch", "expectation_sweep",
    # Gradients
    "parameter_shift_gradient", "finite_difference_gradient",
    "adjoint_gradient", "backprop_gradient", "gradient_landscape",
    "quantum_fisher_information", "cost_gradient",
    "kl_divergence", "mse",
    # Quantum info
    "state_fidelity", "partial_trace", "entanglement_entropy",
    "concurrence", "mutual_information",
    # Circuit library
    "qft", "ghz", "grover_oracle", "hardware_efficient_ansatz", "qaoa_mixer",
    # Export
    "to_openqasm2", "to_openqasm3", "from_openqasm2", "from_openqasm3",
    "UnsupportedGateError", "submit_ibm", "wait_ibm", "IBMBackend", "BraketBackend",
    # FTQC
    "resource_estimate", "ResourceEstimate",
    # Error mitigation
    "zne", "calibration_matrix", "mitigate_readout",
]
