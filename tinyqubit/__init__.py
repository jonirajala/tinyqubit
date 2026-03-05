"""
TinyQubit - A tiny quantum circuit compiler
"""

from .ir import Circuit, Gate, Operation, Parameter
from .dag import DAGCircuit, commutes
from .target import Target, validate, IBM_EAGLE_R3, IBM_BRISBANE, IBM_OSAKA, IBM_KYOTO, IBM_TORINO, IONQ_HARMONY, IONQ_ARIA, RIGETTI_ANKAA, IQM_GARNET, IQM_SPARK
from .compile import transpile, precompile, realize, CompileConfig, PRESET_FAST, PRESET_DEFAULT, PRESET_QUALITY, PRESET_FT
from .simulator import simulate, simulate_density, simulate_batch, states_equal, sample, to_unitary, probabilities, marginal_counts, verify
from .mps import simulate_mps, mps_to_statevector
from .export import to_openqasm2, to_openqasm3, from_openqasm2, from_openqasm3, UnsupportedGateError
from .export.backends import submit_ibm, wait_ibm
from .noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise
)
from .observable import Observable, X, Y, Z, expectation, expectation_batch, expectation_sweep
from .gradient import parameter_shift_gradient, finite_difference_gradient, adjoint_gradient, gradient_landscape, quantum_fisher_information, cost_gradient
from .info import state_fidelity, partial_trace, entanglement_entropy, concurrence, mutual_information
from .schedule import circuit_duration, idle_periods
from .passes.fuse import fuse_1q_gates
from .passes.dd import dynamic_decoupling
from .library import qft, ghz, grover_oracle, hardware_efficient_ansatz, qaoa_mixer
from .ftqc import resource_estimate, ResourceEstimate
from .mitigation import zne, calibration_matrix, mitigate_readout

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
    "validate",
    "IBM_EAGLE_R3",
    "IBM_BRISBANE",
    "IBM_OSAKA",
    "IBM_KYOTO",
    "IBM_TORINO",
    "IONQ_HARMONY",
    "IONQ_ARIA",
    "RIGETTI_ANKAA",
    "IQM_GARNET",
    "IQM_SPARK",
    "transpile",
    "precompile",
    "realize",
    "CompileConfig",
    "PRESET_FAST",
    "PRESET_DEFAULT",
    "PRESET_QUALITY",
    "PRESET_FT",
    "circuit_duration",
    "idle_periods",
    # Simulation
    "simulate",
    "simulate_density",
    "simulate_batch",
    "states_equal",
    "sample",
    "to_unitary",
    "probabilities",
    "marginal_counts",
    "verify",
    "simulate_mps",
    "mps_to_statevector",
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
    "dynamic_decoupling",
    # Circuit library
    "qft",
    "ghz",
    "grover_oracle",
    "hardware_efficient_ansatz",
    "qaoa_mixer",
    # Export
    "to_openqasm2",
    "to_openqasm3",
    "from_openqasm2",
    "from_openqasm3",
    "UnsupportedGateError",
    "submit_ibm",
    "wait_ibm",
    # FTQC
    "resource_estimate",
    "ResourceEstimate",
    # Error mitigation
    "zne",
    "calibration_matrix",
    "mitigate_readout",
]
