"""Simulator package — re-exports for backward compatibility."""
from .statevector import (
    simulate_statevector, _apply_single_qubit, _apply_two_qubit,
    _apply_three_qubit, _apply_four_qubit, _apply_diagonal_1q, _build_gate_unitary,
    _DIAG_PHASE, _apply_measure, _apply_reset, _apply_gate_noise,
    _apply_batch_1q, _collect_1q_block, _get_perm,
)
from .density import simulate_density
from .stabilizer import is_clifford, simulate_stabilizer
from .mps import simulate_mps, mps_to_statevector, MPSState, mps_expectation, mps_sample, mps_probabilities
from .simulator import simulate, states_equal, sample, probabilities, marginal_counts, verify, to_unitary
from .noise import (
    NoiseModel, depolarizing, amplitude_damping, phase_damping,
    readout_error, realistic_noise,
)
