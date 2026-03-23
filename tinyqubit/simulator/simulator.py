"""Simulator dispatch, utilities, and verification."""
from __future__ import annotations
import numpy as np
from math import pi
from typing import TYPE_CHECKING
from ..ir import Circuit, Gate, _has_parameter, _get_gate_matrix
from .statevector import simulate_statevector, _apply_single_qubit, _apply_two_qubit, _apply_three_qubit, _apply_four_qubit
from .density import simulate_density
from .stabilizer import is_clifford, simulate_stabilizer
from .mps import simulate_mps, mps_to_statevector, MPSState, mps_expectation, mps_sample, mps_probabilities

if TYPE_CHECKING:
    from .noise import NoiseModel


def simulate(circuit: Circuit, seed: int | None = None, noise_model: "NoiseModel | None" = None,
             batch_ops: bool = True) -> tuple[np.ndarray | MPSState, dict[int, int]]:
    """Simulate circuit. Returns (statevector or MPSState, classical_bits dict).

    Auto-dispatches: Clifford circuits → stabilizer tableau, >28 qubits → MPS,
    otherwise statevector. Noise or custom initial state forces statevector path.
    """
    n = circuit.n_qubits

    if not circuit._validated:
        for op in circuit.ops:
            if _has_parameter(op.params):
                raise TypeError(f"Cannot simulate: {op.gate.name} has unbound Parameter. Call circuit.bind() first.")
            for q in op.qubits:
                if not (0 <= q < n):
                    raise ValueError(f"Invalid qubit index {q} for {n}-qubit circuit in {op.gate.name}")
        circuit._validated = True

    if noise_model is None and circuit._initial_state is None and is_clifford(circuit):
        return simulate_stabilizer(circuit, seed)

    if noise_model is None and circuit._initial_state is None and n > 28:
        tensors, classical = simulate_mps(circuit, seed=seed)
        return MPSState(tensors, n), classical

    return simulate_statevector(circuit, n, seed, noise_model, batch_ops)


def states_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if two states are equal up to global phase."""
    if isinstance(a, MPSState) or isinstance(b, MPSState):
        raise TypeError("states_equal does not support MPSState. Convert to statevector first.")
    if a.shape != b.shape: return False
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < tol or norm_b < tol: return norm_a < tol and norm_b < tol
    return np.isclose(np.abs(np.vdot(a, b)) / (norm_a * norm_b), 1.0, atol=tol)

def sample(state, shots: int, seed: int = None) -> dict[str, int]:
    """Sample measurement outcomes from statevector or MPSState. Returns {bitstring: count}."""
    if isinstance(state, MPSState):
        return mps_sample(state.tensors, shots, seed)
    rng = np.random.default_rng(seed)
    probs = np.abs(state) ** 2
    probs /= probs.sum()  # Normalize to handle numerical drift
    outcomes = rng.choice(len(state), size=shots, p=probs)
    values, counts = np.unique(outcomes, return_counts=True)
    n_bits = int(np.log2(len(state)))
    return {format(v, f'0{n_bits}b'): int(c) for v, c in zip(values, counts)}

def probabilities(circuit: Circuit, wires: list[int] | None = None, seed: int | None = None) -> np.ndarray:
    """Compute measurement probabilities. Optionally marginal over specified wires."""
    if circuit.is_parameterized and circuit.param_values:
        circuit = circuit.bind()
    state, _ = simulate(circuit, seed=seed)
    if isinstance(state, MPSState):
        return mps_probabilities(state.tensors, wires)
    probs = np.abs(state) ** 2
    if wires is None: return probs
    n = circuit.n_qubits
    probs = probs.reshape([2] * n)
    trace_out = tuple(i for i in range(n) if i not in wires)
    if trace_out: probs = probs.sum(axis=trace_out)
    order = [sorted(wires).index(w) for w in wires]
    result = probs.transpose(order).flatten()
    assert abs(result.sum() - 1.0) < 1e-10, "probabilities do not sum to 1"
    return result

def marginal_counts(counts: dict[str, int], wires: list[int]) -> dict[str, int]:
    """Extract marginal counts for specified wire positions."""
    result = {}
    for bitstring, count in counts.items():
        key = ''.join(bitstring[w] for w in wires)
        result[key] = result.get(key, 0) + count
    return result

def verify(original: Circuit, compiled: Circuit, tracker=None, tol: float = 1e-9, n_samples: int = 5) -> bool:
    """Check circuit equivalence, including across random parameter values for parametric circuits."""

    def _check_equiv(c1: Circuit, c2: Circuit) -> bool:
        n = c1.n_qubits
        has_nonunitary = any(op.gate in (Gate.MEASURE, Gate.RESET) or op.condition is not None for op in c1.ops)
        if n <= 12 and not has_nonunitary:
            U1, U2 = to_unitary(c1), to_unitary(c2)
            if tracker is not None:
                final_perm = [tracker.logical_to_phys(i) for i in range(n)]
                initial_perm = list(tracker.initial_layout) if tracker.initial_layout else list(range(n))
                U2 = np.transpose(U2.reshape([2]*n + [2]*n),
                                  final_perm + [p + n for p in initial_perm]).reshape(2**n, 2**n)
            return abs(np.trace(U1.conj().T @ U2)) / (2**n) > 1 - tol
        # Statevector fallback
        s1, _ = simulate(c1)
        s2, _ = simulate(c2)
        if tracker is not None:
            perm = [tracker.logical_to_phys(i) for i in range(n)]
            s2 = np.transpose(s2.reshape([2] * n), perm).reshape(-1)
        return states_equal(s1, s2, tol)

    if original.is_parameterized:
        rng = np.random.default_rng(42)
        param_names = sorted(p.name for p in original.parameters)
        for _ in range(n_samples):
            vals = {name: rng.uniform(0, 2 * pi) for name in param_names}
            c1, c2 = original.bind(vals), compiled.bind(vals)
            if not _check_equiv(c1, c2):
                return False
        return True
    return _check_equiv(original, compiled)


def to_unitary(circuit: Circuit) -> np.ndarray:
    """Build full unitary matrix of the circuit. Max 12 qubits."""
    n = circuit.n_qubits
    if n > 12: raise ValueError("to_unitary supports at most 12 qubits")
    if circuit._initial_state is not None: raise ValueError("to_unitary does not support initialized circuits")
    for op in circuit.ops:
        if op.gate in (Gate.MEASURE, Gate.RESET): raise ValueError(f"to_unitary does not support {op.gate.name}")
        if op.condition is not None: raise ValueError("to_unitary does not support conditional operations")
        if _has_parameter(op.params): raise TypeError(f"Cannot compute unitary: {op.gate.name} has unbound Parameter")
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for op in circuit.ops:
        if op.gate.n_qubits == 1:
            mat = _get_gate_matrix(op.gate, op.params)
            for k in range(dim): U[:, k] = _apply_single_qubit(U[:, k], mat, op.qubits[0], n)
        elif op.gate.n_qubits == 2:
            for k in range(dim): U[:, k] = _apply_two_qubit(U[:, k], op.gate, op.qubits[0], op.qubits[1], n, op.params)
        else:
            for k in range(dim): U[:, k] = _apply_three_qubit(U[:, k], op.gate, *op.qubits, n)
    return U
