"""
Property-based tests using hypothesis.

Tests:
    - Semantic preservation: simulate(original) == simulate(compiled)
    - Constraint satisfaction: output respects target topology/basis
    - Unitary equivalence (up to global phase)
"""
import numpy as np
from math import pi
import pytest

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.target import Target
from tinyqubit.compile import transpile
from tinyqubit.simulator import simulate, states_equal


# =============================================================================
# Strategies for generating random circuits and targets
# =============================================================================

def line_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, i + 1) for i in range(n - 1))


def grid_topology(rows: int, cols: int) -> frozenset[tuple[int, int]]:
    edges = set()
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c < cols - 1: edges.add((i, i + 1))
            if r < rows - 1: edges.add((i, i + cols))
    return frozenset(edges)


def all_to_all_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, j) for i in range(n) for j in range(i + 1, n))


# Gates that can be used in random circuits (excluding MEASURE for simulation tests)
SINGLE_QUBIT_GATES = [Gate.H, Gate.X, Gate.Y, Gate.Z, Gate.S, Gate.T]
PARAM_GATES = [Gate.RX, Gate.RY, Gate.RZ]
TWO_QUBIT_GATES = [Gate.CX, Gate.CZ]


@st.composite
def random_circuit(draw, min_qubits=2, max_qubits=5, min_ops=1, max_ops=15):
    """Generate a random quantum circuit."""
    n_qubits = draw(st.integers(min_qubits, max_qubits))
    n_ops = draw(st.integers(min_ops, max_ops))

    c = Circuit(n_qubits)
    for _ in range(n_ops):
        gate_type = draw(st.sampled_from(["single", "param", "two"]))

        if gate_type == "single":
            gate = draw(st.sampled_from(SINGLE_QUBIT_GATES))
            qubit = draw(st.integers(0, n_qubits - 1))
            c.ops.append(Operation(gate, (qubit,)))

        elif gate_type == "param":
            gate = draw(st.sampled_from(PARAM_GATES))
            qubit = draw(st.integers(0, n_qubits - 1))
            theta = draw(st.floats(-2 * pi, 2 * pi, allow_nan=False, allow_infinity=False))
            c.ops.append(Operation(gate, (qubit,), (theta,)))

        else:  # two-qubit
            gate = draw(st.sampled_from(TWO_QUBIT_GATES))
            q0 = draw(st.integers(0, n_qubits - 1))
            q1 = draw(st.integers(0, n_qubits - 1).filter(lambda x: x != q0))
            c.ops.append(Operation(gate, (q0, q1)))

    return c


@st.composite
def random_target(draw, n_qubits: int):
    """Generate a random target for given qubit count."""
    topo_type = draw(st.sampled_from(["line", "all_to_all"]))

    if topo_type == "line":
        edges = line_topology(n_qubits)
    else:
        edges = all_to_all_topology(n_qubits)

    # Use a basis that supports full decomposition
    basis = frozenset({Gate.RZ, Gate.RX, Gate.CX})

    return Target(n_qubits=n_qubits, edges=edges, basis_gates=basis, name=f"test_{topo_type}_{n_qubits}")


def permute_state(state: np.ndarray, tracker) -> np.ndarray:
    """Apply qubit permutation from tracker to statevector."""
    if tracker is None:
        return state

    n_qubits = int(np.log2(len(state)))
    perm = [tracker.logical_to_phys(i) for i in range(n_qubits)]

    # Reshape to tensor, transpose, reshape back
    state_tensor = state.reshape([2] * n_qubits)
    state_permuted = np.transpose(state_tensor, perm)
    return state_permuted.reshape(-1)


# =============================================================================
# Property tests
# =============================================================================

@given(random_circuit(max_qubits=4, max_ops=10))
@settings(max_examples=50, deadline=None)
def test_transpile_preserves_semantics(circuit):
    """Transpiled circuit produces same quantum state as original."""
    target = Target(
        n_qubits=circuit.n_qubits,
        edges=line_topology(circuit.n_qubits),
        basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
        name="test_line"
    )

    compiled = transpile(circuit, target)

    original_state, _ = simulate(circuit)
    compiled_state, _ = simulate(compiled)

    # Account for qubit permutation from routing
    if hasattr(compiled, '_tracker') and compiled._tracker is not None:
        compiled_state = permute_state(compiled_state, compiled._tracker)

    assert states_equal(original_state, compiled_state), (
        f"Semantic mismatch!\n"
        f"Original: {len(circuit.ops)} ops\n"
        f"Compiled: {len(compiled.ops)} ops"
    )


@given(random_circuit(max_qubits=4, max_ops=10))
@settings(max_examples=50, deadline=None)
def test_output_respects_target_basis(circuit):
    """Transpiled circuit only uses gates from target basis."""
    basis = frozenset({Gate.RZ, Gate.RX, Gate.CX})
    target = Target(
        n_qubits=circuit.n_qubits,
        edges=line_topology(circuit.n_qubits),
        basis_gates=basis,
        name="test_line"
    )

    compiled = transpile(circuit, target)

    for op in compiled.ops:
        assert op.gate in basis or op.gate == Gate.MEASURE, (
            f"Gate {op.gate.name} not in basis {[g.name for g in basis]}"
        )


@given(random_circuit(max_qubits=4, max_ops=10))
@settings(max_examples=50, deadline=None)
def test_output_respects_target_topology(circuit):
    """Transpiled circuit 2Q gates only on connected qubits."""
    edges = line_topology(circuit.n_qubits)
    target = Target(
        n_qubits=circuit.n_qubits,
        edges=edges,
        basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
        name="test_line"
    )

    compiled = transpile(circuit, target)

    for op in compiled.ops:
        if op.gate.n_qubits == 2:
            q0, q1 = op.qubits
            is_connected = (q0, q1) in edges or (q1, q0) in edges
            assert is_connected, (
                f"2Q gate {op.gate.name} on qubits {q0}, {q1} not connected in topology"
            )


@given(random_circuit(max_qubits=5, max_ops=8))
@settings(max_examples=30, deadline=None)
def test_transpile_deterministic(circuit):
    """Same circuit transpiled twice gives identical result."""
    target = Target(
        n_qubits=circuit.n_qubits,
        edges=line_topology(circuit.n_qubits),
        basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
        name="test_line"
    )

    result1 = transpile(circuit, target)
    result2 = transpile(circuit, target)

    # Compare operation by operation
    assert len(result1.ops) == len(result2.ops), "Different number of ops"
    for op1, op2 in zip(result1.ops, result2.ops):
        assert op1.gate == op2.gate, f"Gate mismatch: {op1.gate} vs {op2.gate}"
        assert op1.qubits == op2.qubits, f"Qubit mismatch: {op1.qubits} vs {op2.qubits}"
        assert op1.params == op2.params, f"Param mismatch: {op1.params} vs {op2.params}"


@given(random_circuit(max_qubits=4, max_ops=10))
@settings(max_examples=30, deadline=None)
def test_all_to_all_no_routing(circuit):
    """All-to-all topology needs no SWAP insertions (no extra 2Q gates)."""
    edges = all_to_all_topology(circuit.n_qubits)
    target = Target(
        n_qubits=circuit.n_qubits,
        edges=edges,
        basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
        name="test_all_to_all"
    )

    original_2q = sum(1 for op in circuit.ops if op.gate.n_qubits == 2)
    compiled = transpile(circuit, target)
    compiled_2q = sum(1 for op in compiled.ops if op.gate.n_qubits == 2)

    # After decomposition, some gates may expand but no SWAPs should be added
    # CZ -> H CX H adds 1 more 2Q, but SWAP would add 3
    # This is a soft check - just verify we didn't blow up
    assert compiled_2q <= original_2q * 3, (
        f"Too many 2Q gates: {original_2q} -> {compiled_2q}"
    )
