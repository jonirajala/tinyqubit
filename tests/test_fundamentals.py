"""Tests for Phase 1.1: Circuit Fundamentals."""
import numpy as np
import pytest
from tinyqubit import Circuit, Gate, Operation, Parameter, simulate
from tinyqubit.simulator import to_unitary, probabilities, marginal_counts


class TestToUnitary:
    def test_identity(self):
        assert np.allclose(to_unitary(Circuit(1)), np.eye(2))

    def test_x_gate(self):
        c = Circuit(1)
        c.x(0)
        assert np.allclose(to_unitary(c), [[0, 1], [1, 0]])

    def test_unitarity_1q(self):
        c = Circuit(1)
        c.h(0).s(0).t(0)
        U = to_unitary(c)
        assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)

    def test_unitarity_2q(self):
        c = Circuit(2)
        c.h(0).cx(0, 1).rz(1, 0.5)
        U = to_unitary(c)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

    def test_cx_matrix(self):
        c = Circuit(2)
        c.cx(0, 1)
        U = to_unitary(c)
        # q0=MSB: |10⟩=idx2 → |11⟩=idx3
        assert np.isclose(abs(U[0, 0]), 1)
        assert np.isclose(abs(U[3, 2]), 1)

    def test_3q_gate(self):
        c = Circuit(3)
        c.ccx(0, 1, 2)
        U = to_unitary(c)
        assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-10)

    def test_wrapper(self):
        c = Circuit(1)
        c.h(0)
        assert np.allclose(c.to_unitary(), to_unitary(c))

    def test_consistent_with_simulate(self):
        c = Circuit(2)
        c.h(0).cx(0, 1)
        U = to_unitary(c)
        state, _ = simulate(c)
        basis0 = np.zeros(4, dtype=complex); basis0[0] = 1.0
        assert np.allclose(U @ basis0, state, atol=1e-10)

    def test_rejects_measure(self):
        c = Circuit(1); c.measure(0)
        with pytest.raises(ValueError): to_unitary(c)

    def test_rejects_reset(self):
        c = Circuit(1); c.reset(0)
        with pytest.raises(ValueError): to_unitary(c)

    def test_rejects_parameter(self):
        c = Circuit(1); c.rx(0, Parameter("t"))
        with pytest.raises(TypeError): to_unitary(c)

    def test_rejects_too_many_qubits(self):
        with pytest.raises(ValueError): to_unitary(Circuit(13))

    def test_rejects_initialized(self):
        c = Circuit(1); c.initialize([0, 1])
        with pytest.raises(ValueError): to_unitary(c)


class TestInverse:
    def test_roundtrip_identity(self):
        c = Circuit(2)
        c.h(0).cx(0, 1).s(1).rz(0, 0.7)
        U, U_inv = to_unitary(c), to_unitary(c.inverse())
        assert np.allclose(U @ U_inv, np.eye(4), atol=1e-10)

    def test_s_to_sdg(self):
        c = Circuit(1); c.s(0)
        assert c.inverse().ops[0].gate == Gate.SDG

    def test_sdg_to_s(self):
        c = Circuit(1); c.sdg(0)
        assert c.inverse().ops[0].gate == Gate.S

    def test_t_to_tdg(self):
        c = Circuit(1); c.t(0)
        assert c.inverse().ops[0].gate == Gate.TDG

    def test_param_negation(self):
        c = Circuit(1); c.rx(0, 1.5)
        assert c.inverse().ops[0].params == (-1.5,)

    def test_cp_param_negation(self):
        c = Circuit(2); c.cp(0, 1, 0.8)
        assert c.inverse().ops[0].params == (-0.8,)

    def test_reverses_order(self):
        c = Circuit(1); c.h(0).x(0)
        inv = c.inverse()
        assert inv.ops[0].gate == Gate.X
        assert inv.ops[1].gate == Gate.H

    def test_self_adjoint_gates(self):
        for g in [Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.CX, Gate.CZ, Gate.SWAP, Gate.CCX, Gate.CCZ]:
            c = Circuit(max(3, g.n_qubits))
            c.ops.append(Operation(g, tuple(range(g.n_qubits))))
            assert c.inverse().ops[0].gate == g

    def test_rejects_measure(self):
        c = Circuit(1); c.measure(0)
        with pytest.raises(ValueError): c.inverse()

    def test_rejects_parameter(self):
        c = Circuit(1); c.rx(0, Parameter("t"))
        with pytest.raises(ValueError): c.inverse()

    def test_rejects_initialized(self):
        c = Circuit(1); c.initialize([0, 1])
        with pytest.raises(ValueError): c.inverse()


class TestProbabilities:
    def test_h_gate(self):
        c = Circuit(1); c.h(0)
        assert np.allclose(probabilities(c), [0.5, 0.5])

    def test_sums_to_one(self):
        c = Circuit(2); c.h(0).cx(0, 1)
        assert np.isclose(probabilities(c).sum(), 1.0)

    def test_marginal_wires(self):
        c = Circuit(2); c.h(0).cx(0, 1)  # Bell state
        assert np.allclose(probabilities(c, wires=[0]), [0.5, 0.5])

    def test_wire_ordering(self):
        c = Circuit(2); c.x(0)  # |10⟩
        assert np.allclose(probabilities(c, wires=[0]), [0, 1])  # q0=1
        assert np.allclose(probabilities(c, wires=[1]), [1, 0])  # q1=0

    def test_all_wires(self):
        c = Circuit(2); c.h(0).cx(0, 1)
        assert np.allclose(probabilities(c), probabilities(c, wires=[0, 1]))


class TestMarginalCounts:
    def test_basic(self):
        counts = {"00": 50, "01": 10, "10": 30, "11": 10}
        assert marginal_counts(counts, [0]) == {"0": 60, "1": 40}

    def test_second_wire(self):
        counts = {"00": 50, "01": 10, "10": 30, "11": 10}
        assert marginal_counts(counts, [1]) == {"0": 80, "1": 20}

    def test_multi_wire(self):
        counts = {"000": 100}
        assert marginal_counts(counts, [0, 2]) == {"00": 100}

    def test_preserves_total(self):
        counts = {"00": 50, "01": 10, "10": 30, "11": 10}
        m = marginal_counts(counts, [0])
        assert sum(m.values()) == sum(counts.values())


class TestInitialize:
    def test_custom_state(self):
        c = Circuit(1); c.initialize([0, 1])
        state, _ = simulate(c)
        assert np.allclose(np.abs(state), [0, 1])

    def test_normalizes(self):
        c = Circuit(1); c.initialize([3, 4])
        state, _ = simulate(c)
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_with_gates(self):
        c = Circuit(1); c.initialize([0, 1]); c.x(0)  # |1⟩ → |0⟩
        state, _ = simulate(c)
        assert np.allclose(np.abs(state), [1, 0])

    def test_wrong_size(self):
        c = Circuit(2)
        with pytest.raises(ValueError): c.initialize([0, 1])

    def test_preserved_by_bind(self):
        c = Circuit(1); c.initialize([0, 1]); c.rx(0, Parameter("t"))
        state, _ = simulate(c.bind({"t": 0.5}))
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_2q_initial_state(self):
        c = Circuit(2); c.initialize([0, 0, 0, 1])  # |11⟩
        state, _ = simulate(c)
        assert np.isclose(abs(state[3]), 1.0)
