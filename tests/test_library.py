"""Tests for circuit library factories."""
import math
import numpy as np
from tinyqubit import Circuit, Gate, simulate
from tinyqubit.simulator import to_unitary
from tinyqubit.qml.library import qft, ghz, grover_oracle, hardware_efficient_ansatz, qaoa_mixer


class TestQFT:
    def test_uniform_superposition(self):
        """QFT of |0⟩ → uniform superposition."""
        sv, _ = simulate(qft(3))
        assert np.allclose(np.abs(sv), 1 / math.sqrt(8))

    def test_unitarity(self):
        U = to_unitary(qft(2))
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

    def test_matches_dft(self):
        """QFT(2) unitary matches normalized DFT matrix (up to bit-reversal)."""
        U = to_unitary(qft(2))
        # Standard DFT matrix (normalized)
        n = 4
        omega = np.exp(2j * math.pi / n)
        F = np.array([[omega ** (i * j) for j in range(n)] for i in range(n)]) / math.sqrt(n)
        # QFT should match DFT up to global phase per column
        for col in range(n):
            ratio = U[:, col] / F[:, col]
            nonzero = np.abs(F[:, col]) > 1e-10
            assert np.allclose(ratio[nonzero], ratio[nonzero][0], atol=1e-10)


class TestGHZ:
    def test_state(self):
        """GHZ(3) → equal superposition of |000⟩ and |111⟩."""
        sv, _ = simulate(ghz(3))
        assert abs(abs(sv[0]) - 1 / math.sqrt(2)) < 1e-10
        assert abs(abs(sv[7]) - 1 / math.sqrt(2)) < 1e-10
        assert np.allclose(np.abs(sv[1:7]), 0, atol=1e-10)

    def test_op_count(self):
        for n in [2, 3, 5]:
            c = ghz(n)
            assert len(c.ops) == n  # 1 H + (n-1) CX


class TestGroverOracle:
    def test_2q_phase_flip(self):
        """Oracle marks |11⟩ → that amplitude gets phase -1."""
        c = Circuit(2)
        c.h(0); c.h(1)
        oracle = grover_oracle(2, [0b11])
        for op in oracle.ops:
            c.ops.append(op)
        sv, _ = simulate(c)
        # |11⟩ = index 3 should be negative, others positive
        assert sv[3].real < 0
        assert all(sv[i].real > 0 for i in [0, 1, 2])

    def test_3q_phase_flip(self):
        """Oracle marks |101⟩ → that amplitude gets phase -1."""
        c = Circuit(3)
        for q in range(3):
            c.h(q)
        oracle = grover_oracle(3, [0b101])
        for op in oracle.ops:
            c.ops.append(op)
        sv, _ = simulate(c)
        # |101⟩ with MSB ordering: q0=1,q1=0,q2=1 → index 5
        assert sv[5].real < 0

    def test_multiple_marks(self):
        """Marking two states flips both relative to unmarked."""
        c = Circuit(2)
        c.h(0); c.h(1)
        oracle = grover_oracle(2, [0b00, 0b11])
        for op in oracle.ops:
            c.ops.append(op)
        sv, _ = simulate(c)
        # Marked states (0,3) should have opposite sign to unmarked (1,2)
        assert np.sign(sv[0].real) == np.sign(sv[3].real)
        assert np.sign(sv[1].real) == np.sign(sv[2].real)
        assert np.sign(sv[0].real) != np.sign(sv[1].real)


class TestHEA:
    def test_parameters(self):
        c = hardware_efficient_ansatz(3, 2)
        assert c.is_parameterized
        assert len(c.parameters) == 2 * 3 * 2  # depth * qubits * 2 (ry+rz)

    def test_bind(self):
        c = hardware_efficient_ansatz(3, 2)
        values = {p.name: 0.1 for p in c.parameters}
        bound = c.bind(values)
        assert not bound.is_parameterized


class TestQAOA:
    def test_structure(self):
        """Triangle graph → 3 RZZ + 3 RX per layer."""
        graph = [(0, 1), (1, 2), (0, 2)]
        c = qaoa_mixer(graph, p=1)
        rzz_ops = [op for op in c.ops if op.gate == Gate.RZZ]
        rx_ops = [op for op in c.ops if op.gate == Gate.RX]
        assert len(rzz_ops) == 3
        assert len(rx_ops) == 3
        assert c.is_parameterized

    def test_layers(self):
        """p=2 doubles cost+mixer layers."""
        graph = [(0, 1), (1, 2)]
        c1 = qaoa_mixer(graph, p=1)
        c2 = qaoa_mixer(graph, p=2)
        rzz1 = sum(1 for op in c1.ops if op.gate == Gate.RZZ)
        rzz2 = sum(1 for op in c2.ops if op.gate == Gate.RZZ)
        assert rzz2 == 2 * rzz1
