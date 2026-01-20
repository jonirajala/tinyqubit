"""
Tests for compilation report and circuit visualization.
"""
import pytest
from tinyqubit.ir import Circuit, Gate
from tinyqubit.target import Target
from tinyqubit.compile import transpile


# =============================================================================
# Test Fixtures
# =============================================================================

def line_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, i + 1) for i in range(n - 1))

def all_to_all_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, j) for i in range(n) for j in range(i + 1, n))

@pytest.fixture
def line_target():
    return Target(
        n_qubits=5,
        edges=line_topology(5),
        basis_gates=frozenset({Gate.RX, Gate.RZ, Gate.CX}),
        name="line_5"
    )

@pytest.fixture
def all_to_all_target():
    return Target(
        n_qubits=5,
        edges=all_to_all_topology(5),
        basis_gates=frozenset({Gate.RX, Gate.RZ, Gate.CX}),
        name="all_to_all_5"
    )

@pytest.fixture
def bell_circuit():
    return Circuit(2).h(0).cx(0, 1)


# =============================================================================
# Transpile Tests
# =============================================================================

class TestTranspile:

    def test_returns_circuit(self, line_target):
        c = Circuit(3).h(0).cx(0, 1)
        result = transpile(c, line_target)
        assert isinstance(result, Circuit)

    def test_verbosity_prints(self, line_target, capsys):
        c = Circuit(3).h(0).cx(0, 1)
        transpile(c, line_target, verbosity=2)
        captured = capsys.readouterr()
        assert "TinyQubit Compilation Report" in captured.out
        assert "SUMMARY" in captured.out

    def test_verbosity_0_silent(self, line_target, capsys):
        c = Circuit(3).h(0).cx(0, 1)
        transpile(c, line_target, verbosity=0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbosity_1_minimal(self, line_target, capsys):
        c = Circuit(3).h(0).cx(0, 1)
        transpile(c, line_target, verbosity=1)
        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "PASSES" not in captured.out

    def test_verbosity_3_verbose(self, line_target, capsys):
        c = Circuit(3).h(0).cx(0, 1)
        transpile(c, line_target, verbosity=3)
        captured = capsys.readouterr()
        assert "OPS" in captured.out


# =============================================================================
# Circuit.draw() Tests
# =============================================================================

class TestCircuitDraw:

    def test_empty_circuit(self, capsys):
        c = Circuit(2)
        c.draw()
        output = capsys.readouterr().out
        assert "q0:" in output
        assert "q1:" in output

    def test_single_qubit_gates(self, capsys):
        c = Circuit(2).h(0).x(1)
        c.draw()
        output = capsys.readouterr().out
        assert "H" in output
        assert "X" in output

    def test_cx_gate_rendering(self, capsys):
        c = Circuit(2).cx(0, 1)
        c.draw()
        output = capsys.readouterr().out
        assert "●" in output
        assert "X" in output
        assert "│" in output  # connector line

    def test_cz_gate_rendering(self, capsys):
        c = Circuit(2).cz(0, 1)
        c.draw()
        output = capsys.readouterr().out
        assert output.count("●") == 2

    def test_swap_gate_rendering(self, capsys):
        c = Circuit(2).swap(0, 1)
        c.draw()
        output = capsys.readouterr().out
        assert "╳" in output

    def test_multi_qubit_gate_connectors(self, capsys):
        c = Circuit(3).cx(0, 2)
        c.draw()
        output = capsys.readouterr().out
        assert "│" in output

    def test_rotation_gates(self, capsys):
        c = Circuit(1).rz(0, 1.57)
        c.draw()
        output = capsys.readouterr().out
        assert "RZ" in output

    def test_measure_gate(self, capsys):
        c = Circuit(1).measure(0)
        c.draw()
        output = capsys.readouterr().out
        assert "M" in output

    def test_complex_circuit(self, capsys):
        c = Circuit(3).h(0).cx(0, 1).cx(1, 2).rz(2, 0.5).measure(0)
        c.draw()
        output = capsys.readouterr().out
        assert "q0:" in output
        assert "q1:" in output
        assert "q2:" in output
        assert "H" in output
        assert "M" in output

    def test_bell_circuit_draw(self, bell_circuit, capsys):
        bell_circuit.draw()
        output = capsys.readouterr().out
        assert "H" in output
        assert "●" in output or "X" in output

    def test_draw_after_transpile(self, line_target, capsys):
        c = Circuit(3).h(0).cx(0, 2)
        result = transpile(c, line_target)
        result.draw()
        output = capsys.readouterr().out
        assert "q0:" in output
