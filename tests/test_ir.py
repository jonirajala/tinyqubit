"""
Tests for core IR types.

Tests:
    - Gate enum completeness and properties
    - Operation creation and immutability
    - Circuit construction and method chaining
    - OpenQASM export
"""
import pytest
from tinyqubit.ir import Gate, Operation, Circuit


# =============================================================================
# Gate Enum Tests
# =============================================================================

def test_gate_count():
    """We have exactly 16 primitive gates"""
    assert len(Gate) == 16


def test_gate_n_qubits():
    """Single vs two-qubit gates"""
    single = [Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.T, Gate.SDG, Gate.TDG,
              Gate.RX, Gate.RY, Gate.RZ, Gate.MEASURE]
    two = [Gate.CX, Gate.CZ, Gate.CP, Gate.SWAP]

    for g in single:
        assert g.n_qubits == 1, f"{g} should be 1-qubit"
    for g in two:
        assert g.n_qubits == 2, f"{g} should be 2-qubit"


def test_gate_n_params():
    """Only rotation gates have parameters"""
    parametric = [Gate.RX, Gate.RY, Gate.RZ]
    non_parametric = [Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.T,
                     Gate.CX, Gate.CZ, Gate.MEASURE]

    for g in parametric:
        assert g.n_params == 1, f"{g} should have 1 param"
    for g in non_parametric:
        assert g.n_params == 0, f"{g} should have 0 params"


# =============================================================================
# Operation Dataclass Tests
# =============================================================================

def test_operation_creation():
    """Basic operation creation"""
    op = Operation(Gate.X, (0,))
    assert op.gate == Gate.X
    assert op.qubits == (0,)
    assert op.params == ()


def test_operation_with_params():
    """Parametric operation"""
    op = Operation(Gate.RZ, (0,), (1.57,))
    assert op.gate == Gate.RZ
    assert op.qubits == (0,)
    assert op.params == (1.57,)


def test_operation_two_qubit():
    """Two-qubit operation"""
    op = Operation(Gate.CX, (0, 1))
    assert op.qubits == (0, 1)


def test_operation_frozen():
    """Operations are immutable"""
    op = Operation(Gate.X, (0,))
    with pytest.raises(AttributeError):
        op.gate = Gate.Y


def test_operation_equality():
    """Same operations are equal"""
    op1 = Operation(Gate.X, (0,))
    op2 = Operation(Gate.X, (0,))
    op3 = Operation(Gate.X, (1,))

    assert op1 == op2
    assert op1 != op3


def test_operation_hashable():
    """Operations can be used in sets/dicts"""
    op1 = Operation(Gate.X, (0,))
    op2 = Operation(Gate.X, (0,))
    op3 = Operation(Gate.X, (1,))

    s = {op1, op2, op3}
    assert len(s) == 2  # op1 and op2 are equal


# =============================================================================
# Circuit Builder Tests
# =============================================================================

def test_circuit_init():
    """Circuit initializes correctly"""
    c = Circuit(3)
    assert c.n_qubits == 3
    assert c.ops == []


def test_circuit_single_gate():
    """Adding a single gate"""
    c = Circuit(1).x(0)
    assert len(c.ops) == 1
    assert c.ops[0] == Operation(Gate.X, (0,))


def test_circuit_chaining():
    """Gate methods return self for chaining"""
    c = Circuit(2)
    result = c.h(0).cx(0, 1)
    assert result is c
    assert len(c.ops) == 2


def test_circuit_all_single_qubit_gates():
    """All single-qubit gate methods work"""
    c = Circuit(1)
    c.x(0).y(0).z(0)
    c.h(0).s(0).t(0)

    assert len(c.ops) == 6
    gates = [op.gate for op in c.ops]
    assert gates == [Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.T]


def test_circuit_rotation_gates():
    """Rotation gate methods work with parameters"""
    c = Circuit(1)
    c.rx(0, 1.1).ry(0, 2.2).rz(0, 3.3)

    assert len(c.ops) == 3
    assert c.ops[0] == Operation(Gate.RX, (0,), (1.1,))
    assert c.ops[1] == Operation(Gate.RY, (0,), (2.2,))
    assert c.ops[2] == Operation(Gate.RZ, (0,), (3.3,))


def test_circuit_two_qubit_gates():
    """Two-qubit gate methods work"""
    c = Circuit(2)
    c.cx(0, 1).cz(1, 0)

    assert len(c.ops) == 2
    assert c.ops[0] == Operation(Gate.CX, (0, 1))
    assert c.ops[1] == Operation(Gate.CZ, (1, 0))


def test_circuit_measure():
    """Measure method works"""
    c = Circuit(2).measure(0).measure(1)

    assert len(c.ops) == 2
    assert c.ops[0] == Operation(Gate.MEASURE, (0,))
    assert c.ops[1] == Operation(Gate.MEASURE, (1,))


def test_circuit_bell_state():
    """Classic Bell state circuit"""
    c = Circuit(2).h(0).cx(0, 1)

    assert len(c.ops) == 2
    assert c.ops[0].gate == Gate.H
    assert c.ops[1].gate == Gate.CX
    assert c.ops[1].qubits == (0, 1)


def test_circuit_ghz_state():
    """GHZ state circuit"""
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2)

    assert len(c.ops) == 3
    assert c.ops[0].gate == Gate.H
    assert c.ops[1].qubits == (0, 1)
    assert c.ops[2].qubits == (1, 2)


# =============================================================================
# OpenQASM Export Tests
# =============================================================================

def test_qasm_header():
    """QASM header is correct"""
    c = Circuit(2)
    qasm = c.to_openqasm()

    lines = qasm.split('\n')
    assert lines[0] == 'OPENQASM 2.0;'
    assert lines[1] == 'include "qelib1.inc";'
    assert lines[2] == 'qreg q[2];'


def test_qasm_empty_circuit():
    """Empty circuit exports valid QASM"""
    c = Circuit(3)
    qasm = c.to_openqasm()

    assert 'OPENQASM 2.0;' in qasm
    assert 'qreg q[3];' in qasm
    assert 'creg' not in qasm  # No measurements = no creg


def test_qasm_single_qubit_gates():
    """Single qubit gates export correctly"""
    c = Circuit(1).h(0).x(0).y(0).z(0).s(0).t(0)
    qasm = c.to_openqasm()

    assert 'h q[0];' in qasm
    assert 'x q[0];' in qasm
    assert 'y q[0];' in qasm
    assert 'z q[0];' in qasm
    assert 's q[0];' in qasm
    assert 't q[0];' in qasm


def test_qasm_two_qubit_gates():
    """Two qubit gates export correctly"""
    c = Circuit(2).cx(0, 1).cz(1, 0)
    qasm = c.to_openqasm()

    assert 'cx q[0], q[1];' in qasm
    assert 'cz q[1], q[0];' in qasm


def test_qasm_parametric_gates():
    """Rotation gates include parameters"""
    c = Circuit(1).rx(0, 1.5).ry(0, 2.5).rz(0, 3.5)
    qasm = c.to_openqasm()

    assert 'rx(1.5) q[0];' in qasm
    assert 'ry(2.5) q[0];' in qasm
    assert 'rz(3.5) q[0];' in qasm


def test_qasm_measurement_adds_creg():
    """Measurements trigger creg declaration"""
    c = Circuit(2).measure(0)
    qasm = c.to_openqasm()

    assert 'creg c[2];' in qasm
    assert 'measure q[0] -> c[0];' in qasm


def test_qasm_multiple_measurements():
    """Multiple measurements work"""
    c = Circuit(3).measure(0).measure(1).measure(2)
    qasm = c.to_openqasm()

    assert 'creg c[3];' in qasm
    assert 'measure q[0] -> c[0];' in qasm
    assert 'measure q[1] -> c[1];' in qasm
    assert 'measure q[2] -> c[2];' in qasm


def test_qasm_creg_before_gates():
    """creg appears before gates in output"""
    c = Circuit(2).h(0).measure(0)
    lines = c.to_openqasm().split('\n')

    creg_idx = next(i for i, l in enumerate(lines) if 'creg' in l)
    gate_idx = next(i for i, l in enumerate(lines) if l.startswith('h '))

    assert creg_idx < gate_idx


def test_qasm_no_creg_without_measure():
    """No creg if no measurements"""
    c = Circuit(3).h(0).cx(0, 1).cx(1, 2)
    qasm = c.to_openqasm()

    assert 'creg' not in qasm


def test_qasm_full_circuit():
    """Full circuit with all gate types"""
    c = Circuit(2)
    c.h(0).x(1).cx(0, 1).rz(1, 0.5).measure(0).measure(1)
    qasm = c.to_openqasm()

    # Check structure
    assert 'OPENQASM 2.0;' in qasm
    assert 'qreg q[2];' in qasm
    assert 'creg c[2];' in qasm
    assert 'h q[0];' in qasm
    assert 'x q[1];' in qasm
    assert 'cx q[0], q[1];' in qasm
    assert 'rz(0.5) q[1];' in qasm
    assert 'measure q[0] -> c[0];' in qasm
    assert 'measure q[1] -> c[1];' in qasm


def test_qasm_different_qubit_indices():
    """Gates on different qubits export correct indices"""
    c = Circuit(4).h(0).h(1).h(2).h(3)
    qasm = c.to_openqasm()

    assert 'h q[0];' in qasm
    assert 'h q[1];' in qasm
    assert 'h q[2];' in qasm
    assert 'h q[3];' in qasm


# =============================================================================
# Qiskit Round-Trip (Optional)
# =============================================================================

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
def test_qasm_parseable_by_qiskit():
    """Qiskit can parse our QASM output"""
    c = Circuit(2).h(0).cx(0, 1).rz(1, 0.5).measure(0).measure(1)
    qasm = c.to_openqasm()

    # This will raise if QASM is invalid
    qc = QuantumCircuit.from_qasm_str(qasm)

    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")
def test_qasm_bell_state_qiskit():
    """Bell state QASM parses correctly"""
    c = Circuit(2).h(0).cx(0, 1)
    qasm = c.to_openqasm()

    qc = QuantumCircuit.from_qasm_str(qasm)
    assert qc.num_qubits == 2
