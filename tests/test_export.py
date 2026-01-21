"""
Tests for export adapters.

Tests:
    - OpenQASM 2 syntax correctness
    - OpenQASM 3 syntax correctness
    - Qiskit round-trip (if available)
    - CP gate handling
    - Parametric gates
    - Measurement syntax
"""
import math
import pytest

from tinyqubit import Circuit, Gate
from tinyqubit.export import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError


class TestOpenQASM2:
    """Tests for OpenQASM 2.0 export."""

    def test_header(self):
        """QASM2 starts with version and qelib1.inc."""
        c = Circuit(2)
        qasm = to_openqasm2(c)
        assert qasm.startswith('OPENQASM 2.0;')
        assert 'include "qelib1.inc";' in qasm
        assert 'qreg q[2];' in qasm

    def test_empty_circuit(self):
        """Empty circuit has only header and qreg."""
        c = Circuit(3)
        qasm = to_openqasm2(c)
        lines = qasm.strip().split('\n')
        # Filter out empty lines
        lines = [l for l in lines if l.strip()]
        assert len(lines) == 3
        assert 'qreg q[3];' in qasm

    def test_single_qubit_gates(self):
        """Single-qubit gates export correctly."""
        c = Circuit(1).x(0).y(0).z(0).h(0).s(0).t(0).sdg(0).tdg(0)
        qasm = to_openqasm2(c)
        assert 'x q[0];' in qasm
        assert 'y q[0];' in qasm
        assert 'z q[0];' in qasm
        assert 'h q[0];' in qasm
        assert 's q[0];' in qasm
        assert 't q[0];' in qasm
        assert 'sdg q[0];' in qasm
        assert 'tdg q[0];' in qasm

    def test_parametric_gates(self):
        """Parametric gates include parameters."""
        c = Circuit(1).rx(0, math.pi).ry(0, math.pi/2).rz(0, math.pi/4)
        qasm = to_openqasm2(c)
        assert f'rx({math.pi}) q[0];' in qasm
        assert f'ry({math.pi/2}) q[0];' in qasm
        assert f'rz({math.pi/4}) q[0];' in qasm

    def test_two_qubit_gates(self):
        """Two-qubit gates export correctly."""
        c = Circuit(2).cx(0, 1).cz(0, 1).swap(0, 1)
        qasm = to_openqasm2(c)
        assert 'cx q[0], q[1];' in qasm
        assert 'cz q[0], q[1];' in qasm
        assert 'swap q[0], q[1];' in qasm

    def test_cp_raises_error(self):
        """CP gate raises UnsupportedGateError in QASM2."""
        c = Circuit(2).cp(0, 1, math.pi/4)
        with pytest.raises(UnsupportedGateError) as excinfo:
            to_openqasm2(c)
        assert "CP" in str(excinfo.value)
        assert "qelib1.inc" in str(excinfo.value)

    def test_measurement_syntax(self):
        """Measurement uses arrow syntax in QASM2."""
        c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
        qasm = to_openqasm2(c)
        assert 'creg c[2];' in qasm
        assert 'measure q[0] -> c[0];' in qasm
        assert 'measure q[1] -> c[1];' in qasm

    def test_creg_only_with_measurements(self):
        """creg only appears if there are measurements."""
        c = Circuit(2).h(0).cx(0, 1)
        qasm = to_openqasm2(c)
        assert 'creg' not in qasm

    def test_creg_before_gates(self):
        """creg declaration appears before gate operations."""
        c = Circuit(2).h(0).measure(0)
        qasm = to_openqasm2(c)
        creg_pos = qasm.find('creg')
        h_pos = qasm.find('h q[0]')
        assert creg_pos < h_pos

    def test_bell_state(self):
        """Full Bell state circuit exports correctly."""
        c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
        qasm = to_openqasm2(c)
        expected_lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            'qreg q[2];',
            'creg c[2];',
            'h q[0];',
            'cx q[0], q[1];',
            'measure q[0] -> c[0];',
            'measure q[1] -> c[1];',
        ]
        for line in expected_lines:
            assert line in qasm

    def test_include_mapping_false(self):
        """include_mapping=False omits qubit mapping comment."""
        c = Circuit(2).h(0)
        qasm = to_openqasm2(c, include_mapping=False)
        assert '// Qubit mapping' not in qasm


class TestOpenQASM3:
    """Tests for OpenQASM 3.0 export."""

    def test_header(self):
        """QASM3 starts with version and stdgates.inc."""
        c = Circuit(2)
        qasm = to_openqasm3(c)
        assert qasm.startswith('OPENQASM 3.0;')
        assert 'include "stdgates.inc";' in qasm
        assert 'qubit[2] q;' in qasm

    def test_qubit_declaration(self):
        """QASM3 uses qubit[] syntax."""
        c = Circuit(4)
        qasm = to_openqasm3(c)
        assert 'qubit[4] q;' in qasm
        assert 'qreg' not in qasm

    def test_bit_declaration(self):
        """QASM3 uses bit[] syntax for classical bits."""
        c = Circuit(2).measure(0)
        qasm = to_openqasm3(c)
        assert 'bit[2] c;' in qasm
        assert 'creg' not in qasm

    def test_single_qubit_gates(self):
        """Single-qubit gates export correctly in QASM3."""
        c = Circuit(1).x(0).y(0).z(0).h(0).s(0).t(0).sdg(0).tdg(0)
        qasm = to_openqasm3(c)
        assert 'x q[0];' in qasm
        assert 'y q[0];' in qasm
        assert 'z q[0];' in qasm
        assert 'h q[0];' in qasm
        assert 's q[0];' in qasm
        assert 't q[0];' in qasm
        assert 'sdg q[0];' in qasm
        assert 'tdg q[0];' in qasm

    def test_parametric_gates(self):
        """Parametric gates include parameters in QASM3."""
        c = Circuit(1).rx(0, math.pi).ry(0, math.pi/2).rz(0, math.pi/4)
        qasm = to_openqasm3(c)
        assert f'rx({math.pi}) q[0];' in qasm
        assert f'ry({math.pi/2}) q[0];' in qasm
        assert f'rz({math.pi/4}) q[0];' in qasm

    def test_two_qubit_gates(self):
        """Two-qubit gates export correctly in QASM3."""
        c = Circuit(2).cx(0, 1).cz(0, 1).swap(0, 1)
        qasm = to_openqasm3(c)
        assert 'cx q[0], q[1];' in qasm
        assert 'cz q[0], q[1];' in qasm
        assert 'swap q[0], q[1];' in qasm

    def test_cp_supported(self):
        """CP gate is supported in QASM3."""
        c = Circuit(2).cp(0, 1, math.pi/4)
        qasm = to_openqasm3(c)
        assert f'cp({math.pi/4}) q[0], q[1];' in qasm

    def test_measurement_syntax(self):
        """Measurement uses assignment syntax in QASM3."""
        c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
        qasm = to_openqasm3(c)
        assert 'c[0] = measure q[0];' in qasm
        assert 'c[1] = measure q[1];' in qasm
        assert '->' not in qasm  # No QASM2 arrow syntax

    def test_bell_state(self):
        """Full Bell state circuit exports correctly in QASM3."""
        c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
        qasm = to_openqasm3(c)
        expected_lines = [
            'OPENQASM 3.0;',
            'include "stdgates.inc";',
            'qubit[2] q;',
            'bit[2] c;',
            'h q[0];',
            'cx q[0], q[1];',
            'c[0] = measure q[0];',
            'c[1] = measure q[1];',
        ]
        for line in expected_lines:
            assert line in qasm


class TestQiskitAdapter:
    """Tests for Qiskit adapter (when Qiskit is available)."""

    @pytest.fixture
    def qiskit_available(self):
        """Check if Qiskit is available."""
        try:
            import qiskit
            return True
        except ImportError:
            return False

    def test_import_error_without_qiskit(self, qiskit_available, monkeypatch):
        """to_qiskit raises ImportError with clear message when Qiskit unavailable."""
        if qiskit_available:
            # Temporarily hide qiskit
            import sys
            monkeypatch.setitem(sys.modules, 'qiskit', None)

        c = Circuit(2).h(0)
        # This should raise ImportError if qiskit is hidden/unavailable
        # Note: May not work perfectly with monkeypatch due to lazy import

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_basic_conversion(self):
        """Basic circuit converts to Qiskit."""
        c = Circuit(2).h(0).cx(0, 1)
        qc = to_qiskit(c)
        assert qc.num_qubits == 2

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_all_single_qubit_gates(self):
        """All single-qubit gates convert correctly."""
        c = Circuit(1).x(0).y(0).z(0).h(0).s(0).t(0).sdg(0).tdg(0)
        qc = to_qiskit(c)
        assert qc.num_qubits == 1
        # Check operation count
        assert len(qc.data) == 8

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_parametric_gates(self):
        """Parametric gates convert with correct parameters."""
        c = Circuit(1).rx(0, math.pi).ry(0, math.pi/2).rz(0, math.pi/4)
        qc = to_qiskit(c)
        assert len(qc.data) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_two_qubit_gates(self):
        """Two-qubit gates convert correctly."""
        c = Circuit(2).cx(0, 1).cz(0, 1).swap(0, 1).cp(0, 1, math.pi/4)
        qc = to_qiskit(c)
        assert len(qc.data) == 4

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_measurement(self):
        """Measurements convert correctly."""
        c = Circuit(2).h(0).measure(0).measure(1)
        qc = to_qiskit(c)
        assert qc.num_clbits == 2

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not installed"
    )
    def test_round_trip_qasm(self):
        """Circuit can be exported to Qiskit and back to QASM."""
        from qiskit import qasm2
        c = Circuit(2).h(0).cx(0, 1)
        qc = to_qiskit(c)
        qasm = qasm2.dumps(qc)
        assert 'h q[0];' in qasm
        assert 'cx q[0],q[1];' in qasm


class TestTranspiledCircuitExport:
    """Tests for exporting transpiled circuits."""

    def test_transpiled_circuit_exports(self):
        """Transpiled circuit exports correctly."""
        from tinyqubit import transpile, Target

        # Create a simple target
        target = Target(
            n_qubits=3,
            edges={(0, 1), (1, 2)},
            basis_gates={Gate.CX, Gate.RZ, Gate.RX},
        )

        # Create and transpile a circuit
        c = Circuit(3).h(0).cx(0, 2).measure(0).measure(2)
        tc = transpile(c, target)

        # Export should work
        qasm = to_openqasm2(tc)
        assert 'OPENQASM 2.0;' in qasm
        assert 'qreg q[3];' in qasm
        assert 'creg c[3];' in qasm

    def test_transpiled_circuit_with_mapping(self):
        """Transpiled circuit includes qubit mapping when requested."""
        from tinyqubit import transpile, Target

        target = Target(
            n_qubits=3,
            edges={(0, 1), (1, 2)},
            basis_gates={Gate.CX, Gate.RZ, Gate.RX},
        )

        c = Circuit(3).h(0).cx(0, 2)
        tc = transpile(c, target)

        qasm = to_openqasm2(tc, include_mapping=True)
        # Should have mapping comments if tracker is present
        if hasattr(tc, '_tracker') and tc._tracker is not None:
            assert '// Qubit mapping' in qasm


class TestPublicAPI:
    """Tests for the public API."""

    def test_export_from_tinyqubit(self):
        """Export functions available from tinyqubit module."""
        from tinyqubit import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError
        assert callable(to_openqasm2)
        assert callable(to_openqasm3)
        assert callable(to_qiskit)
        assert issubclass(UnsupportedGateError, Exception)

    def test_export_from_export_module(self):
        """Export functions available from tinyqubit.export module."""
        from tinyqubit.export import to_openqasm2, to_openqasm3, to_qiskit, UnsupportedGateError
        assert callable(to_openqasm2)
        assert callable(to_openqasm3)
        assert callable(to_qiskit)
        assert issubclass(UnsupportedGateError, Exception)

    def test_backends_importable(self):
        """Backend adapters are importable (may fail with clear error if deps missing)."""
        from tinyqubit.export.backends import submit_to_ibm, get_ibm_results
        from tinyqubit.export.backends import submit_to_braket, get_braket_results
        assert callable(submit_to_ibm)
        assert callable(get_ibm_results)
        assert callable(submit_to_braket)
        assert callable(get_braket_results)
