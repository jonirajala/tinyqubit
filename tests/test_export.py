"""
Tests for export adapters.

Tests:
    - OpenQASM 2 syntax correctness
    - OpenQASM 3 syntax correctness
    - CP gate handling
    - Parametric gates
    - Measurement syntax
"""
import math
import numpy as np
import pytest

from tinyqubit import Circuit, Gate
from tinyqubit.export import to_openqasm2, to_openqasm3, from_openqasm2, UnsupportedGateError


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

    def test_ccx_supported(self):
        """CCX gate exports correctly in QASM2."""
        c = Circuit(3).ccx(0, 1, 2)
        qasm = to_openqasm2(c)
        assert 'ccx q[0], q[1], q[2];' in qasm

    def test_ccz_raises_error(self):
        """CCZ gate raises UnsupportedGateError in QASM2."""
        c = Circuit(3).ccz(0, 1, 2)
        with pytest.raises(UnsupportedGateError) as excinfo:
            to_openqasm2(c)
        assert "CCZ" in str(excinfo.value)

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

    def test_ccx_supported(self):
        """CCX gate is supported in QASM3."""
        c = Circuit(3).ccx(0, 1, 2)
        qasm = to_openqasm3(c)
        assert 'ccx q[0], q[1], q[2];' in qasm

    def test_ccz_supported(self):
        """CCZ gate is supported in QASM3."""
        c = Circuit(3).ccz(0, 1, 2)
        qasm = to_openqasm3(c)
        assert 'ccz q[0], q[1], q[2];' in qasm

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
        from tinyqubit import to_openqasm2, to_openqasm3, UnsupportedGateError
        assert callable(to_openqasm2)
        assert callable(to_openqasm3)
        assert issubclass(UnsupportedGateError, Exception)

    def test_export_from_export_module(self):
        """Export functions available from tinyqubit.export module."""
        from tinyqubit.export import to_openqasm2, to_openqasm3, UnsupportedGateError
        assert callable(to_openqasm2)
        assert callable(to_openqasm3)
        assert issubclass(UnsupportedGateError, Exception)

    def test_backends_importable(self):
        """Backend adapters are importable."""
        from tinyqubit.export.backends import submit_ibm, wait_ibm, list_ibm_backends, ibm_target
        from tinyqubit.export.backends import submit_to_braket, get_braket_results
        assert callable(submit_ibm)
        assert callable(wait_ibm)
        assert callable(list_ibm_backends)
        assert callable(ibm_target)
        assert callable(submit_to_braket)
        assert callable(get_braket_results)


class TestDynamicCircuitExport:
    """Tests for dynamic circuit features in QASM export."""

    def test_measure_explicit_classical_bit_qasm2(self):
        """QASM2 uses explicit classical bit from op.classical_bit."""
        c = Circuit(2, n_classical=3)
        c.measure(0, 2)  # Measure qubit 0 into classical bit 2
        qasm = to_openqasm2(c)
        assert 'creg c[3];' in qasm
        assert 'measure q[0] -> c[2];' in qasm

    def test_measure_explicit_classical_bit_qasm3(self):
        """QASM3 uses explicit classical bit from op.classical_bit."""
        c = Circuit(2, n_classical=3)
        c.measure(0, 2)
        qasm = to_openqasm3(c)
        assert 'bit[3] c;' in qasm
        assert 'c[2] = measure q[0];' in qasm

    def test_conditional_qasm2(self):
        """QASM2 exports conditionals with if statement."""
        c = Circuit(2)
        c.x(0).measure(0, 0)
        with c.c_if(0, 1):
            c.x(1)
        qasm = to_openqasm2(c)
        assert 'if (c[0] == 1) x q[1];' in qasm

    def test_conditional_qasm3(self):
        """QASM3 exports conditionals with braces."""
        c = Circuit(2)
        c.x(0).measure(0, 0)
        with c.c_if(0, 1):
            c.x(1)
        qasm = to_openqasm3(c)
        assert 'if (c[0] == 1) { x q[1]; }' in qasm

    def test_conditional_triggers_classical_register(self):
        """Classical register created when conditionals present (even without measure)."""
        c = Circuit(2, n_classical=1)
        # Assuming classical bit 0 was set externally or by prior circuit
        with c.c_if(0, 1):
            c.x(0)
        qasm = to_openqasm2(c)
        assert 'creg c[1];' in qasm

    def test_reset_qasm2(self):
        """QASM2 exports reset operation."""
        c = Circuit(1).x(0).reset(0)
        qasm = to_openqasm2(c)
        assert 'reset q[0];' in qasm

    def test_reset_qasm3(self):
        """QASM3 exports reset operation."""
        c = Circuit(1).x(0).reset(0)
        qasm = to_openqasm3(c)
        assert 'reset q[0];' in qasm

    def test_multi_param_gate_formatting(self):
        """Gates with multiple params format correctly."""
        # CP has one param, but test the formatting handles multiple
        c = Circuit(2).cp(0, 1, 0.5)
        qasm = to_openqasm3(c)  # CP only works in QASM3
        assert 'cp(0.5) q[0], q[1];' in qasm

    def test_teleportation_circuit_exports(self):
        """Full teleportation circuit with mid-circuit measurement exports correctly."""
        c = Circuit(3, n_classical=2)
        # Prepare Bell pair
        c.h(1).cx(1, 2)
        # Bell measurement
        c.cx(0, 1).h(0)
        c.measure(0, 0).measure(1, 1)
        # Conditional corrections
        with c.c_if(1, 1):
            c.x(2)
        with c.c_if(0, 1):
            c.z(2)

        qasm2 = to_openqasm2(c)
        assert 'creg c[2];' in qasm2
        assert 'measure q[0] -> c[0];' in qasm2
        assert 'measure q[1] -> c[1];' in qasm2
        assert 'if (c[1] == 1) x q[2];' in qasm2
        assert 'if (c[0] == 1) z q[2];' in qasm2

        qasm3 = to_openqasm3(c)
        assert 'bit[2] c;' in qasm3
        assert 'c[0] = measure q[0];' in qasm3
        assert 'if (c[1] == 1) { x q[2]; }' in qasm3


class TestOpenQASM2Import:
    """Tests for from_openqasm2() parser."""

    def test_round_trip_bell_state(self):
        """Bell state survives export → import round-trip."""
        c = Circuit(2).h(0).cx(0, 1)
        imported = from_openqasm2(to_openqasm2(c))
        assert imported.n_qubits == 2
        assert len(imported.ops) == len(c.ops)
        for orig, imp in zip(c.ops, imported.ops):
            assert orig.gate == imp.gate
            assert orig.qubits == imp.qubits

    def test_round_trip_all_1q_gates(self):
        """All single-qubit gates round-trip."""
        c = Circuit(1).x(0).y(0).z(0).h(0).s(0).t(0).sdg(0).tdg(0).sx(0)
        imported = from_openqasm2(to_openqasm2(c))
        assert [op.gate for op in imported.ops] == [op.gate for op in c.ops]

    def test_round_trip_parametric(self):
        """Parametric gates (RX/RY/RZ) round-trip with correct params."""
        c = Circuit(1).rx(0, math.pi).ry(0, math.pi / 2).rz(0, 0.123)
        imported = from_openqasm2(to_openqasm2(c))
        for orig, imp in zip(c.ops, imported.ops):
            assert orig.gate == imp.gate
            assert math.isclose(orig.params[0], imp.params[0])

    def test_round_trip_multi_qubit(self):
        """Multi-qubit gates (CX, CZ, SWAP, CCX) round-trip."""
        c = Circuit(3).cx(0, 1).cz(0, 2).swap(1, 2).ccx(0, 1, 2)
        imported = from_openqasm2(to_openqasm2(c))
        for orig, imp in zip(c.ops, imported.ops):
            assert orig.gate == imp.gate
            assert orig.qubits == imp.qubits

    def test_round_trip_measure_reset(self):
        """Measure and reset round-trip."""
        c = Circuit(2).x(0).measure(0, 0).reset(1)
        imported = from_openqasm2(to_openqasm2(c))
        assert imported.ops[1].gate == Gate.MEASURE
        assert imported.ops[1].classical_bit == 0
        assert imported.ops[2].gate == Gate.RESET

    def test_round_trip_conditional(self):
        """Conditional operations round-trip."""
        c = Circuit(2, n_classical=2)
        c.x(0).measure(0, 0)
        with c.c_if(0, 1):
            c.x(1)
        imported = from_openqasm2(to_openqasm2(c))
        cond_op = [op for op in imported.ops if op.condition is not None][0]
        assert cond_op.gate == Gate.X
        assert cond_op.condition == (0, 1)

    def test_pi_expressions(self):
        """Parser handles pi expressions from external QASM."""
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nrx(pi/2) q[0];\nry(3*pi/4) q[0];\nrz(-pi) q[0];'
        c = from_openqasm2(qasm)
        assert math.isclose(c.ops[0].params[0], np.pi / 2)
        assert math.isclose(c.ops[1].params[0], 3 * np.pi / 4)
        assert math.isclose(c.ops[2].params[0], -np.pi)

    def test_external_qasm(self):
        """Hand-written QASM (not from our export) parses correctly."""
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""
        c = from_openqasm2(qasm)
        assert c.n_qubits == 3
        assert c.n_classical == 3
        assert len(c.ops) == 6
        assert c.ops[0].gate == Gate.H
        assert c.ops[1].gate == Gate.CX
        assert c.ops[1].qubits == (0, 1)

    def test_comments_ignored(self):
        """Comments are stripped during parsing."""
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n// a comment\nh q[0]; // inline comment'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 1
        assert c.ops[0].gate == Gate.H

    def test_missing_qreg_raises(self):
        """Missing qreg raises ValueError."""
        with pytest.raises(ValueError, match="No qreg"):
            from_openqasm2('OPENQASM 2.0;\nh q[0];')

    def test_unknown_gate_raises(self):
        """Unknown gate name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gate"):
            from_openqasm2('OPENQASM 2.0;\nqreg q[1];\nfoo q[0];')

    def test_barrier_skipped(self):
        """Barrier lines are silently skipped."""
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\nbarrier q[0], q[1];\ncx q[0], q[1];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 2

    def test_symmetric_canonicalization(self):
        """CZ and SWAP qubits are canonicalized on import."""
        qasm = 'OPENQASM 2.0;\nqreg q[2];\ncz q[1], q[0];\nswap q[1], q[0];'
        c = from_openqasm2(qasm)
        assert c.ops[0].qubits == (0, 1)
        assert c.ops[1].qubits == (0, 1)

    def test_u1_decomposition(self):
        """u1(λ) decomposes to rz(λ)."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nu1(pi/4) q[0];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 1
        assert c.ops[0].gate == Gate.RZ
        assert math.isclose(c.ops[0].params[0], np.pi / 4)

    def test_u2_decomposition(self):
        """u2(φ,λ) decomposes to rz(λ), ry(π/2), rz(φ)."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nu2(pi/2, pi/4) q[0];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 3
        assert [op.gate for op in c.ops] == [Gate.RZ, Gate.RY, Gate.RZ]
        assert math.isclose(c.ops[0].params[0], np.pi / 4)
        assert math.isclose(c.ops[1].params[0], np.pi / 2)
        assert math.isclose(c.ops[2].params[0], np.pi / 2)

    def test_u3_decomposition(self):
        """u3(θ,φ,λ) decomposes to rz(λ), ry(θ), rz(φ)."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nu3(pi, pi/2, pi/4) q[0];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 3
        assert [op.gate for op in c.ops] == [Gate.RZ, Gate.RY, Gate.RZ]
        assert math.isclose(c.ops[0].params[0], np.pi / 4)   # λ
        assert math.isclose(c.ops[1].params[0], np.pi)        # θ
        assert math.isclose(c.ops[2].params[0], np.pi / 2)    # φ

    def test_u3_unitary_correctness(self):
        """u3 decomposition produces the correct unitary (up to global phase)."""
        from tinyqubit import to_unitary
        # u3(π/2, π/3, π/6) on 1 qubit
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nu3(1.5707963267948966, 1.0471975511965976, 0.5235987755982988) q[0];'
        c = from_openqasm2(qasm)
        U = to_unitary(c)
        # Compare against analytical U3 matrix
        theta, phi, lam = np.pi / 2, np.pi / 3, np.pi / 6
        expected = np.array([
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
        ])
        # Equal up to global phase
        phase = U[0, 0] / expected[0, 0] if abs(expected[0, 0]) > 1e-10 else U[1, 0] / expected[1, 0]
        assert np.allclose(U, phase * expected, atol=1e-10)

    def test_id_gate_skipped(self):
        """Identity gate is silently skipped."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nid q[0];\nh q[0];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 1
        assert c.ops[0].gate == Gate.H

    def test_multi_creg(self):
        """Multiple creg declarations accumulate."""
        qasm = 'OPENQASM 2.0;\nqreg q[2];\ncreg a[1];\ncreg b[1];\nmeasure q[0] -> a[0];\nmeasure q[1] -> b[0];'
        c = from_openqasm2(qasm)
        assert c.n_classical == 2

    def test_custom_gate_def_skipped(self):
        """Custom gate definitions are skipped (usage raises if unknown)."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\ngate mygate a {\n  h a;\n}\nh q[0];'
        c = from_openqasm2(qasm)
        assert len(c.ops) == 1

    def test_pi_multiplication_expressions(self):
        """Parser handles pi*factor expressions from Cirq-generated QASM."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\nrx(pi*0.5) q[0];\nry(pi*-0.25) q[0];'
        c = from_openqasm2(qasm)
        assert math.isclose(c.ops[0].params[0], np.pi * 0.5)
        assert math.isclose(c.ops[1].params[0], np.pi * -0.25)
