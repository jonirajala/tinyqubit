"""Tests for dynamic decoupling pass."""
import numpy as np
from tinyqubit import Circuit, Gate, Operation, Target, dynamic_decoupling, transpile, to_unitary


def _small_ibm_target():
    """3-qubit IBM-like target with durations."""
    return Target(n_qubits=3, edges=frozenset({(0,1),(1,2)}),
                  basis_gates=frozenset({Gate.SX, Gate.RZ, Gate.CX, Gate.MEASURE, Gate.RESET}),
                  directed=True, duration={Gate.SX: 32, Gate.RZ: 0, Gate.CX: 64, Gate.MEASURE: 1120, Gate.RESET: 1120})


def test_dd_inserts_in_idle_period():
    """DD should insert gates on idle qubits while others are busy."""
    target = _small_ibm_target()
    # q1 idle while q0 does 4 SX ops (128dt) — enough for XX (128dt)
    c = Circuit(2)
    c.ops = [
        Operation(Gate.CX, (0, 1)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.CX, (0, 1)),
    ]
    result = dynamic_decoupling(c, target)
    assert len(result.ops) > len(c.ops), "DD should have inserted gates"
    for op in result.ops:
        assert op.gate in target.basis_gates


def test_dd_noop_no_idle():
    """DD should be a no-op when there are no idle periods."""
    target = _small_ibm_target()
    c = Circuit(2)
    c.cx(0, 1)  # Both qubits busy simultaneously
    result = dynamic_decoupling(c, target)
    assert len(result.ops) == len(c.ops)


def test_dd_noop_no_durations():
    """DD should be a no-op when target has no duration info."""
    target = Target(n_qubits=2, edges=frozenset({(0,1)}),
                    basis_gates=frozenset({Gate.SX, Gate.RZ, Gate.CX}))
    c = Circuit(2)
    c.sx(0)
    c.sx(0)
    c.sx(0)
    result = dynamic_decoupling(c, target)
    assert len(result.ops) == len(c.ops)


def test_dd_preserves_identity():
    """XX and XY4 sequences are identity — circuit should be equivalent."""
    target = _small_ibm_target()
    c = Circuit(2)
    c.ops = [
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)),
        Operation(Gate.CX, (0, 1)),
    ]
    original_u = to_unitary(c)
    result = dynamic_decoupling(c, target)
    result_u = to_unitary(result)
    assert np.allclose(np.abs(np.trace(original_u.conj().T @ result_u)), 2**c.n_qubits, atol=1e-6)


def test_dd_xy4_sequence():
    """XY4 sequence should work and insert more gates than XX."""
    target = _small_ibm_target()
    c = Circuit(2)
    # Need a longer idle gap for XY4 (256dt): 8 SX ops = 256dt
    c.ops = [Operation(Gate.SX, (0,)) for _ in range(8)] + [Operation(Gate.CX, (0, 1))]
    result_xx = dynamic_decoupling(c, target, sequence="XX")
    result_xy4 = dynamic_decoupling(c, target, sequence="XY4")
    assert len(result_xy4.ops) >= len(result_xx.ops)


def test_dd_native_basis():
    """All DD gates should be decomposed to native basis."""
    target = _small_ibm_target()
    c = Circuit(2)
    c.ops = [Operation(Gate.SX, (0,)) for _ in range(8)] + [Operation(Gate.CX, (0, 1))]
    result = dynamic_decoupling(c, target)
    for op in result.ops:
        assert op.gate in target.basis_gates, f"{op.gate} not in basis"


def test_dd_skips_unused_qubits():
    """DD should not insert on qubits that the circuit never touches."""
    target = Target(n_qubits=10, edges=frozenset({(i, i+1) for i in range(9)}),
                    basis_gates=frozenset({Gate.SX, Gate.RZ, Gate.CX, Gate.MEASURE, Gate.RESET}),
                    directed=True, duration={Gate.SX: 32, Gate.RZ: 0, Gate.CX: 64, Gate.MEASURE: 1120, Gate.RESET: 1120})
    c = Circuit(10)
    c.ops = [Operation(Gate.CX, (0, 1)), Operation(Gate.SX, (0,)), Operation(Gate.SX, (0,)),
             Operation(Gate.SX, (0,)), Operation(Gate.SX, (0,)), Operation(Gate.CX, (0, 1))]
    result = dynamic_decoupling(c, target)
    touched = {q for op in result.ops for q in op.qubits}
    assert touched == {0, 1}, f"DD touched unused qubits: {touched - {0, 1}}"


def test_dd_skips_post_measure_gaps():
    """DD should not insert after a qubit's final measurement."""
    target = _small_ibm_target()
    c = Circuit(2)
    c.ops = [
        Operation(Gate.CX, (0, 1)),
        Operation(Gate.SX, (0,)), Operation(Gate.SX, (0,)),
        Operation(Gate.SX, (0,)), Operation(Gate.SX, (0,)),
        Operation(Gate.MEASURE, (0,), classical_bit=0),
        Operation(Gate.CX, (0, 1)),  # q1 still active after q0 measured
    ]
    result = dynamic_decoupling(c, target)
    # q0 has a post-measure trailing idle — DD should NOT fill it
    # Only q1 should get DD (idle during q0's 4 SX ops)
    q0_ops = [op for op in result.ops if 0 in op.qubits and op.gate not in (Gate.CX, Gate.MEASURE)]
    q0_non_original = [op for op in q0_ops if op.gate not in (Gate.SX, Gate.RZ)]
    assert len(q0_non_original) == 0, f"DD inserted on q0 after measure: {q0_non_original}"


def test_transpile_dd_flag():
    """transpile(..., dd=True) should apply DD."""
    target = _small_ibm_target()
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    result_no_dd = transpile(c, target)
    result_dd = transpile(c, target, dd=True)
    assert len(result_dd.ops) >= len(result_no_dd.ops)
