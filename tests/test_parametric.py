"""Tests for parametric (symbolic) circuits."""
import pytest
import numpy as np
from math import pi

from tinyqubit.ir import Circuit, Gate, Operation, Parameter, _has_parameter
from tinyqubit.simulator import simulate
from tinyqubit.passes.fuse import fuse_1q_gates
from tinyqubit.passes.optimize import optimize
from tinyqubit.passes.decompose import decompose
from tinyqubit.export.qasm import to_openqasm2, to_openqasm3


# ── Parameter class ──

def test_parameter_repr():
    p = Parameter("theta")
    assert repr(p) == "Parameter('theta')"

def test_parameter_equality():
    a, b = Parameter("x"), Parameter("x")
    assert a == b
    assert a != Parameter("y")

def test_parameter_hash():
    a, b = Parameter("x"), Parameter("x")
    assert hash(a) == hash(b)
    assert {a, b} == {Parameter("x")}

def test_parameter_not_equal_to_string():
    assert Parameter("x") != "x"

def test_has_parameter():
    assert _has_parameter((Parameter("x"),))
    assert _has_parameter((1.0, Parameter("x")))
    assert not _has_parameter((1.0, 2.0))
    assert not _has_parameter(())


# ── Circuit builder ──

def test_rx_accepts_parameter():
    p = Parameter("theta")
    c = Circuit(1)
    c.rx(0, p)
    assert c.ops[0].params == (p,)

def test_ry_accepts_parameter():
    p = Parameter("theta")
    c = Circuit(1)
    c.ry(0, p)
    assert c.ops[0].params == (p,)

def test_rz_accepts_parameter():
    p = Parameter("theta")
    c = Circuit(1)
    c.rz(0, p)
    assert c.ops[0].params == (p,)

def test_cp_accepts_parameter():
    p = Parameter("phi")
    c = Circuit(2)
    c.cp(0, 1, p)
    assert c.ops[0].params == (p,)


# ── is_parameterized / parameters ──

def test_is_parameterized_true():
    c = Circuit(1)
    c.ry(0, Parameter("a"))
    assert c.is_parameterized

def test_is_parameterized_false():
    c = Circuit(1)
    c.ry(0, 1.0)
    assert not c.is_parameterized

def test_parameters_property():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(2)
    c.ry(0, a).rz(1, b).ry(0, a)
    assert c.parameters == {a, b}

def test_parameters_empty():
    c = Circuit(1)
    c.h(0)
    assert c.parameters == set()


# ── bind ──

def test_bind_full():
    p = Parameter("theta")
    c = Circuit(1)
    c.ry(0, p)
    bound = c.bind({"theta": pi / 2})
    assert not bound.is_parameterized
    assert bound.ops[0].params == (pi / 2,)

def test_bind_partial():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(2)
    c.ry(0, a).rz(1, b)
    bound = c.bind({"a": 1.0})
    assert bound.is_parameterized
    assert bound.ops[0].params == (1.0,)
    assert bound.ops[1].params == (b,)

def test_bind_returns_new_circuit():
    p = Parameter("x")
    c = Circuit(1)
    c.ry(0, p)
    bound = c.bind({"x": 0.5})
    assert c.is_parameterized  # original unchanged
    assert not bound.is_parameterized

def test_bind_preserves_non_parametric():
    c = Circuit(2)
    c.h(0).cx(0, 1).ry(0, Parameter("p"))
    bound = c.bind({"p": 1.0})
    assert bound.ops[0].gate == Gate.H
    assert bound.ops[1].gate == Gate.CX

def test_bind_preserves_condition():
    p = Parameter("theta")
    c = Circuit(1, 1)
    c._current_condition = (0, 1)
    c.ry(0, p)
    c._current_condition = None
    bound = c.bind({"theta": 0.5})
    assert bound.ops[0].condition == (0, 1)


# ── Simulator ──

def test_simulate_raises_on_unbound():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    with pytest.raises(TypeError, match="unbound Parameter"):
        simulate(c)

def test_simulate_works_after_bind():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    bound = c.bind({"theta": pi})
    state, _ = simulate(bound)
    # RY(pi)|0> = |1>
    assert abs(state[1]) > 0.99


# ── Passes ──

def test_fuse_skips_parametric():
    p = Parameter("t")
    c = Circuit(1)
    c.h(0).ry(0, p).h(0)  # Would normally fuse to single rotation
    fused = fuse_1q_gates(c)
    # Should preserve all 3 gates (no matrix fusion)
    assert len(fused.ops) == 3
    assert fused.ops[1].params == (p,)

def test_optimize_skips_parametric_merge():
    p = Parameter("a")
    c = Circuit(1)
    c.rz(0, p).rz(0, 0.5)
    opt = optimize(c)
    # Should NOT merge because first is parametric
    assert len(opt.ops) == 2

def test_decompose_skips_cp_parametric():
    p = Parameter("phi")
    basis = frozenset({Gate.RZ, Gate.RX, Gate.CX})
    c = Circuit(2)
    c.cp(0, 1, p)
    dec = decompose(c, basis)
    # CP with param should be left as-is (not decomposed)
    assert any(op.gate == Gate.CP for op in dec.ops)

def test_decompose_passes_ry_parametric():
    """RY decomposition just passes theta through, so it should work with Parameters."""
    p = Parameter("theta")
    basis = frozenset({Gate.RZ, Gate.RX, Gate.CX})
    c = Circuit(1)
    c.ry(0, p)
    dec = decompose(c, basis)
    # Should be decomposed: RY(p) -> RX(pi/2) RZ(p) RX(-pi/2)
    assert not any(op.gate == Gate.RY for op in dec.ops)
    # The parameter should appear in a RZ
    rz_ops = [op for op in dec.ops if op.gate == Gate.RZ]
    assert any(op.params == (p,) for op in rz_ops)


# ── QASM export ──

def test_qasm3_with_parameters():
    a, b = Parameter("alpha"), Parameter("beta")
    c = Circuit(2)
    c.ry(0, a).rz(1, b)
    qasm = to_openqasm3(c)
    assert "input float alpha;" in qasm
    assert "input float beta;" in qasm
    assert "alpha" in qasm
    assert "beta" in qasm

def test_qasm2_raises_on_parametric():
    c = Circuit(1)
    c.ry(0, Parameter("x"))
    with pytest.raises(ValueError, match="parameterized"):
        to_openqasm2(c)

def test_qasm2_ok_after_bind():
    c = Circuit(1)
    c.ry(0, Parameter("x"))
    bound = c.bind({"x": 1.0})
    qasm = to_openqasm2(bound)
    assert "ry" in qasm


# ── End-to-end VQE workflow ──

def test_vqe_workflow():
    """Build parameterized circuit, bind different values, simulate each."""
    theta = Parameter("theta")
    c = Circuit(1)
    c.ry(0, theta)

    results = []
    for val in [0.0, pi / 2, pi]:
        bound = c.bind({"theta": val})
        state, _ = simulate(bound)
        results.append(state)

    # RY(0)|0> = |0>
    assert abs(results[0][0]) > 0.99
    # RY(pi/2)|0> = (|0> + |1>)/sqrt(2)
    assert abs(abs(results[1][0]) - abs(results[1][1])) < 0.01
    # RY(pi)|0> = |1>
    assert abs(results[2][1]) > 0.99
