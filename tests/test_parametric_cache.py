"""Tests for bind_params (in-place rebinding) and transpile cache."""
import numpy as np
from math import pi

from tinyqubit.ir import Circuit, Gate, Parameter
from tinyqubit.simulator import simulate
from tinyqubit.compile import transpile
from tinyqubit.target import Target


# ── bind_params ──

def test_bind_params_matches_bind():
    theta = Parameter("theta")
    c = Circuit(1)
    c.ry(0, theta)
    bound = c.bind({"theta": pi / 3})
    state_bind, _ = simulate(bound)

    c2 = Circuit(1)
    c2.ry(0, theta)
    c2.bind_params({"theta": pi / 3})
    state_bp, _ = simulate(c2)
    assert np.allclose(state_bind, state_bp)

def test_bind_params_returns_self():
    c = Circuit(1)
    c.ry(0, Parameter("x"))
    assert c.bind_params({"x": 1.0}) is c

def test_bind_params_rebindable():
    theta = Parameter("theta")
    c = Circuit(1)
    c.ry(0, theta)
    c.bind_params({"theta": 0.0})
    s0, _ = simulate(c)
    c.bind_params({"theta": pi})
    s1, _ = simulate(c)
    # RY(0)|0> = |0>, RY(pi)|0> = |1>
    assert abs(s0[0]) > 0.99
    assert abs(s1[1]) > 0.99

def test_bind_params_after_transpile(ionq_4):
    theta = Parameter("theta")
    c = Circuit(2)
    c.ry(0, theta).cx(0, 1)
    compiled = transpile(c, ionq_4)
    compiled.bind_params({"theta": pi / 2})
    state, _ = simulate(compiled)
    assert state.shape[0] == 2 ** compiled.n_qubits

def test_bind_params_shared_param():
    p = Parameter("p")
    c = Circuit(2)
    c.ry(0, p).ry(1, p)
    c.bind_params({"p": pi})
    s, _ = simulate(c)
    # Both qubits get RY(pi)|0> = |1>, so state = |11> = index 3
    assert abs(s[3]) > 0.99


# ── transpile cache ──

def test_transpile_cache_hit(ionq_4):
    theta = Parameter("theta")
    c1 = Circuit(2)
    c1.ry(0, theta).cx(0, 1)
    c2 = Circuit(2)
    c2.ry(0, theta).cx(0, 1)
    cache = {}
    r1 = transpile(c1, ionq_4, cache=cache)
    r2 = transpile(c2, ionq_4, cache=cache)
    assert r1 is r2

def test_transpile_cache_different_structure(ionq_4):
    theta = Parameter("theta")
    c1 = Circuit(2)
    c1.ry(0, theta).cx(0, 1)
    c2 = Circuit(2)
    c2.rx(0, theta).cx(0, 1)
    cache = {}
    r1 = transpile(c1, ionq_4, cache=cache)
    r2 = transpile(c2, ionq_4, cache=cache)
    assert r1 is not r2

def test_transpile_cache_different_topology():
    """Same basis gates + n_qubits but different edges must not share cache."""
    basis = frozenset({Gate.RX, Gate.RY, Gate.RZ, Gate.CX})
    t_line = Target(n_qubits=3, edges=frozenset({(0, 1), (1, 2)}), basis_gates=basis, name="t")
    t_ring = Target(n_qubits=3, edges=frozenset({(0, 1), (1, 2), (0, 2)}), basis_gates=basis, name="t")
    c = Circuit(3)
    c.h(0).cx(0, 1).cx(0, 2)
    cache = {}
    r1 = transpile(c, t_line, cache=cache)
    r2 = transpile(c, t_ring, cache=cache)
    assert r1 is not r2
