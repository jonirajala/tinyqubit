"""Tests for trainability diagnostics."""
import numpy as np
from tinyqubit import Circuit, Parameter, Z
from tinyqubit.qml import gradient_variance, expressibility


# --- gradient_variance ---

def test_gradient_variance_keys():
    c = Circuit(1)
    c.ry(0, Parameter("a"))
    c.rz(0, Parameter("b"))
    var = gradient_variance(c, Z(0), n_samples=10, seed=42)
    assert set(var.keys()) == {"a", "b"}


def test_gradient_variance_positive():
    c = Circuit(1)
    c.ry(0, Parameter("a"))
    c.rz(0, Parameter("b"))
    var = gradient_variance(c, Z(0), n_samples=20, seed=42)
    assert all(v >= 0 for v in var.values())


def test_gradient_variance_single_ry():
    """1-qubit RY has non-trivial gradient variance (not a barren plateau)."""
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    var = gradient_variance(c, Z(0), n_samples=50, seed=42)
    assert var["theta"] > 0.01


def test_gradient_variance_deterministic():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    v1 = gradient_variance(c, Z(0), n_samples=20, seed=123)
    v2 = gradient_variance(c, Z(0), n_samples=20, seed=123)
    assert v1 == v2


# --- expressibility ---

def test_expressibility_positive():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    kl = expressibility(c, n_samples=200, seed=42)
    assert kl >= 0


def test_expressibility_deep_more_expressive():
    """Deeper circuit should be more expressive (lower KL) than shallow."""
    shallow = Circuit(2)
    shallow.ry(0, Parameter("a"))
    shallow.ry(1, Parameter("b"))

    deep = Circuit(2)
    deep.ry(0, Parameter("a"))
    deep.ry(1, Parameter("b"))
    deep.cx(0, 1)
    deep.ry(0, Parameter("c"))
    deep.ry(1, Parameter("d"))
    deep.cx(0, 1)
    deep.ry(0, Parameter("e"))
    deep.ry(1, Parameter("f"))

    kl_shallow = expressibility(shallow, n_samples=500, seed=42)
    kl_deep = expressibility(deep, n_samples=500, seed=42)
    assert kl_deep < kl_shallow


def test_expressibility_deterministic():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    kl1 = expressibility(c, n_samples=100, seed=99)
    kl2 = expressibility(c, n_samples=100, seed=99)
    assert kl1 == kl2
