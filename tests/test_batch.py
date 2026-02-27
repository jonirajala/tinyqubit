"""Tests for batch simulation and parameter broadcasting."""
import numpy as np
from tinyqubit import (
    Circuit, Parameter, simulate, simulate_batch,
    expectation, expectation_batch, expectation_sweep, gradient_landscape,
)
from tinyqubit.observable import Z, X


def test_simulate_batch():
    circuits = []
    for gate_fn in [lambda c: c.x(0), lambda c: c.h(0), lambda c: c.y(0)]:
        c = Circuit(1)
        gate_fn(c)
        circuits.append(c)
    batch = simulate_batch(circuits)
    for i, c in enumerate(circuits):
        state, bits = simulate(c)
        np.testing.assert_allclose(batch[i][0], state)
        assert batch[i][1] == bits


def test_expectation_batch():
    obs = Z(0)
    circuits = [Circuit(1), Circuit(1), Circuit(1)]
    circuits[1].x(0)
    circuits[2].h(0)
    results = expectation_batch(circuits, obs)
    expected = np.array([expectation(c, obs) for c in circuits])
    np.testing.assert_allclose(results, expected)


def test_expectation_sweep_cosine():
    theta = Parameter("theta")
    c = Circuit(1)
    c.ry(0, theta)
    obs = Z(0)
    values = np.linspace(0, 2 * np.pi, 20)
    results = expectation_sweep(c, "theta", values, obs)
    np.testing.assert_allclose(results, np.cos(values), atol=1e-10)


def test_expectation_sweep_with_base_values():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(1)
    c.ry(0, a)
    c.rz(0, b)
    obs = Z(0)
    base = {"b": 0.0}
    values = np.linspace(0, 2 * np.pi, 10)
    results = expectation_sweep(c, "a", values, obs, base_values=base)
    # RZ doesn't affect Z expectation, so still cos(a)
    np.testing.assert_allclose(results, np.cos(values), atol=1e-10)


def test_gradient_landscape_shape():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(1)
    c.ry(0, a)
    c.rz(0, b)
    obs = Z(0)
    n = 10
    result = gradient_landscape(c, ["a", "b"], obs, {"a": 0.0, "b": 0.0}, n_points=n)
    assert result.shape == (n, n)


def test_gradient_landscape_known_values():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(1)
    c.ry(0, a)
    c.rz(0, b)
    obs = Z(0)
    n = 5
    ranges = [(0, np.pi), (0, np.pi)]
    result = gradient_landscape(c, ["a", "b"], obs, {"a": 0.0, "b": 0.0},
                                n_points=n, ranges=ranges)
    # Corner (0,0): RY(0)RZ(0)|0⟩ = |0⟩, ⟨Z⟩ = 1
    assert abs(result[0, 0] - 1.0) < 1e-10
    # Corner (n-1, 0): RY(π)RZ(0)|0⟩ = |1⟩, ⟨Z⟩ = -1
    assert abs(result[-1, 0] - (-1.0)) < 1e-10
    # Verify against direct expectation
    bound = c.bind({"a": np.pi, "b": np.pi})
    assert abs(result[-1, -1] - expectation(bound, obs)) < 1e-10
