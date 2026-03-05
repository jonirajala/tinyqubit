"""Tests for FTQC resource estimation."""
from tinyqubit import Circuit, resource_estimate, ResourceEstimate
import pytest


def test_bell_state_no_t():
    c = Circuit(2)
    c.h(0).cx(0, 1)
    r = resource_estimate(c)
    assert r.t_count == 0
    assert r.t_depth == 0
    assert r.clifford_count == 2


def test_t_count():
    c = Circuit(2)
    c.t(0).t(1).tdg(0)
    r = resource_estimate(c)
    assert r.t_count == 3


def test_t_depth():
    # Serial: T on same qubit → each in its own layer
    c1 = Circuit(1)
    c1.t(0).t(0).t(0)
    r1 = resource_estimate(c1)
    assert r1.t_depth == 3

    # Parallel: T on different qubits → single layer
    c2 = Circuit(3)
    c2.t(0).t(1).t(2)
    r2 = resource_estimate(c2)
    assert r2.t_depth == 1


def test_surface_code_distance():
    c = Circuit(1)
    c.h(0)
    # Default error_rate=1e-3, p_logical=1e-10 → d = ceil(log(1e-10/0.1)/log(10*1e-3)) = ceil(-9/log(0.01)) = ceil(-9/-2) = ceil(4.5) = 5
    r = resource_estimate(c)
    assert r.code_distance == 5


def test_physical_qubits_scale():
    c1 = Circuit(2)
    c1.h(0).cx(0, 1)
    r1 = resource_estimate(c1)

    c2 = Circuit(4)
    c2.h(0).cx(0, 1).t(2).t(3)
    r2 = resource_estimate(c2)

    assert r2.physical_qubits > r1.physical_qubits


def test_unknown_code_raises():
    c = Circuit(1)
    c.h(0)
    with pytest.raises(ValueError, match="Unknown code"):
        resource_estimate(c, code="steane")
