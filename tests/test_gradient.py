"""Tests for parameter-shift and finite-difference gradients."""
import numpy as np
import pytest
from tinyqubit import (
    Circuit, Parameter, Z, Observable,
    parameter_shift_gradient, finite_difference_gradient,
)


# --- Analytical match: RY(θ)|0⟩, ⟨Z⟩ = cos(θ), gradient = -sin(θ) ---

@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, np.pi / 2, np.pi, 2.5])
def test_ry_z_gradient(theta):
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    grad = parameter_shift_gradient(c, Z(0), {"theta": theta})
    assert abs(grad["theta"] - (-np.sin(theta))) < 1e-10


# --- Multi-parameter: RY(θ) RZ(φ)|0⟩ ---

def test_multi_parameter():
    theta, phi = Parameter("theta"), Parameter("phi")
    c = Circuit(1)
    c.ry(0, theta)
    c.rz(0, phi)
    vals = {"theta": 0.7, "phi": 1.2}
    grad = parameter_shift_gradient(c, Z(0), vals)
    # ⟨Z⟩ = cos(θ) regardless of RZ, so d/dθ = -sin(θ), d/dφ = 0
    assert abs(grad["theta"] - (-np.sin(0.7))) < 1e-10
    assert abs(grad["phi"]) < 1e-10


# --- Shot-based gradient ---

def test_shot_based():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    theta = np.pi / 3
    grad = parameter_shift_gradient(c, Z(0), {"theta": theta}, shots=50000)
    expected = -np.sin(theta)
    assert abs(grad["theta"] - expected) < 0.1


# --- Finite difference matches parameter-shift ---

def test_finite_difference():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    vals = {"theta": 1.2}
    ps = parameter_shift_gradient(c, Z(0), vals)
    fd = finite_difference_gradient(c, Z(0), vals)
    assert abs(ps["theta"] - fd["theta"]) < 1e-5


# --- Bell state: H·CX with parametric RY, observable Z(0)@Z(1) ---

def test_bell_state_gradient():
    theta = Parameter("theta")
    c = Circuit(2)
    c.ry(0, theta)
    c.cx(0, 1)
    obs = Z(0) @ Z(1)
    grad = parameter_shift_gradient(c, obs, {"theta": 0.8})
    fd = finite_difference_gradient(c, obs, {"theta": 0.8})
    assert abs(grad["theta"] - fd["theta"]) < 1e-5


# --- Identity observable: gradient of constant = 0 ---

def test_identity_gradient():
    c = Circuit(1)
    c.ry(0, Parameter("theta"))
    identity = Observable([(1.0, {})])  # I
    grad = parameter_shift_gradient(c, identity, {"theta": 1.0})
    assert abs(grad["theta"]) < 1e-10


# --- RX gate gradient ---

def test_rx_gradient():
    c = Circuit(1)
    c.rx(0, Parameter("theta"))
    # ⟨Z⟩ = cos(θ) for RX(θ)|0⟩, gradient = -sin(θ)
    theta = 0.9
    grad = parameter_shift_gradient(c, Z(0), {"theta": theta})
    assert abs(grad["theta"] - (-np.sin(theta))) < 1e-10
