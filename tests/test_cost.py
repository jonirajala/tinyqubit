import numpy as np
import tinyqubit as tq
from tinyqubit.cost import cross_entropy_cost, mse_cost, fidelity_cost


def _make_circuit(theta_name="θ"):
    c = tq.Circuit(1)
    c.ry(0, tq.Parameter(theta_name))
    return c


def test_mse_perfect():
    # RY(π)|0⟩ = |1⟩ → ⟨Z⟩ = -1; y = -1 → MSE ≈ 0
    c = _make_circuit()
    X = np.array([[np.pi]])
    y = np.array([-1.0])
    assert mse_cost(c, X, y) < 1e-10


def test_mse_value():
    # RY(0)|0⟩ = |0⟩ → ⟨Z⟩ = +1; y = -1 → MSE = (1-(-1))² = 4
    c = _make_circuit()
    X = np.array([[0.0]])
    y = np.array([-1.0])
    assert abs(mse_cost(c, X, y) - 4.0) < 1e-10


def test_cross_entropy_bounds():
    c = _make_circuit()
    X = np.array([[0.0], [np.pi / 2], [np.pi]])
    y = np.array([0, 1, 0])
    loss = cross_entropy_cost(c, X, y)
    assert loss >= 0


def test_cross_entropy_confident():
    # RY(0)|0⟩ → ⟨Z⟩=+1 → p=1 for y=1; RY(π)|0⟩ → ⟨Z⟩=-1 → p=0 for y=0
    c = _make_circuit()
    X = np.array([[0.0], [np.pi]])
    y = np.array([1, 0])
    loss = cross_entropy_cost(c, X, y)
    assert loss < 0.01


def test_fidelity_cost_zero():
    # X|0⟩ = |1⟩, target = |1⟩ → cost ≈ 0
    c = tq.Circuit(1)
    c.x(0)
    target = np.array([0.0, 1.0])
    assert fidelity_cost(c, target) < 1e-10


def test_fidelity_cost_orthogonal():
    # |0⟩ vs |1⟩ → cost ≈ 1
    c = tq.Circuit(1)  # identity → |0⟩
    target = np.array([0.0, 1.0])
    assert abs(fidelity_cost(c, target) - 1.0) < 1e-10


def test_feature_binding_order():
    # 2 features: a, b sorted → a maps to X[:,0], b maps to X[:,1]
    c = tq.Circuit(1)
    c.ry(0, tq.Parameter("b"))
    c.rz(0, tq.Parameter("a"))
    X = np.array([[0.5, np.pi]])  # a=0.5 (→RZ), b=π (→RY)
    y = np.array([-1.0])
    # With sorted names: a→X[:,0]=0.5, b→X[:,1]=π
    # So RY(π) then RZ(0.5): ⟨Z⟩ ≈ -1 → MSE(y=-1) ≈ 0
    loss = mse_cost(c, X, y)
    assert loss < 0.1
