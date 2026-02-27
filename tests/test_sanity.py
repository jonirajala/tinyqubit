import numpy as np
from tinyqubit import Circuit, simulate, probabilities

def test_norm_after_simulate():
    c = Circuit(3)
    c.h(0); c.cx(0, 1); c.rz(2, 1.23); c.cx(1, 2); c.h(2)
    state, _ = simulate(c)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10

def test_probs_sum():
    c = Circuit(3)
    c.h(0); c.cx(0, 1); c.rz(2, 0.7)
    assert abs(probabilities(c).sum() - 1.0) < 1e-10
    assert abs(probabilities(c, wires=[0, 2]).sum() - 1.0) < 1e-10
    assert abs(probabilities(c, wires=[1]).sum() - 1.0) < 1e-10
