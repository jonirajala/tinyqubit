"""Tests for MPS simulator."""
import numpy as np
import pytest
from tinyqubit import Circuit, simulate, simulate_mps, mps_to_statevector, states_equal, Gate, sample, expectation, expectation_z, Z
from tinyqubit.simulator.mps import MPSState, mps_probabilities


def _mps_sv(c, **kw):
    tensors, cb = simulate_mps(c, **kw)
    return mps_to_statevector(tensors), cb


def _check(circuit, tol=1e-10):
    sv_mps, _ = _mps_sv(circuit)
    sv_ref, _ = simulate(circuit)
    assert states_equal(sv_mps, sv_ref, tol), f"MPS != SV, fidelity={abs(np.vdot(sv_mps, sv_ref))}"


# --- Statevector agreement ---

@pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s", "t", "sdg", "tdg", "sx"])
def test_1q_gates(gate):
    c = Circuit(1)
    getattr(c, gate)(0)
    _check(c)


@pytest.mark.parametrize("gate,param", [("rx", 0.7), ("ry", 1.3), ("rz", 2.1)])
def test_1q_param_gates(gate, param):
    c = Circuit(1)
    getattr(c, gate)(0, param)
    _check(c)


def test_bell_state():
    _check(Circuit(2).h(0).cx(0, 1))


def test_ghz_3():
    _check(Circuit(3).h(0).cx(0, 1).cx(1, 2))


def test_cz():
    _check(Circuit(2).h(0).h(1).cz(0, 1))


def test_swap():
    _check(Circuit(2).x(0).swap(0, 1))


def test_cp():
    _check(Circuit(2).h(0).h(1).cp(0, 1, 1.5))


def test_ecr():
    _check(Circuit(2).h(0).ecr(0, 1))


def test_rzz():
    _check(Circuit(2).h(0).h(1).rzz(0, 1, 0.8))


def test_ccx():
    _check(Circuit(3).x(0).x(1).ccx(0, 1, 2))


def test_ccz():
    _check(Circuit(3).x(0).x(1).x(2).ccz(0, 1, 2))


# --- Non-adjacent gates ---

def test_cx_non_adjacent():
    _check(Circuit(4).h(0).cx(0, 3))


def test_cx_reversed_non_adjacent():
    _check(Circuit(4).h(3).cx(3, 0))


def test_ccx_non_adjacent():
    c = Circuit(5).x(0).x(4).ccx(0, 4, 2)
    _check(c)


# --- Bond dimension cap ---

def test_bond_dim_cap():
    c = Circuit(6).h(0)
    for i in range(5):
        c.cx(i, i + 1)
    tensors, _ = simulate_mps(c, max_bond_dim=4)
    for t in tensors:
        assert t.shape[0] <= 4 and t.shape[2] <= 4, f"bond dim exceeded: {t.shape}"


# --- Measurement ---

def test_measure_deterministic():
    c = Circuit(1).x(0).measure(0)
    for _ in range(5):
        _, cb = simulate_mps(c, seed=42)
        assert cb[0] == 1


def test_bell_measurement_correlated():
    results = []
    for seed in range(20):
        c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
        _, cb = simulate_mps(c, seed=seed)
        results.append((cb[0], cb[1]))
    assert all(a == b for a, b in results), "Bell pair measurements should be correlated"


# --- Reset ---

def test_reset():
    c = Circuit(1).x(0).reset(0).measure(0)
    for seed in range(5):
        _, cb = simulate_mps(c, seed=seed)
        assert cb[0] == 0


# --- Teleportation ---

def test_teleportation():
    for seed in range(10):
        c = Circuit(3, 3)
        c.x(0)              # state to teleport: |1>
        c.h(1).cx(1, 2)     # Bell pair on q1,q2
        c.cx(0, 1).h(0)     # Bell measurement on q0,q1
        c.measure(0, 0).measure(1, 1)
        with c.c_if(1, 1): c.x(2)
        with c.c_if(0, 1): c.z(2)
        c.measure(2, 2)
        _, cb = simulate_mps(c, seed=seed)
        assert cb[2] == 1, f"Teleportation failed with seed={seed}"


# --- Large circuits ---

def test_50q_product_state():
    c = Circuit(50)
    for i in range(50):
        c.x(i)
    tensors, _ = simulate_mps(c)
    # Product state: all bond dims should be 1
    for t in tensors:
        assert t.shape[0] == 1 and t.shape[2] == 1


def test_50q_ghz_measurement():
    c = Circuit(50)
    c.h(0)
    for i in range(49):
        c.cx(i, i + 1)
    for i in range(50):
        c.measure(i)
    _, cb = simulate_mps(c, seed=42)
    vals = set(cb.values())
    assert len(vals) == 1, f"GHZ measurements should all agree, got {cb}"


# --- Auto-dispatch ---

def test_auto_dispatch_large():
    c = Circuit(30)
    c.x(0).ry(1, 0.5)  # ry makes it non-Clifford, forcing MPS path
    state, cb = simulate(c)
    assert isinstance(state, MPSState)


# --- Combined circuit ---

def test_multi_gate_circuit():
    c = Circuit(4)
    c.h(0).cx(0, 1).rz(2, 0.5).ry(3, 1.2)
    c.cx(2, 3).cz(0, 2)
    _check(c)


# --- MPS dispatch paths ---

def test_mps_sample_dispatch():
    c = Circuit(3).x(0).h(1).cx(1, 2)
    tensors, _ = simulate_mps(c)
    counts = sample(MPSState(tensors, 3), 100, seed=42)
    assert all(k[0] == '1' for k in counts)

def test_mps_expectation_dispatch():
    c = Circuit(3).x(0)
    tensors, _ = simulate_mps(c)
    val = expectation(MPSState(tensors, 3), Z(0))
    assert abs(val - (-1.0)) < 1e-10

def test_mps_expectation_vs_statevector():
    c = Circuit(3).h(0).cx(0, 1)
    tensors, _ = simulate_mps(c)
    sv_ref, _ = simulate(c)
    assert abs(expectation(MPSState(tensors, 3), Z(0)) - expectation(sv_ref, Z(0), n_qubits=3)) < 1e-10

def test_mps_expectation_z_dispatch():
    c = Circuit(2).x(0)
    tensors, _ = simulate_mps(c)
    zvals = expectation_z(MPSState(tensors, 2))
    assert abs(zvals[0] - (-1.0)) < 1e-10
    assert abs(zvals[1] - 1.0) < 1e-10

def test_mps_probabilities():
    c = Circuit(3).x(0)
    tensors, _ = simulate_mps(c)
    probs = mps_probabilities(tensors)
    assert abs(probs[4] - 1.0) < 1e-10  # |100> = index 4

def test_mps_probabilities_marginal():
    c = Circuit(3).h(0).cx(0, 1)
    tensors, _ = simulate_mps(c)
    probs = mps_probabilities(tensors, wires=[0])
    assert abs(probs[0] - 0.5) < 1e-6
    assert abs(probs[1] - 0.5) < 1e-6
