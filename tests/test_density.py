import numpy as np
from tinyqubit import Circuit, Gate, NoiseModel, simulate, simulate_density


def test_pure_state_agreement():
    """simulate_density without noise matches |ψ⟩⟨ψ| from statevector sim."""
    c = Circuit(2)
    c.h(0).cx(0, 1)
    sv, _ = simulate(c)
    rho, _ = simulate_density(c)
    np.testing.assert_allclose(rho, np.outer(sv, sv.conj()), atol=1e-12)


def test_pure_state_3q():
    """3-qubit GHZ state agreement."""
    c = Circuit(3)
    c.h(0).cx(0, 1).cx(0, 2)
    sv, _ = simulate(c)
    rho, _ = simulate_density(c)
    np.testing.assert_allclose(rho, np.outer(sv, sv.conj()), atol=1e-12)


def test_noise_determinism():
    """Density matrix noise is exact — independent of seed."""
    c = Circuit(2)
    c.h(0).cx(0, 1)
    nm = NoiseModel().add_depolarizing(0.05)
    rho1, _ = simulate_density(c, noise_model=nm, seed=42)
    rho2, _ = simulate_density(c, noise_model=nm, seed=123)
    np.testing.assert_allclose(rho1, rho2, atol=1e-12)


def test_exact_depolarizing():
    """X + depolarizing(p) gives analytically known result."""
    p = 0.1
    c = Circuit(1)
    c.x(0)
    nm = NoiseModel().add_depolarizing(p, [Gate.X])
    rho, _ = simulate_density(c, noise_model=nm)
    # ρ = (1-2p/3)|1⟩⟨1| + (2p/3)|0⟩⟨0|
    expected = np.array([[2*p/3, 0], [0, 1 - 2*p/3]], dtype=complex)
    np.testing.assert_allclose(rho, expected, atol=1e-12)


def test_trace_preservation():
    c = Circuit(3)
    c.h(0).cx(0, 1).cx(1, 2)
    nm = NoiseModel().add_depolarizing(0.05).add_amplitude_damping(0.02).add_phase_damping(0.03)
    rho, _ = simulate_density(c, noise_model=nm)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10)


def test_positive_semidefinite():
    c = Circuit(2)
    c.h(0).cx(0, 1)
    nm = NoiseModel().add_depolarizing(0.1).add_phase_damping(0.05)
    rho, _ = simulate_density(c, noise_model=nm)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-10)


def test_trajectory_agreement():
    """Average of many trajectories approximates density matrix."""
    c = Circuit(2)
    c.h(0).cx(0, 1)
    nm = NoiseModel().add_depolarizing(0.1)
    rho_exact, _ = simulate_density(c, noise_model=nm)
    rho_avg = np.zeros_like(rho_exact)
    for i in range(5000):
        sv, _ = simulate(c, seed=i, noise_model=nm)
        rho_avg += np.outer(sv, sv.conj())
    rho_avg /= 5000
    np.testing.assert_allclose(rho_avg, rho_exact, atol=0.05)


def test_measurement_deterministic():
    c = Circuit(1)
    c.x(0).measure(0, 0)
    rho, bits = simulate_density(c)
    assert bits[0] == 1
    np.testing.assert_allclose(rho, np.array([[0, 0], [0, 1]], dtype=complex), atol=1e-12)


def test_bell_measurement():
    results = {}
    for seed in range(100):
        c = Circuit(2, 2)
        c.h(0).cx(0, 1).measure(0, 0).measure(1, 1)
        _, bits = simulate_density(c, seed=seed)
        key = (bits[0], bits[1])
        results[key] = results.get(key, 0) + 1
    # Only correlated outcomes
    assert set(results.keys()) <= {(0, 0), (1, 1)}
    assert sum(results.values()) == 100


def test_teleportation():
    c = Circuit(3, 2)
    c.x(0)  # prepare |1⟩
    c.h(1).cx(1, 2)  # Bell pair on 1,2
    c.cx(0, 1).h(0)
    c.measure(0, 0).measure(1, 1)
    with c.c_if(1, 1): c.x(2)
    with c.c_if(0, 1): c.z(2)
    rho, _ = simulate_density(c, seed=42)
    # Partial trace over qubits 0,1 → reduced dm for qubit 2
    rho_t = rho.reshape([2] * 6)
    rho_t = np.trace(rho_t, axis1=0, axis2=3)  # trace out q0
    rho_2 = np.trace(rho_t, axis1=0, axis2=2)  # trace out q1
    np.testing.assert_allclose(rho_2, np.array([[0, 0], [0, 1]], dtype=complex), atol=1e-10)


def test_amplitude_damping_exact():
    """Amplitude damping of |1⟩: p(0) = gamma."""
    gamma = 0.3
    c = Circuit(1)
    c.x(0)
    nm = NoiseModel().add_amplitude_damping(gamma, [Gate.X])
    rho, _ = simulate_density(c, noise_model=nm)
    np.testing.assert_allclose(rho[0, 0], gamma, atol=1e-12)
    np.testing.assert_allclose(rho[1, 1], 1 - gamma, atol=1e-12)
