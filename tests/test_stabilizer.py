"""Tests for stabilizer (tableau) simulator."""
import numpy as np
from math import sqrt

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.simulator import simulate, states_equal
from tinyqubit.simulator.stabilizer import is_clifford, simulate_stabilizer, clifford_t_info, simulate_clifford_t


def _force_statevector(circuit, seed=None):
    """Bypass Clifford detection by setting _initial_state to |0...0>."""
    c2 = Circuit(circuit.n_qubits, circuit.n_classical)
    c2.ops = list(circuit.ops)
    c2._initial_state = np.zeros(2**circuit.n_qubits, dtype=complex)
    c2._initial_state[0] = 1.0
    return simulate(c2, seed=seed)


# -- Clifford detection --

def test_clifford_detection_hcx():
    c = Circuit(2).h(0).cx(0, 1)
    assert is_clifford(c)

def test_clifford_detection_t_gate():
    c = Circuit(1).t(0)
    assert not is_clifford(c)

def test_clifford_detection_rx():
    c = Circuit(1).rx(0, 0.5)
    assert not is_clifford(c)

def test_clifford_detection_all_cliffords():
    c = Circuit(3, 3)
    c.x(0).y(1).z(2).h(0).s(1).sdg(2).sx(0)
    c.cx(0, 1).cz(1, 2).swap(0, 2)
    c.measure(0, 0).reset(1)
    assert is_clifford(c)


# -- Agreement with statevector sim --

_GATE_CIRCUITS = [
    ("H", lambda: Circuit(1).h(0)),
    ("X", lambda: Circuit(1).x(0)),
    ("Y", lambda: Circuit(1).y(0)),
    ("Z", lambda: Circuit(1).z(0)),
    ("S", lambda: Circuit(1).h(0).s(0)),
    ("SDG", lambda: Circuit(1).h(0).sdg(0)),
    ("SX", lambda: Circuit(1).sx(0)),
    ("Bell", lambda: Circuit(2).h(0).cx(0, 1)),
    ("CZ", lambda: Circuit(2).h(0).h(1).cz(0, 1)),
    ("SWAP", lambda: Circuit(2).x(0).swap(0, 1)),
]

def test_gate_agreement():
    for name, make_circuit in _GATE_CIRCUITS:
        c = make_circuit()
        sv_stab, _ = simulate_stabilizer(c, seed=42)
        sv_ref, _ = _force_statevector(c, seed=42)
        assert states_equal(sv_stab, sv_ref), f"{name} gate disagreed"

def test_random_clifford_agreement():
    """Random 4-qubit Clifford circuit."""
    rng = np.random.default_rng(123)
    c = Circuit(4)
    gates_1q = [Gate.H, Gate.S, Gate.SDG, Gate.X, Gate.Y, Gate.Z, Gate.SX]
    gates_2q = [Gate.CX, Gate.CZ, Gate.SWAP]
    for _ in range(30):
        if rng.random() < 0.6:
            g = gates_1q[rng.integers(len(gates_1q))]
            q = int(rng.integers(4))
            c.ops.append(Operation(g, (q,)))
        else:
            g = gates_2q[rng.integers(len(gates_2q))]
            q0, q1 = rng.choice(4, size=2, replace=False)
            q0, q1 = int(q0), int(q1)
            if g in (Gate.CZ, Gate.SWAP):
                q0, q1 = min(q0, q1), max(q0, q1)
            c.ops.append(Operation(g, (q0, q1)))
    sv_stab, _ = simulate_stabilizer(c, seed=42)
    sv_ref, _ = _force_statevector(c, seed=42)
    assert states_equal(sv_stab, sv_ref)


# -- Measurement tests --

def test_bell_measurement_correlation():
    """Both qubits of a Bell pair must agree."""
    for seed in range(20):
        c = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
        _, bits = simulate(c, seed=seed)
        assert bits[0] == bits[1], f"Bell pair disagreed with seed={seed}"

def test_deterministic_measure_one():
    """|1> always measures 1."""
    c = Circuit(1, 1).x(0).measure(0, 0)
    for seed in range(10):
        _, bits = simulate(c, seed=seed)
        assert bits[0] == 1

def test_deterministic_measure_zero():
    """|0> always measures 0."""
    c = Circuit(1, 1).measure(0, 0)
    for seed in range(10):
        _, bits = simulate(c, seed=seed)
        assert bits[0] == 0

def test_seed_determinism():
    c = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    results = [simulate(c, seed=99)[1] for _ in range(5)]
    assert all(r == results[0] for r in results)


# -- Teleportation --

def test_teleportation():
    """Quantum teleportation: transfer |1> from q0 to q2 via Bell pair."""
    c = Circuit(3, 3)
    c.x(0)                       # State to teleport: |1>
    c.h(1).cx(1, 2)              # Bell pair on q1,q2
    c.cx(0, 1).h(0)              # Bell measurement on q0,q1
    c.measure(0, 0).measure(1, 1)
    with c.c_if(1, 1): c.x(2)   # Corrections
    with c.c_if(0, 1): c.z(2)
    c.measure(2, 2)
    for seed in range(20):
        _, bits = simulate(c, seed=seed)
        assert bits[2] == 1, f"Teleportation failed with seed={seed}"


# -- Large circuits --

def test_1000_qubit_ghz():
    """1000-qubit GHZ: all measurements must agree. Statevector must be empty."""
    n = 1000
    c = Circuit(n, n)
    c.h(0)
    for i in range(1, n):
        c.cx(i - 1, i)
    for i in range(n):
        c.measure(i, i)
    sv, bits = simulate(c, seed=42)
    assert sv.shape == (0,), "Should return empty statevector for n>25"
    vals = set(bits.values())
    assert len(vals) == 1, f"GHZ qubits disagree: got {len(vals)} distinct values"


# -- Reset --

def test_reset_puts_qubit_in_zero():
    """After reset, qubit always measures 0."""
    c = Circuit(2, 2).h(0).cx(0, 1).reset(0).measure(0, 0).measure(1, 1)
    for seed in range(20):
        _, bits = simulate(c, seed=seed)
        assert bits[0] == 0, f"Reset qubit measured {bits[0]} at seed={seed}"

def test_reset_deterministic():
    """Reset on |0> is a no-op; reset on |1> flips to |0>."""
    c = Circuit(1, 1).x(0).reset(0).measure(0, 0)
    for seed in range(10):
        _, bits = simulate(c, seed=seed)
        assert bits[0] == 0


# -- Auto-dispatch --

def test_simulate_uses_stabilizer():
    """simulate() auto-dispatches Clifford circuits to stabilizer backend."""
    c = Circuit(2).h(0).cx(0, 1)
    sv, _ = simulate(c)
    expected = np.array([1/sqrt(2), 0, 0, 1/sqrt(2)], dtype=complex)
    assert states_equal(sv, expected)

def test_simulate_statevector_with_noise():
    """Noise model forces statevector path."""
    from tinyqubit.simulator.noise import NoiseModel
    c = Circuit(1).h(0)
    nm = NoiseModel()
    nm.add_depolarizing(0.01, [Gate.H])
    sv, _ = simulate(c, noise_model=nm, seed=42)
    assert sv.shape == (2,)

def test_simulate_statevector_with_initial_state():
    """_initial_state forces statevector path."""
    c = Circuit(1).h(0)
    c._initial_state = np.array([1, 0], dtype=complex)
    sv, _ = simulate(c, seed=42)
    assert sv.shape == (2,)


# -- Clifford+T --

def test_clifford_t_info_detection():
    assert clifford_t_info(Circuit(1).h(0).t(0)) == 1
    assert clifford_t_info(Circuit(1).h(0).t(0).tdg(0)) == 2
    assert clifford_t_info(Circuit(1).rx(0, 0.5)) == -1
    assert clifford_t_info(Circuit(1).h(0)) == 0

def test_clifford_t_ht_agreement():
    """H-T circuit matches statevector."""
    c = Circuit(1).h(0).t(0)
    sv_ct, _ = simulate_clifford_t(c)
    sv_ref, _ = _force_statevector(c)
    assert states_equal(sv_ct, sv_ref)

def test_clifford_t_tdg_agreement():
    c = Circuit(1).h(0).tdg(0)
    sv_ct, _ = simulate_clifford_t(c)
    sv_ref, _ = _force_statevector(c)
    assert states_equal(sv_ct, sv_ref)

def test_clifford_t_multi_t():
    """Multiple T gates on different qubits."""
    c = Circuit(3).h(0).h(1).h(2).t(0).cx(0, 1).t(1).cx(1, 2).t(2)
    sv_ct, _ = simulate_clifford_t(c)
    sv_ref, _ = _force_statevector(c)
    assert states_equal(sv_ct, sv_ref)

def test_clifford_t_t_then_cliffords():
    """T followed by Clifford entangling gates."""
    c = Circuit(2).h(0).t(0).cx(0, 1).h(1).s(1)
    sv_ct, _ = simulate_clifford_t(c)
    sv_ref, _ = _force_statevector(c)
    assert states_equal(sv_ct, sv_ref)

def test_clifford_t_deterministic_measure():
    """T on |0> always measures 0 (T is diagonal, doesn't change Z eigenstates)."""
    c = Circuit(1, 1).t(0).measure(0, 0)
    for seed in range(20):
        _, bits = simulate_clifford_t(c, seed=seed)
        assert bits[0] == 0, f"T|0> measured 1 at seed={seed}"

def test_clifford_t_bell_t_measurement_correlation():
    """Entangled pair via T: measurements must correlate."""
    c = Circuit(2, 2).h(0).t(0).cx(0, 1).measure(0, 0).measure(1, 1)
    for seed in range(20):
        _, bits = simulate_clifford_t(c, seed=seed)
        assert bits[0] == bits[1], f"Bell+T pair disagreed at seed={seed}"

def test_clifford_t_reset():
    """Reset works correctly in Clifford+T circuit."""
    c = Circuit(1, 1).h(0).t(0).reset(0).measure(0, 0)
    for seed in range(20):
        _, bits = simulate_clifford_t(c, seed=seed)
        assert bits[0] == 0, f"Reset qubit measured {bits[0]} at seed={seed}"

def test_clifford_t_seed_determinism():
    c = Circuit(2, 2).h(0).t(0).cx(0, 1).measure(0, 0).measure(1, 1)
    results = [simulate_clifford_t(c, seed=99)[1] for _ in range(5)]
    assert all(r == results[0] for r in results)
