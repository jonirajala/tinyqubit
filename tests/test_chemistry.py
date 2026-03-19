"""Tests for quantum chemistry module."""
import numpy as np
import tinyqubit as tq
from tinyqubit.qml.chemistry import (
    jordan_wigner, bravyi_kitaev, molecular_hamiltonian, compute_hamiltonian, hf_state, uccsd_ansatz,
    _pauli_mul, _simplify, _excitation_generator, _PAULI_MAT, _I2,
)

def _exact_gs(mol, **kwargs):
    H, nq, _ = molecular_hamiltonian(mol, **kwargs)
    H_mat = np.zeros((2**nq, 2**nq), dtype=complex)
    for coeff, paulis in H.terms:
        term = np.array([[1]], dtype=complex)
        for q in range(nq):
            term = np.kron(term, _PAULI_MAT[paulis[q]] if q in paulis else _I2)
        H_mat += coeff * term
    return float(min(np.linalg.eigvalsh(H_mat).real))


# --- Pauli algebra ---

def test_pauli_mul_identity():
    assert _pauli_mul((1.0, {}), (2.0, {0: 'X'})) == (2.0, {0: 'X'})

def test_pauli_mul_same_cancels():
    c, d = _pauli_mul((1.0, {0: 'X'}), (1.0, {0: 'X'}))
    assert abs(c - 1.0) < 1e-15 and d == {}

def test_pauli_mul_xy_gives_iz():
    c, d = _pauli_mul((1.0, {0: 'X'}), (1.0, {0: 'Y'}))
    assert abs(c - 1j) < 1e-15 and d == {0: 'Z'}

def test_pauli_mul_disjoint_qubits():
    c, d = _pauli_mul((2.0, {0: 'X'}), (3.0, {1: 'Z'}))
    assert abs(c - 6.0) < 1e-15 and d == {0: 'X', 1: 'Z'}

def test_simplify_combines_like_terms():
    terms = [(1.0, {0: 'X'}), (2.0, {0: 'X'}), (0.5, {1: 'Z'})]
    result = _simplify(terms)
    coeffs = {tuple(sorted(p.items())): c for c, p in result}
    assert abs(coeffs[((0, 'X'),)] - 3.0) < 1e-10
    assert abs(coeffs[((1, 'Z'),)] - 0.5) < 1e-10

def test_simplify_drops_near_zero():
    terms = [(1e-12, {0: 'X'}), (1.0, {1: 'Z'})]
    result = _simplify(terms)
    assert len(result) == 1

def test_simplify_deterministic():
    terms = [(0.5, {1: 'Z', 0: 'X'}), (0.3, {0: 'Y'})]
    r1 = _simplify(terms)
    r2 = _simplify(list(reversed(terms)))
    assert r1 == r2


# --- Jordan-Wigner transform ---

def test_jw_h2_term_count():
    H, nq, ne = molecular_hamiltonian('h2')
    assert nq == 4 and ne == 2
    assert len(H.terms) == 15

def test_jw_h2_ground_state_energy():
    """FCI energy from exact diag — runtime integrals may differ slightly from PySCF."""
    assert abs(_exact_gs('h2') - (-1.137)) < 0.01

def test_jw_h2_hermitian():
    H, _, _ = molecular_hamiltonian('h2')
    for c, _ in H.terms:
        assert isinstance(c, float)

def test_jw_custom_integrals():
    h1 = np.array([[-0.5]])
    h2 = np.zeros((1, 1, 1, 1))
    obs = jordan_wigner(h1, h2)
    coeffs = {tuple(sorted(p.items())): c for c, p in obs.terms}
    assert abs(coeffs.get((), 0) - (-0.25)) < 1e-10
    assert abs(coeffs.get(((0, 'Z'),), 0) - 0.25) < 1e-10

def test_molecular_hamiltonian_unknown_molecule():
    try:
        molecular_hamiltonian('xyz_fake')
        assert False, "should have raised"
    except ValueError as e:
        assert 'compute_hamiltonian' in str(e).lower()

def test_h2_default_matches_explicit():
    H1, _, _ = molecular_hamiltonian('h2')
    H2, _, _ = molecular_hamiltonian('h2', bond_length=0.735)
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1.keys() == c2.keys()
    for k in c1:
        assert abs(c1[k] - c2[k]) < 1e-12

def test_h2_variable_bond_length():
    """Any bond length now works — no longer limited to hardcoded values."""
    H, nq, ne = molecular_hamiltonian('h2', bond_length=1.0)
    assert nq == 4 and ne == 2 and len(H.terms) == 15
    H_def, _, _ = molecular_hamiltonian('h2')
    c_new = {tuple(sorted(p.items())): c for c, p in H.terms}
    c_def = {tuple(sorted(p.items())): c for c, p in H_def.terms}
    assert abs(c_new[()] - c_def[()]) > 0.01

def test_h2_arbitrary_bond_length():
    """Any bond length is accepted — including previously unavailable ones."""
    H, nq, ne = molecular_hamiltonian('h2', bond_length=0.42)
    assert nq == 4 and ne == 2

def test_lih_default():
    H, nq, ne = molecular_hamiltonian('lih')
    assert nq == 8 and ne == 4
    assert len(H.terms) > 50

def test_lih_cas22():
    """Explicit CAS(2,2) still works."""
    H, nq, ne = molecular_hamiltonian('lih', active_electrons=2, active_orbitals=2)
    assert nq == 4 and ne == 2

def test_lih_ground_state_energy():
    assert abs(_exact_gs('lih', active_electrons=2, active_orbitals=2) - (-7.86)) < 0.02

def test_h2_active_space_identity():
    H1, _, _ = molecular_hamiltonian('h2')
    H2, _, _ = molecular_hamiltonian('h2', active_electrons=2, active_orbitals=2)
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1.keys() == c2.keys()
    for k in c1:
        assert abs(c1[k] - c2[k]) < 1e-12


# --- BeH₂ ---

def test_beh2_default():
    H, nq, ne = molecular_hamiltonian('beh2')
    assert nq == 8 and ne == 4
    assert len(H.terms) >= 50

def test_beh2_cas22():
    H, nq, ne = molecular_hamiltonian('beh2', active_electrons=2, active_orbitals=2)
    assert nq == 4 and ne == 2

def test_beh2_ground_state_energy():
    assert abs(_exact_gs('beh2', active_electrons=2, active_orbitals=2) - (-15.55)) < 0.05

def test_beh2_variable_bond_length():
    H, nq, ne = molecular_hamiltonian('beh2', bond_length=1.0)
    assert nq == 8 and ne == 4
    c_new = {tuple(sorted(p.items())): c for c, p in H.terms}
    H_def, _, _ = molecular_hamiltonian('beh2')
    c_def = {tuple(sorted(p.items())): c for c, p in H_def.terms}
    assert abs(c_new[()] - c_def[()]) > 0.01


# --- H₂O ---

def test_h2o_default():
    H, nq, ne = molecular_hamiltonian('h2o')
    assert nq == 8 and ne == 4
    assert len(H.terms) >= 50

def test_h2o_ground_state_energy():
    assert abs(_exact_gs('h2o', active_electrons=2, active_orbitals=2) - (-74.96)) < 0.05

def test_h2o_variable_bond_length():
    H, nq, ne = molecular_hamiltonian('h2o', bond_length=0.8)
    assert nq == 8 and ne == 4
    c_new = {tuple(sorted(p.items())): c for c, p in H.terms}
    H_def, _, _ = molecular_hamiltonian('h2o')
    c_def = {tuple(sorted(p.items())): c for c, p in H_def.terms}
    assert abs(c_new[()] - c_def[()]) > 0.01


# --- compute_hamiltonian (arbitrary molecules) ---

def test_compute_hamiltonian_h2():
    """compute_hamiltonian with explicit geometry matches molecular_hamiltonian."""
    H1, nq1, ne1 = molecular_hamiltonian('h2')
    geom = np.array([[0., 0., 0.], [0., 0., 0.735 * 1.8897259886]])
    H2, nq2, ne2 = compute_hamiltonian(['H', 'H'], geom, active_electrons=2, active_orbitals=2)
    assert nq1 == nq2 and ne1 == ne2
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1.keys() == c2.keys()
    for k in c1:
        assert abs(c1[k] - c2[k]) < 1e-10


# --- Hartree-Fock state ---

def test_hf_state_correct():
    c = hf_state(4, 2)
    sv, _ = tq.simulate(c)
    assert abs(abs(sv[12]) - 1.0) < 1e-10

def test_hf_state_structure():
    c = hf_state(4, 2)
    assert len(c.ops) == 2
    assert all(op.gate == tq.Gate.X for op in c.ops)
    assert c.ops[0].qubits == (0,) and c.ops[1].qubits == (1,)


# --- UCCSD ansatz ---

def test_uccsd_parameter_count():
    c = uccsd_ansatz(4, 2)
    params = c.parameters
    singles = [p for p in params if p.name.startswith('s_')]
    doubles = [p for p in params if p.name.startswith('d_')]
    assert len(singles) == 4
    assert len(doubles) == 1

def test_uccsd_hf_state_included():
    c = uccsd_ansatz(4, 2, include_hf=True)
    assert c.ops[0].gate == tq.Gate.X and c.ops[0].qubits == (0,)
    assert c.ops[1].gate == tq.Gate.X and c.ops[1].qubits == (1,)

def test_uccsd_no_hf():
    c = uccsd_ansatz(4, 2, include_hf=False)
    assert c.ops[0].gate != tq.Gate.X or c.ops[0].qubits != (0,)

def test_uccsd_at_zero_is_hf():
    c = uccsd_ansatz(4, 2)
    c.init_params(value=0.0)
    sv, _ = tq.simulate(c.bind())
    assert abs(abs(sv[12]) - 1.0) < 1e-10

def test_uccsd_hf_energy():
    H, nq, ne = molecular_hamiltonian('h2')
    hf_circ = hf_state(nq, ne)
    e_hf = tq.expectation(hf_circ, H)
    uccsd = uccsd_ansatz(nq, ne)
    uccsd.init_params(value=0.0)
    e_uccsd = tq.expectation(uccsd, H)
    assert abs(e_hf - e_uccsd) < 1e-10

def test_uccsd_gradient_nonzero():
    from tinyqubit.qml.optim import adjoint_gradient
    H, nq, ne = molecular_hamiltonian('h2')
    c = uccsd_ansatz(nq, ne)
    c.init_params(value=0.0)
    grads = adjoint_gradient(c, H)
    assert abs(grads['d_0']) > 0.01

def test_uccsd_spin_adapted_fewer_params():
    c_full = uccsd_ansatz(8, 4)
    c_spin = uccsd_ansatz(8, 4, spin_adapted=True)
    assert len(c_spin.parameters) < len(c_full.parameters)

def test_uccsd_spin_adapted_hf_energy():
    H, nq, ne = molecular_hamiltonian('lih')
    c = uccsd_ansatz(nq, ne, spin_adapted=True)
    c.init_params(value=0.0)
    e_sa = tq.expectation(c, H)
    e_hf = tq.expectation(hf_state(nq, ne), H)
    assert abs(e_sa - e_hf) < 1e-10


# --- Excitation generators ---

def test_single_excitation_generator_hermitian():
    gen = _excitation_generator(0, 2)
    for c, _ in gen:
        assert abs(c.real) < 1e-10

def test_double_excitation_generator_term_count():
    gen = _excitation_generator((0, 1), (2, 3))
    assert len(gen) == 8


# --- ADAPT-VQE ---

def test_adapt_vqe_h2_converges():
    from tinyqubit.qml.chemistry import adapt_vqe
    H, nq, ne = molecular_hamiltonian('h2')
    circuit, energy, history = adapt_vqe(H, nq, ne, max_iters=10, opt_steps=200, stepsize=0.1)
    assert abs(energy - (-1.137)) < 0.01

def test_adapt_vqe_returns_evaluable_circuit():
    from tinyqubit.qml.chemistry import adapt_vqe
    H, nq, ne = molecular_hamiltonian('h2')
    circuit, energy, history = adapt_vqe(H, nq, ne, max_iters=5, opt_steps=50)
    e_check = tq.expectation(circuit, H)
    assert abs(e_check - energy) < 1e-6

def test_adapt_vqe_history_structure():
    from tinyqubit.qml.chemistry import adapt_vqe
    H, nq, ne = molecular_hamiltonian('h2')
    _, _, history = adapt_vqe(H, nq, ne, max_iters=3, opt_steps=20)
    assert len(history) <= 3
    for e, g in history:
        assert isinstance(e, float) and isinstance(g, float)

def test_adapt_vqe_threshold_stops_early():
    from tinyqubit.qml.chemistry import adapt_vqe
    H, nq, ne = molecular_hamiltonian('h2')
    _, _, history = adapt_vqe(H, nq, ne, max_iters=20, threshold=1.0, opt_steps=50)
    assert len(history) < 20

def test_adapt_vqe_improves_over_hf():
    """ADAPT-VQE energy should be significantly below HF."""
    from tinyqubit.qml.chemistry import adapt_vqe
    H, nq, ne = molecular_hamiltonian('h2')
    e_hf = tq.expectation(hf_state(nq, ne), H)
    _, energy, _ = adapt_vqe(H, nq, ne, max_iters=5, opt_steps=100)
    assert energy < e_hf - 0.01


# --- Bravyi-Kitaev mapping ---

def _hamiltonian_matrix(H: Observable, nq: int) -> np.ndarray:
    H_mat = np.zeros((2**nq, 2**nq), dtype=complex)
    for coeff, paulis in H.terms:
        term = np.array([[1]], dtype=complex)
        for q in range(nq):
            term = np.kron(term, _PAULI_MAT[paulis[q]] if q in paulis else _I2)
        H_mat += coeff * term
    return H_mat

def test_bk_h2_eigenvalues_match_jw():
    """BK and JW Hamiltonians must have identical eigenvalues."""
    H_jw, nq, _ = molecular_hamiltonian('h2', mapping='jw')
    H_bk, _, _ = molecular_hamiltonian('h2', mapping='bk')
    eigs_jw = sorted(np.linalg.eigvalsh(_hamiltonian_matrix(H_jw, nq)).real)
    eigs_bk = sorted(np.linalg.eigvalsh(_hamiltonian_matrix(H_bk, nq)).real)
    np.testing.assert_allclose(eigs_jw, eigs_bk, atol=1e-10)

def test_bk_lih_eigenvalues_match_jw():
    H_jw, nq, _ = molecular_hamiltonian('lih', mapping='jw')
    H_bk, _, _ = molecular_hamiltonian('lih', mapping='bk')
    eigs_jw = sorted(np.linalg.eigvalsh(_hamiltonian_matrix(H_jw, nq)).real)
    eigs_bk = sorted(np.linalg.eigvalsh(_hamiltonian_matrix(H_bk, nq)).real)
    np.testing.assert_allclose(eigs_jw, eigs_bk, atol=1e-10)

def test_bk_hermitian():
    H, _, _ = molecular_hamiltonian('h2', mapping='bk')
    for c, _ in H.terms:
        assert isinstance(c, float)

def test_bk_custom_integrals():
    h1 = np.array([[-0.5]])
    h2 = np.zeros((1, 1, 1, 1))
    obs_jw = jordan_wigner(h1, h2)
    obs_bk = bravyi_kitaev(h1, h2)
    c_jw = {tuple(sorted(p.items())): c for c, p in obs_jw.terms}
    c_bk = {tuple(sorted(p.items())): c for c, p in obs_bk.terms}
    assert c_jw.keys() == c_bk.keys()
    for k in c_jw:
        assert abs(c_jw[k] - c_bk[k]) < 1e-10

def test_bk_default_is_jw():
    H1, _, _ = molecular_hamiltonian('h2')
    H2, _, _ = molecular_hamiltonian('h2', mapping='jw')
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1 == c2

def test_bk_ground_state_energy():
    """BK ground state from exact diag must match JW."""
    H_bk, nq, _ = molecular_hamiltonian('h2', mapping='bk')
    eigs = np.linalg.eigvalsh(_hamiltonian_matrix(H_bk, nq)).real
    assert abs(min(eigs) - (-1.137)) < 0.01


# --- Qubit tapering ---

def _gs_from_observable(H, nq):
    return float(min(np.linalg.eigvalsh(_hamiltonian_matrix(H, nq)).real))

def test_taper_h2_preserves_energy():
    from tinyqubit.qml.chemistry import taper
    H, nq, ne = molecular_hamiltonian('h2', active_electrons=2, active_orbitals=2)
    H_tap, nq_tap = taper(H, nq, ne)
    assert nq_tap == 2
    assert abs(_gs_from_observable(H, nq) - _gs_from_observable(H_tap, nq_tap)) < 1e-8

def test_taper_lih_cas22():
    from tinyqubit.qml.chemistry import taper
    H, nq, ne = molecular_hamiltonian('lih', active_electrons=2, active_orbitals=2)
    H_tap, nq_tap = taper(H, nq, ne)
    assert nq_tap == 2
    assert abs(_gs_from_observable(H, nq) - _gs_from_observable(H_tap, nq_tap)) < 1e-8

def test_taper_lih_cas44():
    from tinyqubit.qml.chemistry import taper
    H, nq, ne = molecular_hamiltonian('lih')
    H_tap, nq_tap = taper(H, nq, ne)
    assert nq_tap == 6
    assert abs(_gs_from_observable(H, nq) - _gs_from_observable(H_tap, nq_tap)) < 1e-6

def test_taper_hermitian():
    from tinyqubit.qml.chemistry import taper
    H, nq, ne = molecular_hamiltonian('h2', active_electrons=2, active_orbitals=2)
    H_tap, _ = taper(H, nq, ne)
    for c, _ in H_tap.terms:
        assert isinstance(c, float)

def test_taper_reduces_terms():
    from tinyqubit.qml.chemistry import taper
    H, nq, ne = molecular_hamiltonian('h2', active_electrons=2, active_orbitals=2)
    H_tap, _ = taper(H, nq, ne)
    assert len(H_tap.terms) < len(H.terms)
