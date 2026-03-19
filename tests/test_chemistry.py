"""Tests for quantum chemistry module."""
import numpy as np
import tinyqubit as tq
from tinyqubit.qml.chemistry import (
    jordan_wigner, molecular_hamiltonian, hf_state, uccsd_ansatz,
    _pauli_mul, _simplify, _excitation_generator,
)


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
    assert len(H.terms) == 15  # 1 identity + 4 Z + 6 ZZ + 4 XXYY-type

def test_jw_h2_coefficients():
    """Golden test: H₂ STO-3G JW Hamiltonian coefficients."""
    H, _, _ = molecular_hamiltonian('h2')
    coeffs = {tuple(sorted(p.items())): c for c, p in H.terms}
    assert abs(coeffs[()] - (-0.09281723)) < 1e-5
    assert abs(coeffs[((0, 'Z'),)] - 0.17329634) < 1e-5
    assert abs(coeffs[((2, 'Z'),)] - (-0.22462766)) < 1e-5
    assert abs(coeffs[((0, 'Z'), (1, 'Z'))] - 0.16862656) < 1e-5
    # XXYY-type terms
    xxyy = coeffs[((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y'))]
    assert abs(xxyy - (-0.04532188)) < 1e-5

def test_jw_h2_ground_state_energy():
    """Exact diagonalization should give FCI energy."""
    H, nq, _ = molecular_hamiltonian('h2')
    pauli_mats = {'X': np.array([[0, 1], [1, 0]], dtype=complex),
                  'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                  'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
    I2 = np.eye(2, dtype=complex)
    H_mat = np.zeros((2**nq, 2**nq), dtype=complex)
    for coeff, paulis in H.terms:
        term = np.array([[1]], dtype=complex)
        for q in range(nq):
            term = np.kron(term, pauli_mats[paulis[q]] if q in paulis else I2)
        H_mat += coeff * term
    gs = min(np.linalg.eigvalsh(H_mat).real)
    assert abs(gs - (-1.1386)) < 1e-3

def test_jw_h2_hermitian():
    """Hamiltonian must have real coefficients (imaginary parts cancelled)."""
    H, _, _ = molecular_hamiltonian('h2')
    for c, _ in H.terms:
        assert isinstance(c, float)

def test_jw_custom_integrals():
    """jordan_wigner with trivial 1-orbital system: H = h1 * (I - Z)/2."""
    h1 = np.array([[-0.5]])
    h2 = np.zeros((1, 1, 1, 1))
    obs = jordan_wigner(h1, h2)
    coeffs = {tuple(sorted(p.items())): c for c, p in obs.terms}
    assert abs(coeffs.get((), 0) - (-0.25)) < 1e-10
    assert abs(coeffs.get(((0, 'Z'),), 0) - 0.25) < 1e-10

def test_molecular_hamiltonian_unknown_molecule():
    try:
        molecular_hamiltonian('ch4')
        assert False, "should have raised"
    except ValueError as e:
        assert 'jordan_wigner' in str(e).lower()

def test_h2_default_matches_explicit():
    H1, _, _ = molecular_hamiltonian('h2')
    H2, _, _ = molecular_hamiltonian('h2', bond_length=0.735)
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1.keys() == c2.keys()
    for k in c1:
        assert abs(c1[k] - c2[k]) < 1e-12

def test_h2_variable_bond_length():
    H, nq, ne = molecular_hamiltonian('h2', bond_length=1.0)
    assert nq == 4 and ne == 2 and len(H.terms) == 15
    # Coefficients must differ from default R=0.735
    H_def, _, _ = molecular_hamiltonian('h2')
    c_new = {tuple(sorted(p.items())): c for c, p in H.terms}
    c_def = {tuple(sorted(p.items())): c for c, p in H_def.terms}
    assert abs(c_new[()] - c_def[()]) > 0.01

def test_h2_invalid_bond_length():
    try:
        molecular_hamiltonian('h2', bond_length=0.42)
        assert False, "should have raised"
    except ValueError as e:
        assert '0.735' in str(e)

def test_lih_default():
    H, nq, ne = molecular_hamiltonian('lih')
    assert nq == 4 and ne == 2
    assert len(H.terms) > 15  # more terms than H2 due to off-diagonal integrals

def test_lih_ground_state_energy():
    """Exact diag of LiH CAS(2,2) at equilibrium."""
    H, nq, _ = molecular_hamiltonian('lih')
    pauli_mats = {'X': np.array([[0, 1], [1, 0]], dtype=complex),
                  'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                  'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
    I2 = np.eye(2, dtype=complex)
    H_mat = np.zeros((2**nq, 2**nq), dtype=complex)
    for coeff, paulis in H.terms:
        term = np.array([[1]], dtype=complex)
        for q in range(nq):
            term = np.kron(term, pauli_mats[paulis[q]] if q in paulis else I2)
        H_mat += coeff * term
    gs = min(np.linalg.eigvalsh(H_mat).real)
    assert abs(gs - (-7.8634)) < 0.01

def test_lih_invalid_active_space():
    try:
        molecular_hamiltonian('lih', active_electrons=4, active_orbitals=4)
        assert False, "should have raised"
    except ValueError as e:
        assert 'CAS' in str(e)

def test_h2_active_space_identity():
    H1, _, _ = molecular_hamiltonian('h2')
    H2, _, _ = molecular_hamiltonian('h2', active_electrons=2, active_orbitals=2)
    c1 = {tuple(sorted(p.items())): c for c, p in H1.terms}
    c2 = {tuple(sorted(p.items())): c for c, p in H2.terms}
    assert c1.keys() == c2.keys()
    for k in c1:
        assert abs(c1[k] - c2[k]) < 1e-12


# --- Hartree-Fock state ---

def test_hf_state_correct():
    c = hf_state(4, 2)
    sv, _ = tq.simulate(c)
    # |1100⟩ in MSB ordering: index = 0b1100 = 12
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
    assert len(singles) == 4  # 2 occupied × 2 virtual
    assert len(doubles) == 1  # (0,1) → (2,3)

def test_uccsd_hf_state_included():
    c = uccsd_ansatz(4, 2, include_hf=True)
    assert c.ops[0].gate == tq.Gate.X and c.ops[0].qubits == (0,)
    assert c.ops[1].gate == tq.Gate.X and c.ops[1].qubits == (1,)

def test_uccsd_no_hf():
    c = uccsd_ansatz(4, 2, include_hf=False)
    assert c.ops[0].gate != tq.Gate.X or c.ops[0].qubits != (0,)

def test_uccsd_at_zero_is_hf():
    """All parameters at zero → circuit is identity after HF prep → HF state."""
    c = uccsd_ansatz(4, 2)
    c.init_params(value=0.0)
    sv, _ = tq.simulate(c.bind())
    assert abs(abs(sv[12]) - 1.0) < 1e-10  # |1100⟩

def test_uccsd_hf_energy():
    """HF energy from UCCSD at theta=0 matches direct HF expectation."""
    H, nq, ne = molecular_hamiltonian('h2')
    hf_circ = hf_state(nq, ne)
    e_hf = tq.expectation(hf_circ, H)
    uccsd = uccsd_ansatz(nq, ne)
    uccsd.init_params(value=0.0)
    e_uccsd = tq.expectation(uccsd, H)
    assert abs(e_hf - e_uccsd) < 1e-10

def test_uccsd_gradient_nonzero():
    """Double excitation gradient must be nonzero at HF point."""
    from tinyqubit.qml.optim import adjoint_gradient
    H, nq, ne = molecular_hamiltonian('h2')
    c = uccsd_ansatz(nq, ne)
    c.init_params(value=0.0)
    grads = adjoint_gradient(c, H)
    assert abs(grads['d_0']) > 0.1


# --- Excitation generators ---

def test_single_excitation_generator_hermitian():
    """Single excitation generator must be anti-Hermitian (purely imaginary coefficients)."""
    gen = _excitation_generator(0, 2)
    for c, _ in gen:
        assert abs(c.real) < 1e-10

def test_double_excitation_generator_term_count():
    gen = _excitation_generator((0, 1), (2, 3))
    assert len(gen) == 8
