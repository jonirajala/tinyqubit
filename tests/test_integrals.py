"""Tests for the STO-3G integral engine."""
import numpy as np
from tinyqubit.qml.integrals import (
    _build_basis, _compute_ao_integrals, _nuclear_repulsion, _rhf,
    _boys_array, compute_molecular_integrals,
)
from tinyqubit.qml.chemistry import compute_hamiltonian, molecular_hamiltonian


# --- Boys function ---

def test_boys_f0_zero():
    assert abs(_boys_array(0, 0.0)[0] - 1.0) < 1e-14

def test_boys_f0_one():
    assert abs(_boys_array(0, 1.0)[0] - 0.7468241328) < 1e-8

def test_boys_large_T():
    F = _boys_array(0, 100.0)
    assert abs(F[0] - np.sqrt(np.pi / 400)) < 1e-3

def test_boys_downward_recursion():
    """Downward recursion should give consistent values."""
    T = 15.0
    vals = _boys_array(4, T)
    # Check recursion: F_{n-1} = (2T*F_n + exp(-T)) / (2n-1)
    for n in range(4, 0, -1):
        f_check = (2 * T * vals[n] + np.exp(-T)) / (2 * n - 1)
        assert abs(vals[n - 1] - f_check) < 1e-10


# --- SCF energies ---

def test_h2_scf_energy():
    """H₂ RHF/STO-3G at R=1.4 bohr."""
    symbols = ['H', 'H']
    coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
    basis = _build_basis(symbols, coords)
    S, T, V, eri = _compute_ao_integrals(basis, symbols, coords)
    nuc = _nuclear_repulsion(symbols, coords)
    E_elec, _, _ = _rhf(S, T, V, eri, 2)
    assert abs(E_elec + nuc - (-1.1168)) < 0.01

def test_he_atom_scf():
    """He atom RHF/STO-3G."""
    basis = _build_basis(['He'], np.array([[0., 0., 0.]]))
    S, T, V, eri = _compute_ao_integrals(basis, ['He'], np.array([[0., 0., 0.]]))
    E_elec, _, _ = _rhf(S, T, V, eri, 2)
    assert abs(E_elec - (-2.8078)) < 0.001

def test_be_atom_scf():
    """Be atom RHF/STO-3G."""
    basis = _build_basis(['Be'], np.array([[0., 0., 0.]]))
    S, T, V, eri = _compute_ao_integrals(basis, ['Be'], np.array([[0., 0., 0.]]))
    E_elec, _, _ = _rhf(S, T, V, eri, 4)
    assert abs(E_elec - (-14.352)) < 0.001


# --- Nuclear repulsion ---

def test_nuclear_repulsion_h2():
    """H₂ at R=1.4 bohr: V_nn = 1/1.4."""
    nuc = _nuclear_repulsion(['H', 'H'], np.array([[0., 0., 0.], [0., 0., 1.4]]))
    assert abs(nuc - 1.0 / 1.4) < 1e-10


# --- Basis construction ---

def test_basis_count_h2():
    basis = _build_basis(['H', 'H'], np.array([[0., 0., 0.], [0., 0., 1.4]]))
    assert len(basis) == 2  # 1 s per H

def test_basis_count_h2o():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    basis = _build_basis(['O', 'H', 'H'], coords)
    assert len(basis) == 7  # O: 1s + 2s + 3×2p = 5, H: 1s each = 2


# --- Active space integrals ---

def test_h2_full_space():
    """H₂ with all orbitals active (no frozen core)."""
    h1, h2, nuc, ne = compute_molecular_integrals(
        ['H', 'H'], np.array([[0., 0., 0.], [0., 0., 1.4]]))
    assert h1.shape == (4, 4)  # 2 spatial → 4 spin orbitals
    assert ne == 2

def test_determinism():
    """Same input must give bit-identical output."""
    symbols = ['H', 'H']
    coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
    h1a, h2a, nuca, _ = compute_molecular_integrals(symbols, coords, 2, 2)
    h1b, h2b, nucb, _ = compute_molecular_integrals(symbols, coords, 2, 2)
    assert np.array_equal(h1a, h1b)
    assert np.array_equal(h2a, h2b)
    assert nuca == nucb

def test_arbitrary_geometry():
    """H₂O at non-standard geometry should work without error."""
    geom = np.array([[0., 0., 0.], [0., 0., 2.0], [1.8, 0., -0.5]])
    H, nq, ne = compute_hamiltonian(['O', 'H', 'H'], geom, active_electrons=2, active_orbitals=2)
    assert nq == 4 and ne == 2

def test_lih_scf():
    """LiH RHF/STO-3G at equilibrium."""
    R = 1.546 * 1.8897259886
    basis = _build_basis(['Li', 'H'], np.array([[0., 0., 0.], [0., 0., R]]))
    S, T, V, eri = _compute_ao_integrals(basis, ['Li', 'H'], np.array([[0., 0., 0.], [0., 0., R]]))
    nuc = _nuclear_repulsion(['Li', 'H'], np.array([[0., 0., 0.], [0., 0., R]]))
    E_elec, _, _ = _rhf(S, T, V, eri, 4)
    assert abs(E_elec + nuc - (-7.862)) < 0.01


def test_h2_631g_energy():
    """H₂ RHF/6-31G should be lower than STO-3G."""
    coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
    basis_sto = _build_basis(['H', 'H'], coords, 'sto-3g')
    basis_631 = _build_basis(['H', 'H'], coords, '6-31g')
    S1, T1, V1, eri1 = _compute_ao_integrals(basis_sto, ['H', 'H'], coords)
    S2, T2, V2, eri2 = _compute_ao_integrals(basis_631, ['H', 'H'], coords)
    nuc = _nuclear_repulsion(['H', 'H'], coords)
    e_sto, _, _ = _rhf(S1, T1, V1, eri1, 2)
    e_631, _, _ = _rhf(S2, T2, V2, eri2, 2)
    assert e_631 + nuc < e_sto + nuc  # 6-31G gives lower (better) RHF energy
    assert abs(e_631 + nuc - (-1.1268)) < 0.01  # known H2/6-31G RHF value


def test_d_orbital_overlap():
    """d-orbital self-overlap must be ~1.0 for Fe atom."""
    basis = _build_basis(['Fe'], np.array([[0., 0., 0.]]))
    S, _, _, _ = _compute_ao_integrals(basis, ['Fe'], np.array([[0., 0., 0.]]))
    assert max(abs(np.diag(S) - 1.0)) < 1e-4
