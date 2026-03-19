"""Quantum chemistry: Jordan-Wigner transform, molecular Hamiltonians, UCCSD ansatz."""
from __future__ import annotations
import numpy as np
from math import pi as _pi
from ..ir import Circuit, Parameter
from ..measurement.observable import Observable

# Pauli multiplication table: (result_pauli, phase)
_PAULI_MULT = {
    ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
    ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
    ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
}


def _pauli_mul(t1: tuple[complex, dict], t2: tuple[complex, dict]) -> tuple[complex, dict]:
    """Multiply two Pauli terms (coeff, {qubit: pauli})."""
    c = t1[0] * t2[0]
    merged = dict(t1[1])
    for q, p2 in t2[1].items():
        if q in merged:
            p1 = merged[q]
            if p1 == p2:
                del merged[q]
            else:
                result, phase = _PAULI_MULT[(p1, p2)]
                c *= phase
                if result == 'I':
                    del merged[q]
                else:
                    merged[q] = result
        else:
            merged[q] = p2
    return (c, merged)


def _simplify(terms: list[tuple[complex, dict]]) -> list[tuple[complex, dict]]:
    """Combine like terms, drop near-zero, sort deterministically."""
    grouped: dict[tuple, complex] = {}
    for c, paulis in terms:
        key = tuple(sorted(paulis.items()))
        grouped[key] = grouped.get(key, 0) + c
    return [(c, dict(k)) for k, c in sorted(grouped.items()) if abs(c) > 1e-10]


def _jw_op(p: int, create: bool) -> list[tuple[complex, dict]]:
    """JW encoding of a†_p (create=True) or a_p (create=False)."""
    z_string = {k: 'Z' for k in range(p)}
    d0 = dict(z_string); d0[p] = 'X'
    d1 = dict(z_string); d1[p] = 'Y'
    return [(0.5, d0), (-0.5j if create else 0.5j, d1)]


def _mul_ops(a: list, b: list) -> list:
    return [_pauli_mul(t1, t2) for t1 in a for t2 in b]


def _jw_one_body(h1: np.ndarray, n: int) -> list[tuple[complex, dict]]:
    terms = []
    for p in range(n):
        for q in range(n):
            if abs(h1[p, q]) < 1e-15:
                continue
            op = _mul_ops(_jw_op(p, True), _jw_op(q, False))
            terms.extend((h1[p, q] * c, d) for c, d in op)
    return terms


def _jw_two_body(h2: np.ndarray, n: int) -> list[tuple[complex, dict]]:
    # H₂ = (1/2) Σ_{pqrs} <pq|rs> a†_p a†_q a_s a_r
    # <pq|rs>_physicist = h2_chemist[p,r,q,s]
    terms = []
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    v = h2[p, r, q, s]
                    if abs(v) < 1e-15:
                        continue
                    op = _mul_ops(_mul_ops(_mul_ops(
                        _jw_op(p, True), _jw_op(q, True)),
                        _jw_op(s, False)), _jw_op(r, False))
                    terms.extend((0.5 * v * c, d) for c, d in op)
    return terms


# STO-3G integrals in chemist notation (Mulliken) for 2-spatial-orbital systems.
# Each tuple: (h1_00, h1_01, h1_11, h2_0000, h2_0001, h2_0011, h2_0101, h2_0111, h2_1111, nuc)
# h2 indices use (pq|rs) convention with 8-fold symmetry.
_H2_DATA = {
    0.5:   (-1.4105283931923402, 0.0, -0.2569357501101742, 0.7197060466662034, 0.0, 0.7072398465608030, 0.1688702254936380, 0.0, 0.7448393735424563, 1.0583544979881958),
    0.6:   (-1.3422140240241811, 0.0, -0.3657705374371827, 0.7013377385823411, 0.0, 0.6887931040096968, 0.1737306409794547, 0.0, 0.7245060293379857, 0.8819620816568299),
    0.735: (-1.2563390730032498, 0.0, -0.4718960244306283, 0.6745062250002953, 0.0, 0.6631370544950876, 0.1812875358076718, 0.0, 0.6973161242283644, 0.7199689944489797),
    0.8:   (-1.2178260641715115, 0.0, -0.5096378496128872, 0.6633301613086797, 0.0, 0.6534413812804418, 0.1846267796145410, 0.0, 0.6867915430949000, 0.6614715612426223),
    1.0:   (-1.1108442164433172, 0.0, -0.5891209862310223, 0.6264025138010334, 0.0, 0.6217067734311319, 0.1967905783476326, 0.0, 0.6530707563359346, 0.5291772489940979),
    1.5:   (-0.9081809084385146, 0.0, -0.6653369318569269, 0.5527033964734940, 0.0, 0.5596841665141924, 0.2295359287681033, 0.0, 0.5834207729512356, 0.3527848326627320),
    2.0:   (-0.7789220655576415, 0.0, -0.6702666763402987, 0.5094628221516899, 0.0, 0.5192012675407549, 0.2591384672205846, 0.0, 0.5346641304400876, 0.2645886244970490),
    2.5:   (-0.7001473137381311, 0.0, -0.6540677457758881, 0.4856801055932184, 0.0, 0.4931151113548152, 0.2822100389690942, 0.0, 0.5020597975885854, 0.2116708995976392),
}
_LIH_DATA = {
    # CAS(2,2) active-space integrals; nuc includes frozen-core energy
    1.0:   (-0.8409202346526006, 0.0388118604638738, -0.4069794512341456, 0.5242631026169444, -0.0388118621781447, 0.2466468715703858, 0.0094659307750216, -0.0013893963764109, 0.3390039481920167, -6.6097847347216554),
    1.2:   (-0.8278713130323999, 0.0423369808722260, -0.3857320790079779, 0.5148767890831005, -0.0423369825138071, 0.2376720772075236, 0.0101850782556243, 0.0019915742907765, 0.3399470181481741, -6.6947499748738082),
    1.546: (-0.7808777812837662, -0.0476791296327596, -0.3591840949370927, 0.4910674023733631, 0.0476791311646736, 0.2251974447211591, 0.0125450500384619, -0.0067713558207440, 0.3384181549103699, -6.7924455429464441),
    1.8:   (-0.7413731941530436, 0.0521977063894693, -0.3456342567125165, 0.4736133021401983, -0.0521977075183405, 0.2185509934515109, 0.0154267121004898, 0.0101267417760618, 0.3352660909605326, -6.8408856418061186),
    2.0:   (-0.7103842529519923, -0.0563190495464320, -0.3377713261556503, 0.4602775368273803, 0.0563190503782486, 0.2148354472226688, 0.0186205964433050, -0.0127497025320093, 0.3316631350922356, -6.8704146570838454),
    2.5:   (-0.6381171930415036, 0.0697660378190973, -0.3283613454824998, 0.4288779243881071, -0.0697660385835551, 0.2130131319069598, 0.0323303291064438, 0.0180436174978116, 0.3177514725398956, -6.9235172693019189),
    3.0:   (-0.5764664510696279, -0.0895333498866370, -0.3364804286399476, 0.4009794995901804, 0.0895333505871813, 0.2273700099167799, 0.0610302648064006, -0.0146537056059983, 0.2960111849454110, -6.9588765701171518),
}
_MOLECULE_DATA = {'h2': (_H2_DATA, 0.735, 2, 2), 'lih': (_LIH_DATA, 1.546, 2, 2)}


def _unpack_2orb(data: tuple) -> tuple[np.ndarray, np.ndarray, float]:
    """Reconstruct h1 (2×2) and h2 (2×2×2×2) from compact 10-tuple."""
    h1_00, h1_01, h1_11, h2_0000, h2_0001, h2_0011, h2_0101, h2_0111, h2_1111, nuc = data
    h1 = np.array([[h1_00, h1_01], [h1_01, h1_11]])
    h2 = np.zeros((2, 2, 2, 2))
    for (p, q, r, s), v in [((0,0,0,0), h2_0000), ((0,0,0,1), h2_0001), ((0,0,1,1), h2_0011),
                              ((0,1,0,1), h2_0101), ((0,1,1,1), h2_0111), ((1,1,1,1), h2_1111)]:
        for a, b, c, d in [(p,q,r,s),(q,p,r,s),(p,q,s,r),(q,p,s,r),
                            (r,s,p,q),(s,r,p,q),(r,s,q,p),(s,r,q,p)]:
            h2[a, b, c, d] = v
    return h1, h2, nuc


def _spatial_to_spin(h1_sp: np.ndarray, h2_sp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatial (2-orbital) → spin-orbital (4-orbital) integrals."""
    n = 4
    h1, h2 = np.zeros((n, n)), np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            if p % 2 != q % 2:
                continue
            h1[p, q] = h1_sp[p // 2, q // 2]
            for r in range(n):
                for s in range(n):
                    if r % 2 == s % 2:
                        h2[p, q, r, s] = h2_sp[p // 2, q // 2, r // 2, s // 2]
    return h1, h2


def _get_integrals(molecule: str, bond_length: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Look up integrals for a molecule at a given bond length."""
    data_table, _, _, _ = _MOLECULE_DATA[molecule]
    if bond_length not in data_table:
        available = sorted(data_table.keys())
        raise ValueError(f"Bond length {bond_length} not available for {molecule!r}. Available: {available}")
    h1_sp, h2_sp, nuc = _unpack_2orb(data_table[bond_length])
    h1, h2 = _spatial_to_spin(h1_sp, h2_sp)
    return h1, h2, nuc


# Public API -------

def jordan_wigner(h1: np.ndarray, h2: np.ndarray, nuclear_repulsion: float = 0.0) -> Observable:
    """Convert one/two-electron integrals to a qubit Hamiltonian via Jordan-Wigner."""
    n = h1.shape[0]
    terms = [(nuclear_repulsion, {})] + _jw_one_body(h1, n) + _jw_two_body(h2, n)
    simplified = _simplify(terms)
    return Observable([(complex(c).real if abs(complex(c).imag) < 1e-10 else c, p) for c, p in simplified])


def molecular_hamiltonian(molecule: str = 'h2', bond_length: float | None = None,
                          active_electrons: int | None = None, active_orbitals: int | None = None) -> tuple[Observable, int, int]:
    """Build molecular Hamiltonian. Returns (hamiltonian, n_qubits, n_electrons).

    Supported: 'h2' (STO-3G, 4 qubits) and 'lih' (STO-3G CAS(2,2), 4 qubits).
    """
    mol = molecule.lower()
    if mol not in _MOLECULE_DATA:
        raise ValueError(f"Unknown molecule {molecule!r}. Use jordan_wigner() with custom integrals.")
    _, default_r, default_ne, default_no = _MOLECULE_DATA[mol]
    ne = active_electrons or default_ne
    no = active_orbitals or default_no
    if (ne, no) != (default_ne, default_no):
        raise ValueError(f"Only CAS({default_ne},{default_no}) supported for {molecule!r}.")
    r = bond_length if bond_length is not None else default_r
    h1, h2, nuc = _get_integrals(mol, r)
    return jordan_wigner(h1, h2, nuc), 4, ne


def hf_state(n_qubits: int, n_electrons: int) -> Circuit:
    """Hartree-Fock reference state: X gates on first n_electrons qubits."""
    c = Circuit(n_qubits)
    for i in range(n_electrons):
        c.x(i)
    return c


# UCCSD ansatz via Trotterized excitation generators -------

def _excitation_generator(occupied, virtual) -> list[tuple[complex, dict]]:
    """JW Pauli terms for anti-Hermitian excitation generator (a†...a - h.c.)."""
    if isinstance(occupied, int):
        fwd = _mul_ops(_jw_op(virtual, True), _jw_op(occupied, False))
        bwd = _mul_ops(_jw_op(occupied, True), _jw_op(virtual, False))
    else:
        i, j = occupied
        a, b = virtual
        fwd = _mul_ops(_mul_ops(_mul_ops(
            _jw_op(a, True), _jw_op(b, True)),
            _jw_op(j, False)), _jw_op(i, False))
        bwd = _mul_ops(_mul_ops(_mul_ops(
            _jw_op(i, True), _jw_op(j, True)),
            _jw_op(b, False)), _jw_op(a, False))
    terms = fwd + [(-c, d) for c, d in bwd]
    return [(c, d) for c, d in _simplify(terms) if d]


def _exp_pauli(c: Circuit, param: Parameter, paulis: dict, negate: bool = False) -> None:
    """Append exp(-i * param/2 * P) block. negate flips sign (for generator terms with positive imag coeff)."""
    qubits = sorted(paulis.keys())
    for q in qubits:
        if paulis[q] == 'X': c.h(q)
        elif paulis[q] == 'Y': c.rx(q, _pi / 2)
    for k in range(len(qubits) - 1):
        c.cx(qubits[k], qubits[k + 1])
    if negate:
        c.x(qubits[-1])
    c.rz(qubits[-1], param)
    if negate:
        c.x(qubits[-1])
    for k in range(len(qubits) - 2, -1, -1):
        c.cx(qubits[k], qubits[k + 1])
    for q in qubits:
        if paulis[q] == 'X': c.h(q)
        elif paulis[q] == 'Y': c.rx(q, -_pi / 2)


def uccsd_ansatz(n_qubits: int, n_electrons: int, include_hf: bool = True) -> Circuit:
    """UCCSD ansatz with Trotterized single and double excitations."""
    c = Circuit(n_qubits)
    if include_hf:
        for i in range(n_electrons):
            c.x(i)
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    idx = 0
    for i in occupied:
        for a in virtual:
            gen = _excitation_generator(i, a)
            param = Parameter(f"s_{idx}")
            for coeff, paulis in gen:
                _exp_pauli(c, param, paulis, negate=coeff.imag > 0)
            idx += 1
    idx = 0
    for ii, i in enumerate(occupied):
        for jj, j in enumerate(occupied):
            if jj <= ii:
                continue
            for aa, a in enumerate(virtual):
                for bb, b in enumerate(virtual):
                    if bb <= aa:
                        continue
                    gen = _excitation_generator((i, j), (a, b))
                    param = Parameter(f"d_{idx}")
                    for coeff, paulis in gen:
                        _exp_pauli(c, param, paulis, negate=coeff.imag > 0)
                    idx += 1
    return c
