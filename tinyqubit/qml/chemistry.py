"""Quantum chemistry: Jordan-Wigner & Bravyi-Kitaev transforms, molecular Hamiltonians, UCCSD, ADAPT-VQE."""
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


def _to_real_observable(terms: list[tuple[complex, dict]]) -> Observable:
    simplified = _simplify(terms)
    return Observable([(complex(c).real if abs(complex(c).imag) < 1e-10 else c, p) for c, p in simplified])


def _mul_ops(a: list, b: list) -> list:
    return [_pauli_mul(t1, t2) for t1 in a for t2 in b]


# Jordan-Wigner encoding -------

def _jw_op(p: int, create: bool) -> list[tuple[complex, dict]]:
    """JW encoding of a†_p (create=True) or a_p (create=False)."""
    z_string = {k: 'Z' for k in range(p)}
    d0 = dict(z_string); d0[p] = 'X'
    d1 = dict(z_string); d1[p] = 'Y'
    return [(0.5, d0), (-0.5j if create else 0.5j, d1)]


# Bravyi-Kitaev encoding (via Clifford conjugation of JW) -------

_ENC = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
_DEC = {v: k for k, v in _ENC.items()}

def _conjugate_cnot(paulis: dict, phase: complex, ctrl: int, tgt: int) -> tuple[dict, complex]:
    """Heisenberg picture: CNOT(c→t) P CNOT(c→t)†."""
    p = dict(paulis)
    xc, zc = _ENC[p.get(ctrl, 'I')]
    xt, zt = _ENC[p.get(tgt, 'I')]
    xc2, zc2 = xc, zc ^ zt
    xt2, zt2 = xc ^ xt, zt
    phase *= 1j ** ((xc2 * zc2 + xt2 * zt2 - xc * zc - xt * zt) % 4)
    for q, bits in [(ctrl, (xc2, zc2)), (tgt, (xt2, zt2))]:
        s = _DEC[bits]
        if s == 'I': p.pop(q, None)
        else: p[q] = s
    return p, phase


def _clifford_transform(H: Observable, cnots: list[tuple[int, int]]) -> list[tuple[complex, dict]]:
    """Conjugate every term in H by a sequence of CNOTs."""
    result = []
    for coeff, paulis in H.terms:
        p, phase = dict(paulis), complex(coeff)
        for ctrl, tgt in cnots:
            p, phase = _conjugate_cnot(p, phase, ctrl, tgt)
        result.append((phase, p))
    return result


def _bk_cnots(n: int) -> list[tuple[int, int]]:
    """CNOT list implementing the Fenwick-tree BK basis change."""
    cnots = []
    for j in range(n - 1, -1, -1):
        k = j + 1
        for i in range(k - (k & -k), j):
            cnots.append((i, j))
    return cnots


# Hamiltonian construction -------

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


# Public API -------

def jordan_wigner(h1: np.ndarray, h2: np.ndarray, nuclear_repulsion: float = 0.0) -> Observable:
    """Convert one/two-electron integrals to a qubit Hamiltonian via Jordan-Wigner."""
    n = h1.shape[0]
    return _to_real_observable([(nuclear_repulsion, {})] + _jw_one_body(h1, n) + _jw_two_body(h2, n))


def bravyi_kitaev(h1: np.ndarray, h2: np.ndarray, nuclear_repulsion: float = 0.0) -> Observable:
    """Convert one/two-electron integrals to a qubit Hamiltonian via Bravyi-Kitaev."""
    return _to_real_observable(_clifford_transform(jordan_wigner(h1, h2, nuclear_repulsion), _bk_cnots(h1.shape[0])))


_ANG_TO_BOHR = 1.8897259886

def _h2o_geom(R_bohr: float) -> np.ndarray:
    angle = 104.5 * _pi / 180
    return np.array([[0.0, 0.0, 0.0],
                     [R_bohr * np.sin(angle / 2), 0.0, R_bohr * np.cos(angle / 2)],
                     [-R_bohr * np.sin(angle / 2), 0.0, R_bohr * np.cos(angle / 2)]])

def _ch4_geom(R_bohr: float) -> np.ndarray:
    # Tetrahedral: C at origin, 4 H at vertices of a tetrahedron
    t = R_bohr / (3 ** 0.5)
    return np.array([[0., 0., 0.], [t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]])

# (symbols, geometry_fn(R_bohr), default_R_angstrom, default_active_electrons, default_active_orbitals)
_MOLECULES = {
    'h2':   (['H', 'H'], lambda R: np.array([[0., 0., 0.], [0., 0., R]]), 0.735, 2, 2),
    'lih':  (['Li', 'H'], lambda R: np.array([[0., 0., 0.], [0., 0., R]]), 1.546, 4, 4),
    'beh2': (['H', 'Be', 'H'], lambda R: np.array([[0., 0., -R], [0., 0., 0.], [0., 0., R]]), 1.326, 4, 4),
    'n2':   (['N', 'N'], lambda R: np.array([[0., 0., 0.], [0., 0., R]]), 1.098, 4, 4),
    'h2o':  (['O', 'H', 'H'], _h2o_geom, 0.957, 4, 4),
    'ch4':  (['C', 'H', 'H', 'H', 'H'], _ch4_geom, 1.089, 4, 4),
}


def molecular_hamiltonian(molecule: str = 'h2', bond_length: float | None = None,
                          active_electrons: int | None = None, active_orbitals: int | None = None,
                          mapping: str = 'jw', basis: str = 'sto-3g') -> tuple[Observable, int, int]:
    """Build molecular Hamiltonian via SCF. Returns (hamiltonian, n_qubits, n_electrons)."""
    mol = molecule.lower()
    if mol not in _MOLECULES:
        raise ValueError(f"Unknown molecule {molecule!r}. Use compute_hamiltonian() for arbitrary molecules.")
    symbols, geom_fn, default_R, default_ne, default_no = _MOLECULES[mol]
    R = bond_length if bond_length is not None else default_R
    ne = active_electrons or default_ne
    no = active_orbitals or default_no
    geometry = geom_fn(R * _ANG_TO_BOHR)
    return compute_hamiltonian(symbols, geometry, ne, no, mapping=mapping, basis=basis)


def compute_hamiltonian(symbols: list[str], geometry: np.ndarray,
                        active_electrons: int | None = None,
                        active_orbitals: int | None = None,
                        mapping: str = 'jw', basis: str = 'sto-3g') -> tuple[Observable, int, int]:
    """Compute Hamiltonian from atomic symbols + geometry (Bohr). Any molecule, any geometry."""
    from .integrals import compute_molecular_integrals
    h1, h2, nuc, ne = compute_molecular_integrals(symbols, geometry, active_electrons, active_orbitals, basis)
    transform = bravyi_kitaev if mapping == 'bk' else jordan_wigner
    return transform(h1, h2, nuc), h1.shape[0], ne


_PAULI_MAT = {'X': np.array([[0,1],[1,0]], dtype=complex), 'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
              'Z': np.array([[1,0],[0,-1]], dtype=complex)}
_I2 = np.eye(2, dtype=complex)

def exact_diag(H: Observable, n_qubits: int) -> float:
    """Ground state energy by exact diagonalization."""
    H_mat = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for coeff, paulis in H.terms:
        term = np.array([[1]], dtype=complex)
        for q in range(n_qubits):
            term = np.kron(term, _PAULI_MAT[paulis[q]] if q in paulis else _I2)
        H_mat += coeff * term
    return float(np.linalg.eigvalsh(H_mat).real.min())


def hf_state(n_qubits: int, n_electrons: int) -> Circuit:
    """Hartree-Fock reference state: X gates on first n_electrons qubits."""
    c = Circuit(n_qubits)
    for i in range(n_electrons):
        c.x(i)
    return c


def taper(H: Observable, n_qubits: int, n_electrons: int) -> tuple[Observable, int]:
    """Remove 2 qubits from a JW Hamiltonian via Z₂ spin-parity symmetries.

    Returns (tapered_hamiltonian, new_n_qubits).
    """
    if n_qubits < 4:
        raise ValueError(f"taper requires >= 4 qubits, got {n_qubits}")
    pivots = {n_qubits - 2, n_qubits - 1}
    cnots = [(q, n_qubits - 2) for q in range(0, n_qubits - 2, 2)]
    cnots += [(q, n_qubits - 1) for q in range(1, n_qubits - 1, 2)]
    n_alpha = (n_electrons + 1) // 2
    eigenvalues = {n_qubits - 2: (-1) ** n_alpha, n_qubits - 1: (-1) ** (n_electrons // 2)}
    kept = [q for q in range(n_qubits) if q not in pivots]
    remap = {old: new for new, old in enumerate(kept)}
    transformed = _clifford_transform(H, cnots)
    tapered = []
    for phase, p in transformed:
        skip = False
        for piv in pivots:
            op = p.pop(piv, 'I')
            if op == 'Z':
                phase *= eigenvalues[piv]
            elif op != 'I':
                skip = True
                break
        if skip:
            continue
        tapered.append((phase, {remap[q]: s for q, s in p.items() if q in remap}))
    return _to_real_observable(tapered), n_qubits - 2


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


def _apply_gen(c: Circuit, occ, virt, param: Parameter) -> None:
    for coeff, paulis in _excitation_generator(occ, virt):
        _exp_pauli(c, param, paulis, negate=coeff.imag > 0)


def _excitation_pool(n_qubits: int, n_electrons: int) -> list:
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    pool = [(i, a) for i in occupied for a in virtual]
    pool += [((i, j), (a, b)) for ii, i in enumerate(occupied) for j in occupied[ii + 1:]
             for aa, a in enumerate(virtual) for b in virtual[aa + 1:]]
    return pool


def uccsd_ansatz(n_qubits: int, n_electrons: int, include_hf: bool = True,
                 spin_adapted: bool = False) -> Circuit:
    """UCCSD ansatz with Trotterized single and double excitations.

    spin_adapted=True groups alpha/beta excitations under shared parameters.
    """
    c = Circuit(n_qubits)
    if include_hf:
        for i in range(n_electrons):
            c.x(i)
    if not spin_adapted:
        pool = _excitation_pool(n_qubits, n_electrons)
        s_idx = d_idx = 0
        for occ, virt in pool:
            if isinstance(occ, int):
                _apply_gen(c, occ, virt, Parameter(f"s_{s_idx}")); s_idx += 1
            else:
                _apply_gen(c, occ, virt, Parameter(f"d_{d_idx}")); d_idx += 1
    else:
        n_orb, n_occ = n_qubits // 2, n_electrons // 2
        occ_sp, virt_sp = list(range(n_occ)), list(range(n_occ, n_orb))
        idx = 0
        for i in occ_sp:
            for a in virt_sp:
                p = Parameter(f"s_{idx}")
                for spin in (0, 1):
                    _apply_gen(c, 2*i+spin, 2*a+spin, p)
                idx += 1
        idx = 0
        for ii, i in enumerate(occ_sp):
            for j in occ_sp[ii + 1:]:
                for aa, a in enumerate(virt_sp):
                    for b in virt_sp[aa + 1:]:
                        p = Parameter(f"d_{idx}")
                        seen = set()
                        for si in (0, 1):
                            for sj in (0, 1):
                                oi, oj = 2*i+si, 2*j+sj
                                if oi == oj: continue
                                for sa in (0, 1):
                                    for sb in (0, 1):
                                        va, vb = 2*a+sa, 2*b+sb
                                        if va == vb: continue
                                        key = (min(oi,oj), max(oi,oj), min(va,vb), max(va,vb))
                                        if key in seen: continue
                                        seen.add(key)
                                        _apply_gen(c, key[:2], key[2:], p)
                        idx += 1
    return c


# ADAPT-VQE -------

def _pool_gradients(circuit: Circuit, H: Observable, pool: list) -> np.ndarray:
    from .optim import adjoint_gradient
    grads = np.empty(len(pool))
    for i, (occ, virt) in enumerate(pool):
        trial = Circuit(circuit.n_qubits)
        trial.ops = list(circuit.ops)
        trial.param_values = dict(circuit.param_values)
        p = Parameter(f"_pool_{i}")
        _apply_gen(trial, occ, virt, p)
        trial.param_values[p.name] = 0.0
        grads[i] = adjoint_gradient(trial, H)[p.name]
    return grads


def adapt_vqe(H: Observable, n_qubits: int, n_electrons: int,
              max_iters: int = 20, threshold: float = 1e-3,
              opt_steps: int = 100, stepsize: float = 0.1) -> tuple[Circuit, float, list]:
    """ADAPT-VQE: grow ansatz by iteratively selecting operators with largest gradient.

    Returns (circuit, energy, history) where history is [(energy, max_gradient), ...].
    """
    from .optim import Adam
    from ..measurement.observable import expectation
    pool = _excitation_pool(n_qubits, n_electrons)
    circuit = hf_state(n_qubits, n_electrons)
    history, energy = [], 0.0
    for it in range(max_iters):
        grads = _pool_gradients(circuit, H, pool)
        idx = int(np.argmax(np.abs(grads)))
        max_grad = float(np.abs(grads[idx]))
        if max_grad < threshold:
            break
        _apply_gen(circuit, *pool[idx], Parameter(f"a_{it}"))
        circuit.param_values[f"a_{it}"] = 0.0
        opt = Adam(stepsize=stepsize)
        for _ in range(opt_steps):
            opt.step(circuit, H)
        energy = expectation(circuit.bind(), H)
        history.append((float(energy), max_grad))
    return circuit, float(energy), history
