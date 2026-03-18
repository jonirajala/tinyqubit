"""Traversable wormhole teleportation with SYK model (Jafferis et al., Nature 2022).

Implements the holographic wormhole protocol with the actual SYK Hamiltonian
using Jordan-Wigner-encoded Majorana fermions:
  1. Prepare thermofield double (TFD) — Bell pairs between Left and Right
  2. Inject message qubit into the Left system
  3. Scramble Left (SYK time evolution via Trotterized Pauli rotations)
  4. Apply L-R coupling (negative energy shockwave)
  5. Unscramble Right (inverse SYK evolution)
  6. Measure: information teleports only with one sign of coupling

Layout (7 qubits): q0=REF, q1-q3=Left, q4-q6=Right

Jordan-Wigner encoding (N=6 Majoranas → 3 qubits per side):
  chi_0 = X_0,  chi_1 = Y_0,  chi_2 = Z_0 X_1,
  chi_3 = Z_0 Y_1,  chi_4 = Z_0 Z_1 X_2,  chi_5 = Z_0 Z_1 Y_2

SYK q=4 Hamiltonian: H = sum_{a<b<c<d} J_{abcd} chi_a chi_b chi_c chi_d
Each 4-Majorana product simplifies to a Pauli string; C(6,4) = 15 terms.

exp(-iHt) is implemented via first-order Trotterization: each Pauli string
rotation exp(-i*theta*P) decomposes as basis-change → CNOT staircase → RZ → undo.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import combinations
from tinyqubit import Circuit
from tinyqubit.simulator import simulate

REF = 0
LEFT, RIGHT = [1, 2, 3], [4, 5, 6]
R0 = 4  # output qubit
N_MAJ = 6  # Majorana fermions per side → 3 qubits

# Pauli algebra (I=0, X=1, Y=2, Z=3) ---------
# _PMUL[a][b] = (result, phase_power) where Pa * Pb = i^phase * P_result
_PMUL = [[(0, 0), (1, 0), (2, 0), (3, 0)],   # I
         [(1, 0), (0, 0), (3, 1), (2, 3)],   # X
         [(2, 0), (3, 3), (0, 0), (1, 1)],   # Y
         [(3, 0), (2, 1), (1, 3), (0, 0)]]   # Z
_PNAME = 'IXYZ'


def _pmul(p1, p2):
    """Multiply two Pauli strings, return (result, phase_mod_4)."""
    phase = 0
    r = []
    for a, b in zip(p1, p2):
        ri, pi = _PMUL[a][b]
        r.append(ri)
        phase = (phase + pi) % 4
    return tuple(r), phase


def _pauli_str(p):
    return ''.join(_PNAME[i] for i in p)


# Jordan-Wigner encoding ---------

def _majorana(nq, idx):
    """Majorana chi_idx as Pauli string. chi_{2k} = Z^k X_k, chi_{2k+1} = Z^k Y_k."""
    p = [0] * nq
    q = idx // 2
    for i in range(q):
        p[i] = 3
    p[q] = 1 + idx % 2
    return tuple(p)


# SYK Hamiltonian ---------

def _syk_terms(seed=42):
    """Build SYK q=4 Hamiltonian: list of (coeff, pauli_string) terms."""
    nq = N_MAJ // 2
    rng = np.random.RandomState(seed)
    majors = [_majorana(nq, i) for i in range(N_MAJ)]
    sigma = np.sqrt(6.0 / N_MAJ ** 3)  # Standard SYK normalization: (q-1)!/N^(q-1)
    terms = []
    for a, b, c, d in combinations(range(N_MAJ), 4):
        p, phase = majors[a], 0
        for k in (b, c, d):
            p, ph = _pmul(p, majors[k])
            phase = (phase + ph) % 4
        assert phase % 2 == 0, "4-Majorana product must be Hermitian"
        sign = 1 if phase == 0 else -1
        if any(x != 0 for x in p):
            terms.append((sign * sigma * rng.randn(), p))
    return terms


def _commutes(p1, p2):
    """Two Pauli strings commute iff they anti-commute at an even number of sites."""
    return sum(1 for a, b in zip(p1, p2) if a and b and a != b) % 2 == 0


def _commuting_subset(terms):
    """Greedy maximal commuting subset, largest couplings first."""
    ordered = sorted(terms, key=lambda t: abs(t[0]), reverse=True)
    sel = [ordered[0]]
    for t in ordered[1:]:
        if all(_commutes(t[1], s[1]) for s in sel):
            sel.append(t)
    return sel


# Pauli rotation circuit: exp(-i * angle * P) ---------

def _pauli_rot(qc, pauli, qubits, angle):
    """Decompose exp(-i * angle * P) into native gates on given qubits."""
    active = [(qubits[i], pauli[i]) for i in range(len(pauli)) if pauli[i]]
    if not active:
        return
    # Basis change: X→Z via H, Y→Z via SDG+H
    for q, p in active:
        if p == 1: qc.h(q)
        elif p == 2: qc.sdg(q).h(q)
    # CNOT staircase — computes parity into last qubit
    for i in range(len(active) - 1):
        qc.cx(active[i][0], active[i + 1][0])
    # Phase rotation
    qc.rz(active[-1][0], 2 * angle)
    # Undo CNOT staircase
    for i in range(len(active) - 2, -1, -1):
        qc.cx(active[i][0], active[i + 1][0])
    # Undo basis change
    for q, p in active:
        if p == 1: qc.h(q)
        elif p == 2: qc.h(q).s(q)


# SYK time evolution ---------

def _evolve(qc, qubits, terms, t, n_trotter=1, inverse=False):
    """Apply exp(-iHt) via first-order Trotter. inverse=True applies exp(+iHt)."""
    dt = (-t if inverse else t) / n_trotter
    ordered = list(reversed(terms)) if inverse else terms
    for _ in range(n_trotter):
        for coeff, pauli in ordered:
            _pauli_rot(qc, pauli, qubits, coeff * dt)


# Wormhole circuit ---------

def wormhole(theta, phi, g, t=1.0, seed=42, n_trotter=10, commuting=False):
    """Build the wormhole teleportation circuit with SYK scrambling.

    Args:
        theta, phi: message state cos(t/2)|0> + e^{ip}sin(t/2)|1>
        g: coupling strength (asymmetry: one sign teleports, the other doesn't)
        t: scrambling time
        seed: random seed for SYK couplings
        n_trotter: Trotter steps (more = more accurate; irrelevant if commuting=True)
        commuting: if True, use only commuting terms (exact, no Trotter error)
    """
    terms = _syk_terms(seed)
    if commuting:
        terms = _commuting_subset(terms)
        n_trotter = 1  # Exact for commuting terms
    qc = Circuit(7)

    # 1. Thermofield double: Bell pairs
    for l, r in zip(LEFT, RIGHT):
        qc.h(l).cx(l, r)

    # 2. Inject message
    qc.ry(REF, theta).rz(REF, phi)
    qc.swap(REF, LEFT[0])

    # 3. Scramble left (SYK time evolution)
    _evolve(qc, LEFT, terms, t, n_trotter)

    # 4. L-R coupling (negative energy shockwave)
    for l, r in zip(LEFT, RIGHT):
        qc.rzz(l, r, g)

    # 5. Unscramble right (inverse SYK)
    _evolve(qc, RIGHT, terms, t, n_trotter, inverse=True)

    return qc


def fidelity(state, theta, phi):
    """Teleportation fidelity: overlap of R0 reduced state with target."""
    psi = state.reshape(2 ** R0, 2, 2 ** (7 - R0 - 1))
    rho = np.einsum('iaj,ibj->ab', psi, psi.conj())
    tgt = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return float(np.real(tgt.conj() @ rho @ tgt))


def scan(theta, phi, g_range, **kw):
    return [fidelity(simulate(wormhole(theta, phi, g, **kw))[0], theta, phi) for g in g_range]


def print_scan(g_range, fids, bar_scale=50):
    fmax = max(fids)
    for g, f in zip(g_range, fids):
        bar = "#" * max(1, int((f - 0.3) * bar_scale))
        peak = " <--" if abs(f - fmax) < 1e-8 else ""
        print(f"  g={g:+5.2f}  F={f:.4f}  |{bar}{peak}")
    neg = [f for g, f in zip(g_range, fids) if g < -0.1]
    pos = [f for g, f in zip(g_range, fids) if g > 0.1]
    if neg and pos:
        asym = max(neg) - max(pos)
        sign = "NEGATIVE g wins" if asym > 0 else "POSITIVE g wins"
        print(f"\n  Peak fidelity (g<0): {max(neg):.4f}")
        print(f"  Peak fidelity (g>0): {max(pos):.4f}")
        print(f"  Random baseline:     0.5000")
        print(f"  Asymmetry (neg-pos): {asym:+.4f}  [{sign}]")


def main():
    print("=" * 62)
    print("  SYK Wormhole Teleportation  —  tinyqubit")
    print("  Based on Jafferis, Zlokapa et al., Nature 2022")
    print("=" * 62)

    g_range = np.linspace(3.0, -3.0, 31)
    theta, phi = np.pi / 2, 0.0

    # --- Print SYK model info ---
    terms = _syk_terms(seed=42)
    comm = _commuting_subset(terms)
    print(f"\n  SYK model: N={N_MAJ} Majoranas, {N_MAJ // 2} qubits/side, 7 total")
    print(f"  Hamiltonian: {len(terms)} Pauli terms (C({N_MAJ},4) 4-body interactions)")
    print(f"  Commuting subset: {len(comm)} terms (exact evolution, no Trotter error)")
    comm_paulis = {p for _, p in comm}
    print(f"\n  Terms (coeff × Pauli string):")
    for coeff, pauli in terms[:8]:
        mark = ' ✓' if pauli in comm_paulis else ''
        print(f"    {coeff:+.4f} × {_pauli_str(pauli)}{mark}")
    if len(terms) > 8:
        print(f"    ... ({len(terms) - 8} more)")

    # --- Full SYK (Trotterized) ---
    print(f"\n{'=' * 62}")
    print(f"  FULL SYK ({len(terms)} terms, 20 Trotter steps)")
    print(f"  Message: |+>  |  t=2.0")
    print(f"{'=' * 62}")
    fids_full = scan(theta, phi, g_range, t=2.0, seed=42, n_trotter=20)
    print_scan(g_range, fids_full)

    # --- Commuting subset (exact) ---
    print(f"\n{'=' * 62}")
    print(f"  COMMUTING SUBSET ({len(comm)} terms, exact evolution)")
    print(f"  Message: |+>  |  t=2.0")
    print(f"{'=' * 62}")
    fids_comm = scan(theta, phi, g_range, t=2.0, seed=42, commuting=True)
    print_scan(g_range, fids_comm)

    # --- Multiple message states at best g ---
    best_g = g_range[np.argmax(fids_full)]
    print(f"\n{'=' * 62}")
    print(f"  FIDELITY BY MESSAGE STATE (full SYK, g={best_g:+.2f})")
    print(f"{'=' * 62}")
    for label, th, ph in [('|0>', 0, 0), ('|1>', np.pi, 0), ('|+>', np.pi / 2, 0),
                           ('|->', np.pi / 2, np.pi), ('|i>', np.pi / 2, np.pi / 2)]:
        s, _ = simulate(wormhole(th, ph, best_g, t=2.0, seed=42, n_trotter=20))
        print(f"  {label:5s}  F={fidelity(s, th, ph):.4f}")

    # --- Interpretation ---
    print(f"""
{'=' * 62}
  INTERPRETATION
{'=' * 62}
  The wormhole protocol teleports quantum information through
  an entangled "wormhole" connecting two quantum systems:

    TFD state      = two entangled black holes
    Scrambling     = message falls into left black hole (SYK dynamics)
    L-R coupling   = negative energy shockwave (opens wormhole)
    Unscrambling   = message emerges from right black hole

  KEY RESULT: The +g/-g asymmetry. Unlike a generic scrambler (which
  gives symmetric peaks at both signs), the SYK model's "size winding"
  property causes constructive interference at one coupling sign and
  destructive interference at the other. This asymmetry is the hallmark
  of the holographic (wormhole-like) dynamics in the SYK model.

  NOTE: Kobrin, Schuster & Yao (Nature 2025) showed that for commuting
  Hamiltonians (like the sparsified model in the original experiment),
  "perfect size winding" is a generic property — not necessarily a
  gravitational signature. The debate continues.
""")


if __name__ == "__main__":
    main()
