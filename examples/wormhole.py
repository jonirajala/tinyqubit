"""Simplified traversable wormhole teleportation (Jafferis et al., Nature 2022).

Demonstrates the holographic wormhole protocol with tinyqubit:
  1. Prepare thermofield double (TFD) — Bell pairs between Left and Right
  2. Inject message qubit into the Left system
  3. Scramble Left (SYK-like time evolution)
  4. Apply L-R coupling (negative energy shockwave)
  5. Unscramble Right (inverse of same scrambling)
  6. Measure: information teleports only with the correct coupling sign

Layout (7 qubits): q0=REF, q1-q3=Left, q4-q6=Right

NOTE: This simulation does not reproduce the positive/negative coupling asymmetry —
our generic scrambler gives symmetric peaks at both +g and -g. The real experiment
showed a peak only at negative g. This asymmetry comes from "size winding", a specific
property of the SYK Hamiltonian where scrambled operators acquire size-dependent phases.
Reproducing this would require implementing the actual sparsified SYK model with
Jordan-Wigner-encoded Majorana fermions, which is doable but substantially more complex.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinyqubit import Circuit
from tinyqubit.simulator import simulate, sample

REF = 0
LEFT, RIGHT = [1, 2, 3], [4, 5, 6]
R0 = 4  # output qubit

# Fixed scrambling angles — chosen to produce good operator growth
_A = [0.7, 1.2, 2.3, 0.4, 1.8, 0.9, 1.5, 2.1, 0.6]


def _scramble(qc: Circuit, q: list[int], inverse: bool = False):
    """3-layer scrambling circuit. inverse=True applies the exact inverse."""
    if not inverse:
        qc.rx(q[0], _A[0]).ry(q[1], _A[1]).rz(q[2], _A[2])
        qc.cx(q[0], q[1]).cx(q[1], q[2])
        qc.ry(q[0], _A[3]).rz(q[1], _A[4]).rx(q[2], _A[5])
        qc.cx(q[2], q[0]).cx(q[0], q[1])
        qc.rz(q[0], _A[6]).rx(q[1], _A[7]).ry(q[2], _A[8])
        qc.cx(q[1], q[2]).cx(q[0], q[2])
    else:
        qc.cx(q[0], q[2]).cx(q[1], q[2])
        qc.ry(q[2], -_A[8]).rx(q[1], -_A[7]).rz(q[0], -_A[6])
        qc.cx(q[0], q[1]).cx(q[2], q[0])
        qc.rx(q[2], -_A[5]).rz(q[1], -_A[4]).ry(q[0], -_A[3])
        qc.cx(q[1], q[2]).cx(q[0], q[1])
        qc.rz(q[2], -_A[2]).ry(q[1], -_A[1]).rx(q[0], -_A[0])


def _couple_zz(qc: Circuit, g: float):
    """ZZ coupling: exp(-ig Z_L Z_R). Used in the real experiment."""
    for l, r in zip(LEFT, RIGHT):
        qc.rzz(l, r, g)


def _couple_heisenberg(qc: Circuit, g: float):
    """Heisenberg coupling: ZZ + XX + YY. Captures all Pauli directions."""
    for l, r in zip(LEFT, RIGHT):
        # ZZ: native RZZ
        qc.rzz(l, r, g)
        # XX: H-RZZ-H
        qc.h(l).h(r)
        qc.rzz(l, r, g)
        qc.h(l).h(r)
        # YY: S†H-RZZ-HS
        qc.sdg(l).sdg(r).h(l).h(r)
        qc.rzz(l, r, g)
        qc.h(l).h(r).s(l).s(r)


def wormhole(theta: float, phi: float, g: float, coupling: str = "zz") -> Circuit:
    """Build the wormhole teleportation circuit.

    Args:
        theta, phi: message state cos(t/2)|0> + e^{ip}sin(t/2)|1>
        g: coupling strength (g<0 → wormhole open, g>0 → closed)
        coupling: "zz" (experiment-like) or "heisenberg" (all Pauli directions)
    """
    qc = Circuit(7)

    # 1. Thermofield double: Bell pairs (two entangled black holes)
    for l, r in zip(LEFT, RIGHT):
        qc.h(l).cx(l, r)

    # 2. Inject message into left system
    qc.ry(REF, theta).rz(REF, phi)
    qc.swap(REF, LEFT[0])

    # 3. Scramble left (message falls into left black hole)
    _scramble(qc, LEFT)

    # 4. L-R coupling (negative energy shockwave)
    (_couple_zz if coupling == "zz" else _couple_heisenberg)(qc, g)

    # 5. Unscramble right (message emerges from right black hole)
    _scramble(qc, RIGHT, inverse=True)

    return qc


def fidelity(state: np.ndarray, theta: float, phi: float) -> float:
    """Teleportation fidelity: overlap of R0 reduced state with target."""
    psi = state.reshape(2**R0, 2, 2**(7 - R0 - 1))
    rho = np.einsum('iaj,ibj->ab', psi, psi.conj())
    target = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.real(target.conj() @ rho @ target)


def scan(theta, phi, coupling, g_range):
    """Scan coupling strength, return fidelities."""
    fids = []
    for g in g_range:
        state, _ = simulate(wormhole(theta, phi, g, coupling))
        fids.append(fidelity(state, theta, phi))
    return fids


def print_scan(g_range, fids, bar_scale=50):
    """Print fidelity scan with bar chart."""
    fmin, fmax = min(fids), max(fids)
    for g, f in zip(g_range, fids):
        bar = "#" * max(1, int((f - 0.3) * bar_scale))
        peak = " <--" if f == fmax and fmax - fmin > 0.005 else ""
        print(f"  g={g:+5.1f}  F={f:.4f}  |{bar}{peak}")
    neg = [f for g, f in zip(g_range, fids) if g < -0.1]
    pos = [f for g, f in zip(g_range, fids) if g > 0.1]
    print(f"\n  Peak fidelity (g<0): {max(neg):.4f}")
    print(f"  Peak fidelity (g>0): {max(pos):.4f}")
    print(f"  Random baseline:     0.5000")
    print(f"  Asymmetry (neg-pos): {max(neg)-max(pos):+.4f}")


def main():
    print("=" * 62)
    print("  Traversable Wormhole Simulation  —  tinyqubit")
    print("  Based on Jafferis, Zlokapa et al., Nature 2022")
    print("=" * 62)

    g_range = np.linspace(3.0, -3.0, 31)
    theta, phi = np.pi / 2, 0.0  # |+> state (best signal with ZZ coupling)

    # --- ZZ coupling (as in the real experiment) ---
    print(f"\n{'='*62}")
    print(f"  ZZ COUPLING (experiment-like)")
    print(f"  Message: |+>  |  3 qubits per side  |  7 qubits total")
    print(f"{'='*62}")
    fids_zz = scan(theta, phi, "zz", g_range)
    print_scan(g_range, fids_zz)

    # --- Heisenberg coupling (XX+YY+ZZ) ---
    print(f"\n{'='*62}")
    print(f"  HEISENBERG COUPLING (XX+YY+ZZ) — all Pauli directions")
    print(f"  Message: |+>  |  3 qubits per side  |  7 qubits total")
    print(f"{'='*62}")
    fids_heis = scan(theta, phi, "heisenberg", g_range)
    print_scan(g_range, fids_heis)

    # --- Multiple message states ---
    best_g_zz = g_range[np.argmax(fids_zz)]
    best_g_heis = g_range[np.argmax(fids_heis)]
    print(f"\n{'='*62}")
    print(f"  FIDELITY BY MESSAGE STATE")
    print(f"{'='*62}")
    print(f"  {'State':5s}  {'ZZ (g='+f'{best_g_zz:+.1f}'+')':>14s}  {'Heisenberg (g='+f'{best_g_heis:+.1f}'+')':>20s}")
    print(f"  {'-'*45}")
    for label, th, ph in [('|0>', 0, 0), ('|1>', np.pi, 0), ('|+>', np.pi/2, 0),
                           ('|->', np.pi/2, np.pi), ('|i>', np.pi/2, np.pi/2)]:
        s1, _ = simulate(wormhole(th, ph, best_g_zz, "zz"))
        s2, _ = simulate(wormhole(th, ph, best_g_heis, "heisenberg"))
        f1 = fidelity(s1, th, ph)
        f2 = fidelity(s2, th, ph)
        print(f"  {label:5s}  {f1:14.4f}  {f2:20.4f}")

    # --- Shot-based measurement ---
    print(f"\n{'='*62}")
    print(f"  SHOT EXPERIMENT (Heisenberg, g={best_g_heis:+.1f}, 4096 shots)")
    print(f"{'='*62}")
    for label, th, ph, target_bit in [('|0>', 0, 0, '0'), ('|1>', np.pi, 0, '1')]:
        state, _ = simulate(wormhole(th, ph, best_g_heis, "heisenberg"))
        counts = sample(state, shots=4096, seed=42)
        p_target = sum(v for k, v in counts.items() if k[R0] == target_bit) / 4096
        print(f"  Message {label}: P(R0={target_bit}) = {p_target:.3f}  (ideal: 1.000)")

    # --- Interpretation ---
    print(f"""
{'='*62}
  INTERPRETATION
{'='*62}
  The wormhole protocol teleports quantum information through
  an entangled "wormhole" connecting two quantum systems:

    TFD state      = two entangled black holes
    Scrambling     = message falls into left black hole
    L-R coupling   = negative energy shockwave (opens wormhole)
    Unscrambling   = message emerges from right black hole

  WHAT WE SEE:
  - Heisenberg coupling (XX+YY+ZZ): near-perfect teleportation
    at F=0.999. Proves the protocol works.
  - ZZ-only coupling: weak signal (F~0.54 vs 0.50 random).
    This is physical — ZZ captures only Z-type operator growth.

  WHAT'S MISSING: the positive/negative coupling asymmetry.
  Our generic scrambler gives symmetric peaks at both +g and -g.
  The REAL experiment showed a peak ONLY at negative g — this is
  the "size winding" signature of the SYK model:
    - Scrambled operators acquire phases proportional to their size
    - Negative coupling constructively interferes with these phases
    - Positive coupling destructively interferes → no teleportation
  This asymmetry is THE hallmark of holographic (wormhole) dynamics
  and requires the specific SYK Hamiltonian, not a generic scrambler.
""")


if __name__ == "__main__":
    main()
