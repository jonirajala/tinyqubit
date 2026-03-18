"""Simplified traversable wormhole teleportation (Jafferis et al., Nature 2022).

Demonstrates the holographic wormhole protocol using tinyqubit:
  1. Prepare thermofield double (TFD) — Bell pairs between Left and Right systems
  2. Inject a qubit of information into the Left system
  3. Scramble the Left system (SYK-like time evolution)
  4. Apply ZZ coupling between L and R (negative energy shockwave)
  5. Unscramble the Right system (inverse of same scrambling unitary)
  6. Measure: information should teleport to Right only with negative coupling

Layout (7 qubits):
  q0 = reference qubit (carries the message)
  q1, q2, q3 = Left system (L0, L1, L2)
  q4, q5, q6 = Right system (R0, R1, R2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tinyqubit import Circuit
from tinyqubit.simulator import simulate, sample

REF, L0, L1, L2, R0, R1, R2 = range(7)

# Fixed scrambling angles (reproducible "random" SYK-like interactions)
_ANGLES = [0.7, 1.2, 2.3, 0.4, 1.8, 0.9, 1.5, 2.1, 0.6]


def _scramble_forward(qc: Circuit, qubits: list[int]):
    """Deep scrambling unitary — 3 layers of entangling gates."""
    q0, q1, q2 = qubits
    a = _ANGLES
    # Layer 1: single-qubit rotations + entangling
    qc.rx(q0, a[0]).ry(q1, a[1]).rz(q2, a[2])
    qc.cx(q0, q1).cx(q1, q2)
    # Layer 2
    qc.ry(q0, a[3]).rz(q1, a[4]).rx(q2, a[5])
    qc.cx(q2, q0).cx(q0, q1)
    # Layer 3
    qc.rz(q0, a[6]).rx(q1, a[7]).ry(q2, a[8])
    qc.cx(q1, q2).cx(q0, q2)


def _scramble_inverse(qc: Circuit, qubits: list[int]):
    """Exact inverse of _scramble_forward: reverse gate order, negate angles."""
    q0, q1, q2 = qubits
    a = _ANGLES
    # Layer 3 inverse
    qc.cx(q0, q2).cx(q1, q2)
    qc.ry(q2, -a[8]).rx(q1, -a[7]).rz(q0, -a[6])
    # Layer 2 inverse
    qc.cx(q0, q1).cx(q2, q0)
    qc.rx(q2, -a[5]).rz(q1, -a[4]).ry(q0, -a[3])
    # Layer 1 inverse
    qc.cx(q1, q2).cx(q0, q1)
    qc.rz(q2, -a[2]).ry(q1, -a[1]).rx(q0, -a[0])


def _coupling(qc: Circuit, left: list[int], right: list[int], g: float):
    """Gao-Jafferis-Wall coupling: exp(-ig * sum_k Z_Lk Z_Rk).

    In the gravity dual, g < 0 → negative energy → opens the wormhole.
    """
    for l, r in zip(left, right):
        qc.rzz(l, r, g)


def wormhole_circuit(msg_state: tuple[float, float], g: float) -> Circuit:
    """Build the full wormhole teleportation circuit.

    Args:
        msg_state: (theta, phi) for the message: cos(t/2)|0> + e^{ip}sin(t/2)|1>
        g: coupling strength. g < 0 → traversable wormhole, g > 0 → closed.
    """
    theta, phi = msg_state
    left, right = [L0, L1, L2], [R0, R1, R2]
    qc = Circuit(7)

    # Step 1: Thermofield double — Bell pairs between L and R
    # Holographic dual: two entangled black holes connected by a wormhole
    for l, r in zip(left, right):
        qc.h(l)
        qc.cx(l, r)

    # Step 2: Prepare and inject message into the left system
    qc.ry(REF, theta)
    qc.rz(REF, phi)
    qc.swap(REF, L0)

    # Step 3: Forward time evolution on Left (scrambling)
    # Gravity dual: message falls into the left black hole and scrambles
    _scramble_forward(qc, left)

    # Step 4: L-R coupling (the negative energy shockwave)
    # Gravity dual: negative energy pulse that renders the wormhole traversable
    _coupling(qc, left, right, g)

    # Step 5: Inverse time evolution on Right (unscrambling)
    # Gravity dual: message emerges from the right black hole
    _scramble_inverse(qc, right)

    return qc


def reduced_density_matrix(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Partial trace to get 2x2 reduced density matrix of a single qubit (MSB ordering)."""
    before = 2 ** qubit
    after = 2 ** (n_qubits - qubit - 1)
    psi = state.reshape(before, 2, after)
    return np.einsum('iaj,ibj->ab', psi, psi.conj())


def state_fidelity(state: np.ndarray, target_theta: float, target_phi: float) -> float:
    """Fidelity of R0's reduced state with the target message state."""
    rho = reduced_density_matrix(state, R0, 7)
    target = np.array([np.cos(target_theta / 2), np.exp(1j * target_phi) * np.sin(target_theta / 2)])
    return np.real(target.conj() @ rho @ target)


def main():
    print("=" * 60)
    print("  Traversable Wormhole Simulation (tinyqubit)")
    print("  Based on Jafferis et al., Nature 2022")
    print("=" * 60)

    # Test with |1> state (theta=pi, phi=0)
    theta, phi = np.pi, 0.0
    print(f"\nMessage: |1> state  (theta=pi, phi=0)")

    couplings = np.linspace(3.0, -3.0, 31)
    print(f"\n{'Coupling g':>12s}  {'Fidelity':>10s}  {'Bar'}")
    print("-" * 60)

    fidelities = []
    for g in couplings:
        qc = wormhole_circuit((theta, phi), g)
        state, _ = simulate(qc)
        f = state_fidelity(state, theta, phi)
        fidelities.append(f)
        bar = "#" * int(f * 40)
        print(f"  g = {g:+6.2f}    {f:8.4f}  |{bar}")

    neg_fids = [(g, f) for g, f in zip(couplings, fidelities) if g < 0]
    pos_fids = [(g, f) for g, f in zip(couplings, fidelities) if g > 0]
    best_neg_g, best_neg_f = max(neg_fids, key=lambda x: x[1])
    best_pos_g, best_pos_f = max(pos_fids, key=lambda x: x[1])

    print(f"\nBest fidelity (g < 0, wormhole open):   {best_neg_f:.4f}  at g={best_neg_g:+.2f}")
    print(f"Best fidelity (g > 0, wormhole closed):  {best_pos_f:.4f}  at g={best_pos_g:+.2f}")
    print(f"Random guess baseline:                   0.5000")

    if best_neg_f > best_pos_f + 0.01:
        print("\n>> Negative coupling enables higher teleportation fidelity")
        print(">> — the wormhole is traversable!")
    elif best_pos_f > best_neg_f + 0.01:
        print("\n>> Positive coupling has higher fidelity (asymmetry in scrambling)")
        print(">> Swapping sign convention would show the wormhole effect.")
    else:
        print("\n>> Fidelities are similar — scrambling circuit may need tuning.")

    # Shot-based experiment at the best coupling
    best_g = best_neg_g if best_neg_f > best_pos_f else best_pos_g
    best_f = max(best_neg_f, best_pos_f)
    print(f"\n--- Shot experiment at g = {best_g:+.2f} (1024 shots) ---")
    qc = wormhole_circuit((theta, phi), best_g)
    state, _ = simulate(qc)
    # Measure just R0
    qc2 = Circuit(7)
    qc2._initial_state = state
    qc2.measure(R0, 0)
    _, classical = simulate(qc2)
    print(f"  Single-shot R0 = {classical[0]}")

    counts = sample(state, shots=1024, seed=42)
    r0_one = sum(v for k, v in counts.items() if k[R0] == '1')
    r0_zero = 1024 - r0_one
    print(f"  R0 = |0>: {r0_zero}/1024,  R0 = |1>: {r0_one}/1024")
    print(f"  P(target |1>) = {r0_one/1024:.3f}  (fidelity: {best_f:.4f})")

    # Verify: try multiple message states
    print(f"\n--- Fidelity across different message states (g = {best_g:+.2f}) ---")
    for label, th, ph in [('|0>', 0, 0), ('|1>', np.pi, 0), ('|+>', np.pi/2, 0),
                           ('|-> ', np.pi/2, np.pi), ('|i>', np.pi/2, np.pi/2)]:
        qc = wormhole_circuit((th, ph), best_g)
        state, _ = simulate(qc)
        f = state_fidelity(state, th, ph)
        print(f"  {label:4s}  fidelity = {f:.4f}")


if __name__ == "__main__":
    main()
