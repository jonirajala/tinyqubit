"""Stabilizer (tableau) simulator — CHP algorithm, O(n^2) per gate, Clifford-only."""
from __future__ import annotations
import numpy as np
from ..ir import Circuit, Gate

_CLIFFORD_GATES = frozenset({Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.S, Gate.SDG, Gate.SX,
                              Gate.CX, Gate.CZ, Gate.SWAP, Gate.MEASURE, Gate.RESET})


def is_clifford(circuit: Circuit) -> bool:
    return all(op.gate in _CLIFFORD_GATES for op in circuit.ops)


def _rowmult_phase(x1, z1, x2, z2, r1: bool, r2: bool, n: int) -> bool:
    """Compute phase bit of Pauli row product. Shared by tableau and Gaussian elimination."""
    phase = 2 * int(r1) + 2 * int(r2)
    for j in range(n):
        if x2[j] and z2[j]:  # source=Y
            phase += (1 if (x1[j] and not z1[j]) else (-1 if (not x1[j] and z1[j]) else 0))
        elif x2[j]:  # source=X
            phase += (1 if (not x1[j] and z1[j]) else (-1 if (x1[j] and z1[j]) else 0))
        elif z2[j]:  # source=Z: g = x1*(2*z1-1)
            phase += (1 if (x1[j] and z1[j]) else (-1 if (x1[j] and not z1[j]) else 0))
    return (phase % 4) == 2


class StabilizerState:
    """Aaronson-Gottesman stabilizer tableau.

    State: 2n rows (n destabilizers + n stabilizers), each row is (x[n], z[n], r).
    x[i,j] and z[i,j] encode Pauli on qubit j for row i. r[i] is the phase bit (0 or 1 -> +1 or -1).
    """
    __slots__ = ('n', 'x', 'z', 'r')

    def __init__(self, n: int):
        self.n = n
        self.x = np.zeros((2 * n, n), dtype=bool)
        self.z = np.zeros((2 * n, n), dtype=bool)
        self.r = np.zeros(2 * n, dtype=bool)
        for i in range(n):
            self.x[i, i] = True      # destabilizer row i = X_i
            self.z[n + i, i] = True   # stabilizer row i = Z_i

    def h(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()

    def s(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def sdg(self, q: int):
        self.r ^= self.x[:, q] & ~self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def x_gate(self, q: int):
        self.r ^= self.z[:, q]

    def y_gate(self, q: int):
        self.r ^= self.x[:, q] ^ self.z[:, q]

    def z_gate(self, q: int):
        self.r ^= self.x[:, q]

    def sx(self, q: int):
        self.h(q); self.s(q); self.h(q)

    def cx(self, a: int, b: int):
        self.r ^= self.x[:, a] & self.z[:, b] & ~(self.x[:, b] ^ self.z[:, a])
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    def cz(self, a: int, b: int):
        self.h(b); self.cx(a, b); self.h(b)

    def swap(self, a: int, b: int):
        self.x[:, a], self.x[:, b] = self.x[:, b].copy(), self.x[:, a].copy()
        self.z[:, a], self.z[:, b] = self.z[:, b].copy(), self.z[:, a].copy()

    def _rowmult(self, target: int, source: int):
        """Multiply row target by row source (Pauli product with phase tracking)."""
        self.r[target] = _rowmult_phase(
            self.x[target], self.z[target], self.x[source], self.z[source],
            self.r[target], self.r[source], self.n)
        self.x[target] ^= self.x[source]
        self.z[target] ^= self.z[source]

    def measure(self, q: int, rng: np.random.Generator) -> int:
        """CHP measurement: find anticommuting stabilizer, random/deterministic outcome."""
        n = self.n
        p = None
        for i in range(n, 2 * n):
            if self.x[i, q]:
                p = i; break

        if p is not None:
            # Random outcome
            for i in range(2 * n):
                if i != p and self.x[i, q]:
                    self._rowmult(i, p)
            self.x[p - n] = self.x[p].copy()
            self.z[p - n] = self.z[p].copy()
            self.r[p - n] = self.r[p]
            self.x[p] = False; self.z[p] = False
            self.z[p, q] = True
            outcome = int(rng.integers(2))
            self.r[p] = bool(outcome)
            return outcome
        else:
            # Deterministic outcome from destabilizers
            r_acc = False
            x_acc, z_acc = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            for i in range(n):
                if self.x[i, q]:
                    r_acc = _rowmult_phase(x_acc, z_acc, self.x[n + i], self.z[n + i], r_acc, self.r[n + i], n)
                    x_acc ^= self.x[n + i]; z_acc ^= self.z[n + i]
            return int(r_acc)

    def reset(self, q: int, rng: np.random.Generator):
        if self.measure(q, rng) == 1:
            self.x_gate(q)

    def to_statevector(self) -> np.ndarray:
        """Reconstruct statevector from tableau via Gaussian elimination. n <= 25 only."""
        n = self.n
        if n > 25:
            return np.zeros(0, dtype=complex)

        # Copy stabilizer generators for row reduction
        sx, sz, sr = self.x[n:].copy(), self.z[n:].copy(), self.r[n:].copy()

        # Gaussian elimination: find X pivot columns
        row_perm = list(range(n))
        n_pivots = 0
        for col in range(n):
            found = None
            for ri in range(n_pivots, n):
                if sx[row_perm[ri], col]:
                    found = ri; break
            if found is None:
                continue
            row_perm[n_pivots], row_perm[found] = row_perm[found], row_perm[n_pivots]
            pivot = row_perm[n_pivots]
            for ri in range(n):
                row = row_perm[ri]
                if row != pivot and sx[row, col]:
                    sr[row] = _rowmult_phase(sx[row], sz[row], sx[pivot], sz[pivot], sr[row], sr[pivot], n)
                    sx[row] ^= sx[pivot]; sz[row] ^= sz[pivot]
            n_pivots += 1

        # Z-only rows determine the seed via GF(2) linear system: Z·k = r
        seed = 0
        n_z = n - n_pivots
        if n_z > 0:
            z_indices = [row_perm[ri] for ri in range(n_pivots, n)]
            aug = np.zeros((n_z, n + 1), dtype=bool)
            for i, row in enumerate(z_indices):
                aug[i, :n] = sz[row]
                aug[i, n] = sr[row]
            cur, z_pivots = 0, []
            for col in range(n):
                piv = None
                for row in range(cur, n_z):
                    if aug[row, col]:
                        piv = row; break
                if piv is None:
                    continue
                aug[[cur, piv]] = aug[[piv, cur]]
                for row in range(n_z):
                    if row != cur and aug[row, col]:
                        aug[row] ^= aug[cur]
                z_pivots.append(col)
                cur += 1
            for i, col in enumerate(z_pivots):
                if aug[i, n]:
                    seed |= (1 << (n - 1 - col))

        # Build state: start from seed, project with X-containing generators
        dim = 2 ** n
        all_bits = dim - 1
        state = np.zeros(dim, dtype=complex)
        state[seed] = 1.0
        # NOTE: 1-pivot pure-X short-circuit: state has only 2 non-zero elements
        if n_pivots == 1:
            row = row_perm[0]
            z_mask = sum((1 << (n - 1 - q)) for q in range(n) if sz[row, q])
            if z_mask == 0:
                x_mask = sum((1 << (n - 1 - q)) for q in range(n) if sx[row, q])
                sign = -1.0 if sr[row] else 1.0
                inv_sqrt2 = 1.0 / np.sqrt(2)
                state[seed] = inv_sqrt2
                state[seed ^ x_mask] = sign * inv_sqrt2
                return state
        indices = None  # lazy allocation
        for ri in range(n_pivots):
            row = row_perm[ri]
            sign = -1.0 if sr[row] else 1.0
            # Vectorized Pauli application: P|b⟩ = phase(b) × |b ⊕ x_mask⟩
            x_mask = sum((1 << (n - 1 - q)) for q in range(n) if sx[row, q])
            z_mask = sum((1 << (n - 1 - q)) for q in range(n) if sz[row, q])
            if z_mask == 0 and x_mask == all_bits:
                p_state = state[::-1]
            elif z_mask == 0:
                if indices is None: indices = np.arange(dim, dtype=np.int64)
                p_state = state[indices ^ x_mask]
            else:
                if indices is None: indices = np.arange(dim, dtype=np.int64)
                b_orig = indices ^ x_mask
                n_y = bin(x_mask & z_mask).count('1')
                v = (b_orig & z_mask).astype(np.int32)
                v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
                phase = (1j ** n_y) * (1 - 2 * (v & 1)).astype(complex)
                p_state = phase * state[b_orig]
            state = state + sign * p_state
            norm = np.linalg.norm(state)
            if norm > 1e-15:
                state /= norm
        return state


def simulate_stabilizer(circuit: Circuit, seed: int | None = None) -> tuple[np.ndarray, dict[int, int]]:
    n = circuit.n_qubits
    rng = np.random.default_rng(seed)
    classical = {i: 0 for i in range(circuit.n_classical)}
    tab = StabilizerState(n)

    _dispatch = {
        Gate.H: tab.h, Gate.S: tab.s, Gate.SDG: tab.sdg,
        Gate.X: tab.x_gate, Gate.Y: tab.y_gate, Gate.Z: tab.z_gate, Gate.SX: tab.sx,
    }

    for op in circuit.ops:
        if op.condition is not None and classical.get(op.condition[0]) != op.condition[1]:
            continue
        if op.gate == Gate.MEASURE:
            outcome = tab.measure(op.qubits[0], rng)
            if op.classical_bit is not None:
                classical[op.classical_bit] = outcome
        elif op.gate == Gate.RESET:
            tab.reset(op.qubits[0], rng)
        elif op.gate in _dispatch:
            _dispatch[op.gate](op.qubits[0])
        elif op.gate == Gate.CX:
            tab.cx(op.qubits[0], op.qubits[1])
        elif op.gate == Gate.CZ:
            tab.cz(op.qubits[0], op.qubits[1])
        elif op.gate == Gate.SWAP:
            tab.swap(op.qubits[0], op.qubits[1])

    sv = tab.to_statevector()
    return sv, classical
