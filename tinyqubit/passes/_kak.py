"""KAK (Cartan) decomposition of 2-qubit unitaries.

Decomposes any U ∈ U(4) as: U = g * kron(A1, A0) · Ud · kron(B1, B0)
where Ud = exp(i(xx·XX + yy·YY + zz·ZZ)), with pi/4 >= xx >= yy >= |zz| >= 0.

Uses SVD-based bidiagonalization (Cirq approach) for robust handling of
degenerate cases. No scipy dependency — numpy only.
"""
from __future__ import annotations
import numpy as np
from math import pi, sqrt

_I2 = np.eye(2, dtype=complex)

# Magic basis: transforms computational ↔ Bell basis
# In this basis, local unitaries become SO(4), entangling part becomes diagonal
_M = np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j],
], dtype=complex) / sqrt(2)
_Md = _M.conj().T

# Converts diagonal phases to (global_phase, xx, yy, zz)
# Inverse of: diag phases are (xx+yy-zz, -xx+yy+zz, xx-yy+zz, -xx-yy-zz)
_KAK_GAMMA = np.array([
    [1, 1, 1, 1],
    [1, 1, -1, -1],
    [-1, 1, -1, 1],
    [1, -1, -1, 1],
], dtype=float) * 0.25


def _to_su4(U: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize U to SU(4), return (SU4_matrix, global_phase)."""
    d = np.linalg.det(U)
    phase = np.angle(d) / 4
    return U * np.exp(-1j * phase), phase


def _nearest_su2(M: np.ndarray) -> np.ndarray:
    """Project 2x2 matrix to SU(2) via polar decomposition."""
    U, _, Vh = np.linalg.svd(M)
    S = U @ Vh
    d = np.linalg.det(S)
    return S * np.exp(-1j * np.angle(d) / 2)


def _bidiag_real_pair(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneously bidiagonalize real matrices A, B.

    Find real orthogonal L, R such that L @ (A+iB) @ R is diagonal.
    A and B come from Re(M_magic) and Im(M_magic) where M_magic is unitary in magic basis.

    Algorithm:
    1. SVD of A gives A = Ua @ diag(Sa) @ Va^T
    2. For degenerate singular values of A, apply a single orthogonal Q (from eigh)
       to diagonalize Bp = Ua^T @ B @ Va within each degenerate block.
       Using the same Q on both sides preserves diag(Sa).

    Returns (L, diag_phases, R) where L, R are real orthogonal and diag_phases
    are the diagonal entries of L @ (A + iB) @ R.
    """
    Ua, Sa, VaT = np.linalg.svd(A)
    if np.linalg.det(Ua) < 0:
        Ua[:, -1] *= -1
        Sa[-1] *= -1
    if np.linalg.det(VaT) < 0:
        VaT[-1, :] *= -1
        Sa[-1] *= -1

    Va = VaT.T
    Bp = Ua.T @ B @ Va

    # Within each degenerate block of Sa, diagonalize Bp symmetrically.
    # Use Q^T @ Bp @ Q = diag  (since Bp block is symmetric for unitary M_magic).
    # This preserves diag(Sa) since Q^T @ (s*I) @ Q = s*I.
    tol = 1e-9
    n = len(Sa)
    Q = np.eye(n)

    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(abs(Sa[i]) - abs(Sa[j])) < tol:
            j += 1
        if j - i > 1:
            block = Bp[i:j, i:j]
            # Symmetrize for numerical stability
            block = (block + block.T) / 2
            eigvals, eigvecs = np.linalg.eigh(block)
            Q[i:j, i:j] = eigvecs
        i = j

    # Apply Q: L = Q^T @ Ua^T, R = Va @ Q
    L = Q.T @ Ua.T
    R = Va @ Q

    M_complex = A + 1j * B
    D = L @ M_complex @ R
    diag_phases = np.diag(D).copy()

    # Ensure special orthogonal (det = +1)
    if np.linalg.det(L) < 0:
        L[0, :] *= -1
        diag_phases[0] = -diag_phases[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
        diag_phases[0] = -diag_phases[0]

    return L, diag_phases, R


def _extract_su2_pair(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (U0, U1) from K ≈ kron(U1, U0) up to global phase."""
    blocks = [K[0:2, 0:2], K[0:2, 2:4], K[2:4, 0:2], K[2:4, 2:4]]
    norms = [np.linalg.norm(b) for b in blocks]
    best = blocks[int(np.argmax(norms))]
    U0 = _nearest_su2(best / np.linalg.norm(best))
    right = np.kron(_I2, U0.conj().T)
    prod = K @ right
    U1 = _nearest_su2(np.array([
        [prod[0, 0], prod[0, 2]],
        [prod[2, 0], prod[2, 2]],
    ]))
    return U0, U1


def kak_decompose(U: np.ndarray, tol: float = 1e-9):
    """KAK decomposition of a 4x4 unitary.

    Returns: (A0, A1, xx, yy, zz, B0, B1) where:
        - A0, A1, B0, B1 are 2x2 SU(2)
        - xx, yy, zz are raw interaction angles (not canonicalized)
        - U ≈ phase * kron(A1,A0) @ Ud(xx,yy,zz) @ kron(B1,B0)
          where Ud = exp(i*(xx*XX + yy*YY + zz*ZZ))
    """
    Usu4, global_phase = _to_su4(U)

    # Transform to magic basis
    M_magic = _Md @ Usu4 @ _M

    # Bidiagonalize: find orthogonal L, R such that L @ M_magic @ R is diagonal
    A = M_magic.real.copy()
    B = M_magic.imag.copy()
    L, diag_d, R = _bidiag_real_pair(A, B)

    # Extract interaction angles from diagonal phases
    phases = np.angle(diag_d)
    raw = _KAK_GAMMA @ phases  # [global_phase_correction, xx, yy, zz]
    xx, yy, zz = float(raw[1]), float(raw[2]), float(raw[3])

    # Local unitaries
    A_kron = _M @ L.T.astype(complex) @ _Md  # kron(A1, A0)
    B_kron = _M @ R.T.astype(complex) @ _Md  # kron(B1, B0)

    A0, A1 = _extract_su2_pair(A_kron)
    B0, B1 = _extract_su2_pair(B_kron)

    return A0, A1, xx, yy, zz, B0, B1


def _canonicalize_weyl(xx: float, yy: float, zz: float) -> tuple[float, float, float]:
    """Map interaction angles to Weyl chamber: pi/4 >= xx >= yy >= |zz| >= 0."""
    # Normalize to [-pi/2, pi/2]
    def norm(v):
        v = v % pi
        if v > pi / 2:
            v -= pi
        return v

    xx, yy, zz = norm(xx), norm(yy), norm(zz)

    # Sort by absolute value descending
    vals = sorted([abs(xx), abs(yy), abs(zz)], reverse=True)
    xx, yy, zz = vals[0], vals[1], vals[2]

    # Fold into [0, pi/4]
    if xx > pi / 4:
        xx = pi / 2 - xx
    if yy > pi / 4:
        yy = pi / 2 - yy
    if zz > pi / 4:
        zz = pi / 2 - zz

    return xx, yy, zz


def cx_count(xx: float, yy: float, zz: float, tol: float = 1e-6) -> int:
    """Minimum CX gates needed for given interaction angles (accepts raw or canonical)."""
    xx, yy, zz = _canonicalize_weyl(xx, yy, zz)
    if abs(xx) < tol and abs(yy) < tol and abs(zz) < tol:
        return 0
    if abs(xx - pi / 4) < tol and abs(yy) < tol and abs(zz) < tol:
        return 1
    if abs(zz) < tol:
        return 2
    return 3
