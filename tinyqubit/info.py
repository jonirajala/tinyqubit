from __future__ import annotations
import numpy as np


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


def partial_trace(statevector: np.ndarray, keep: list[int]) -> np.ndarray:
    n = int(np.log2(len(statevector)))
    keep_sorted = sorted(keep)
    trace_out = [i for i in range(n) if i not in keep_sorted]
    psi = statevector.reshape([2] * n)
    psi = np.transpose(psi, keep_sorted + trace_out)
    d_keep = 2 ** len(keep_sorted)
    psi = psi.reshape(d_keep, -1)
    return psi @ psi.conj().T


def _von_neumann_entropy(rho: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def entanglement_entropy(statevector: np.ndarray, partition: list[int]) -> float:
    return _von_neumann_entropy(partial_trace(statevector, partition))


def concurrence(statevector: np.ndarray) -> float:
    if len(statevector) != 4:
        raise ValueError("concurrence requires a 2-qubit state")
    a, b, c, d = statevector
    return float(2 * np.abs(a * d - b * c))


def mutual_information(statevector: np.ndarray, partition_a: list[int], partition_b: list[int]) -> float:
    sa = entanglement_entropy(statevector, partition_a)
    sb = entanglement_entropy(statevector, partition_b)
    sab = entanglement_entropy(statevector, partition_a + partition_b)
    return sa + sb - sab
