"""Trainability diagnostics: barren plateau detection and expressibility."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .observable import Observable
from .gradient import parameter_shift_gradient
from .simulator import simulate
from .info import state_fidelity


def gradient_variance(circuit: Circuit, observable: Observable, n_samples: int = 100,
                      seed: int | None = None) -> dict[str, float]:
    """Per-parameter gradient variance over random parameter sets. Low variance → barren plateau."""
    rng = np.random.default_rng(seed)
    names = sorted(p.name for p in circuit.parameters)
    grads = np.array([[parameter_shift_gradient(circuit, observable,
                       dict(zip(names, rng.uniform(0, 2 * np.pi, len(names)))))[n]
                       for n in names] for _ in range(n_samples)])
    return dict(zip(names, grads.var(axis=0)))


def expressibility(circuit: Circuit, n_samples: int = 1000, n_bins: int = 75,
                   seed: int | None = None) -> float:
    """KL divergence of fidelity distribution from Haar-random. Lower → more expressible."""
    rng = np.random.default_rng(seed)
    names = sorted(p.name for p in circuit.parameters)
    n_p, N = len(names), 2 ** circuit.n_qubits
    fids = np.empty(n_samples)
    for i in range(n_samples):
        s1, _ = simulate(circuit.bind(dict(zip(names, rng.uniform(0, 2 * np.pi, n_p)))))
        s2, _ = simulate(circuit.bind(dict(zip(names, rng.uniform(0, 2 * np.pi, n_p)))))
        fids[i] = state_fidelity(s1, s2)
    hist, edges = np.histogram(fids, bins=n_bins, range=(0, 1), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    haar = (N - 1) * (1 - centers) ** (N - 2)
    mask = (hist > 0) & (haar > 0)
    return float(np.sum(hist[mask] * np.log(hist[mask] / haar[mask])) * (edges[1] - edges[0]))
