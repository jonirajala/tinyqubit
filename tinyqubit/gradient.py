"""Parameter-shift and finite-difference gradient computation."""
from __future__ import annotations
import numpy as np
from .ir import Circuit
from .observable import Observable, expectation
from .simulator import simulate, sample


def _shot_expectation(circuit: Circuit, observable: Observable, shots: int, seed: int | None = None) -> float:
    """Estimate ⟨ψ|O|ψ⟩ by measurement sampling with basis rotations."""
    rng = np.random.default_rng(seed)
    result = 0.0
    for coeff, paulis in observable.terms:
        rc = Circuit(circuit.n_qubits, circuit.n_classical)
        rc._initial_state, rc.ops = circuit._initial_state, list(circuit.ops)
        for q, p in paulis.items():
            if p == 'X': rc.h(q)
            elif p == 'Y':
                rc.rz(q, -np.pi / 2)
                rc.h(q)
        state, _ = simulate(rc)
        counts = sample(state, shots, seed=int(rng.integers(2**31)))
        ev = sum((-1) ** (sum(int(bs[q]) for q in paulis) % 2) * cnt for bs, cnt in counts.items())
        result += coeff * ev / shots
    return result


def parameter_shift_gradient(circuit: Circuit, observable: Observable,
                             values: dict[str, float], shots: int | None = None) -> dict[str, float]:
    """Compute gradient via the parameter-shift rule: df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2"""
    shift = np.pi / 2
    exp_fn = expectation if shots is None else lambda c, o: _shot_expectation(c, o, shots)
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**values, param.name: values[param.name] + shift}
        v_minus = {**values, param.name: values[param.name] - shift}
        e_plus = exp_fn(circuit.bind(v_plus), observable)
        e_minus = exp_fn(circuit.bind(v_minus), observable)
        grad[param.name] = (e_plus - e_minus) / 2
    return grad


def finite_difference_gradient(circuit: Circuit, observable: Observable,
                               values: dict[str, float], epsilon: float = 1e-7) -> dict[str, float]:
    """Compute gradient via symmetric finite differences: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / 2ε"""
    grad = {}
    for param in sorted(circuit.parameters, key=lambda p: p.name):
        v_plus = {**values, param.name: values[param.name] + epsilon}
        v_minus = {**values, param.name: values[param.name] - epsilon}
        e_plus = expectation(circuit.bind(v_plus), observable)
        e_minus = expectation(circuit.bind(v_minus), observable)
        grad[param.name] = (e_plus - e_minus) / (2 * epsilon)
    return grad
