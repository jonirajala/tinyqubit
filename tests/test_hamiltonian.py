import numpy as np
import tinyqubit as tq
from tinyqubit.hamiltonian import maxcut_hamiltonian


def test_maxcut_single_edge():
    # H = 0.5(I - Z0 Z1): eigenvalue 1 when qubits differ, 0 when same
    H = maxcut_hamiltonian([(0, 1)])
    assert len(H.terms) == 2
    # |01⟩ → cut = 1
    c = tq.Circuit(2)
    c.x(1)
    assert abs(tq.expectation(c, H) - 1.0) < 1e-10
    # |00⟩ → cut = 0
    c2 = tq.Circuit(2)
    assert abs(tq.expectation(c2, H)) < 1e-10


def test_maxcut_triangle():
    # Triangle graph: 3 edges, max cut = 2 (can cut at most 2 of 3 edges)
    edges = [(0, 1), (1, 2), (0, 2)]
    H = maxcut_hamiltonian(edges)
    assert len(H.terms) == 6  # 2 terms per edge
    # |010⟩ → cuts edges (0,1) and (1,2), not (0,2) → cut = 2
    c = tq.Circuit(3)
    c.x(1)
    assert abs(tq.expectation(c, H) - 2.0) < 1e-10


def test_maxcut_weighted():
    # Weighted edge: H = w * 0.5(I - Z0 Z1)
    H = maxcut_hamiltonian([(0, 1, 3.0)])
    # |01⟩ → cut = 3.0
    c = tq.Circuit(2)
    c.x(1)
    assert abs(tq.expectation(c, H) - 3.0) < 1e-10
    # |00⟩ → cut = 0
    c2 = tq.Circuit(2)
    assert abs(tq.expectation(c2, H)) < 1e-10


def test_maxcut_with_qaoa():
    # Simple QAOA-style circuit: build circuit, check expectation is in valid range
    edges = [(0, 1), (1, 2)]
    H = maxcut_hamiltonian(edges)
    c = tq.Circuit(3)
    # Layer of Hadamards (uniform superposition)
    for q in range(3):
        c.h(q)
    # Single QAOA layer with fixed angles
    for i, j in edges:
        c.cx(i, j)
        c.rz(j, 0.5)
        c.cx(i, j)
    for q in range(3):
        c.rx(q, 0.7)
    val = tq.expectation(c, H)
    # Max cut for path graph (0-1-2) is 2, min is 0
    assert 0.0 <= val <= 2.0 + 1e-10
