"""Tests for Circuit JSON serialization round-trip."""
import json, math
import numpy as np
from tinyqubit import Circuit, Parameter, simulate, states_equal
from tinyqubit.tracker import QubitTracker


def _roundtrip(c):
    return Circuit.from_json(c.to_json())


def test_basic_roundtrip():
    c = Circuit(3)
    c.h(0).cx(0, 1).x(2).z(1).s(0).t(2).sdg(0).tdg(2).sx(1)
    c.cz(0, 1).swap(1, 2).ecr(0, 1)
    c.ccx(0, 1, 2).ccz(0, 1, 2)
    c2 = _roundtrip(c)
    assert c2.n_qubits == c.n_qubits
    assert c2.ops == c.ops


def test_parametric_gates():
    c = Circuit(2)
    c.rx(0, 1.23).ry(1, 0.5).rz(0, math.pi)
    c.cp(0, 1, 0.75).rzz(0, 1, 0.42)
    c2 = _roundtrip(c)
    assert c2.ops == c.ops


def test_parameter_objects():
    theta = Parameter("theta", trainable=True)
    phi = Parameter("phi", trainable=False)
    c = Circuit(2)
    c.rx(0, theta).ry(1, phi)
    c2 = _roundtrip(c)
    p0, p1 = c2.ops[0].params[0], c2.ops[1].params[0]
    assert isinstance(p0, Parameter) and p0.name == "theta" and p0.trainable is True
    assert isinstance(p1, Parameter) and p1.name == "phi" and p1.trainable is False


def test_measure_reset_conditional():
    c = Circuit(2, 2)
    c.h(0).measure(0, 0)
    c.reset(1)
    with c.c_if(0, 1):
        c.x(1)
    c2 = _roundtrip(c)
    assert c2.ops == c.ops
    assert c2.ops[1].classical_bit == 0
    assert c2.ops[3].condition == (0, 1)


def test_initial_state():
    c = Circuit(2)
    c.initialize([0, 1, 0, 0])
    c.h(0)
    c2 = _roundtrip(c)
    assert c2._initial_state is not None
    assert np.allclose(c2._initial_state, c._initial_state)


def test_tracker_metadata():
    c = Circuit(3)
    c.h(0)
    t = QubitTracker(5, [2, 3, 0])
    c._tracker = t
    c2 = _roundtrip(c)
    assert hasattr(c2, '_tracker')
    assert c2._tracker.n_qubits == 5
    assert c2._tracker.initial_layout == [2, 3, 0]
    assert c2._tracker.logical_to_physical == t.logical_to_physical
    assert c2._tracker.physical_to_logical == t.physical_to_logical


def test_statevector_agreement():
    c = Circuit(2)
    c.h(0).cx(0, 1).rz(1, 0.5)
    c2 = _roundtrip(c)
    assert states_equal(simulate(c)[0], simulate(c2)[0])


def test_empty_circuit():
    c = Circuit(1)
    c2 = _roundtrip(c)
    assert c2.n_qubits == 1 and c2.ops == []


def test_json_is_valid():
    c = Circuit(2)
    c.h(0).cx(0, 1)
    data = json.loads(c.to_json())
    assert data["n_qubits"] == 2
    assert len(data["ops"]) == 2
