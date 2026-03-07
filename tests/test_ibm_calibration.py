"""Tests for IBM calibration data parsing (_parse_edge_errors)."""
from tinyqubit.hardware.ibm_native import _parse_edge_errors


def _gate(qubits, error):
    return {"qubits": qubits, "gate": "ecr", "parameters": [{"name": "gate_error", "value": error}, {"name": "gate_length", "value": 660}]}


def test_parse_basic():
    props = {"gates": [_gate([0, 1], 0.01), _gate([2, 3], 0.02)]}
    edges = frozenset([(0, 1), (2, 3)])
    result = _parse_edge_errors(props, edges)
    assert result == {(0, 1): 0.01, (2, 3): 0.02}


def test_parse_both_directions():
    props = {"gates": [_gate([1, 0], 0.01), _gate([0, 1], 0.03)]}
    edges = frozenset([(0, 1), (1, 0)])
    result = _parse_edge_errors(props, edges)
    assert result[(0, 1)] == 0.03  # max of both directions


def test_parse_missing_edges_filled():
    props = {"gates": [_gate([0, 1], 0.01)]}
    edges = frozenset([(0, 1), (2, 3)])
    result = _parse_edge_errors(props, edges)
    assert result[(0, 1)] == 0.01
    assert result[(2, 3)] == 0.01  # filled with max known


def test_parse_no_2q_gates():
    props = {"gates": [{"qubits": [0], "gate": "sx", "parameters": [{"name": "gate_error", "value": 0.001}]}]}
    edges = frozenset([(0, 1)])
    assert _parse_edge_errors(props, edges) == {}


def test_parse_mixed_gate_types():
    props = {"gates": [
        {"qubits": [0], "gate": "sx", "parameters": [{"name": "gate_error", "value": 0.001}]},
        _gate([0, 1], 0.05),
        {"qubits": [2], "gate": "rz", "parameters": [{"name": "gate_error", "value": 0.0}]},
    ]}
    edges = frozenset([(0, 1)])
    result = _parse_edge_errors(props, edges)
    assert result == {(0, 1): 0.05}
