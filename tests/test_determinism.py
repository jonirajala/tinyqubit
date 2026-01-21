"""
Golden tests for determinism.

Tests:
    - Same input → exact same output
    - Cross-platform consistency
    - Version-to-version stability

Uses tests/golden/ for expected outputs.
"""
import json
from pathlib import Path
import pytest
from math import pi

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.target import Target
from tinyqubit.compile import transpile

GOLDEN_DIR = Path(__file__).parent / "golden"


# =============================================================================
# Serialization helpers
# =============================================================================

def load_circuit(data: dict) -> Circuit:
    """Load a Circuit from JSON dict."""
    c = Circuit(data["n_qubits"])
    for op_data in data["ops"]:
        gate = Gate[op_data["gate"]]
        qubits = tuple(op_data["qubits"])
        params = tuple(op_data.get("params", []))
        c.ops.append(Operation(gate, qubits, params))
    return c


def load_target(data: dict) -> Target:
    """Load a Target from JSON dict."""
    return Target(
        n_qubits=data["n_qubits"],
        edges=frozenset(tuple(e) for e in data["edges"]),
        basis_gates=frozenset(Gate[g] for g in data["basis_gates"]),
        name=data["name"]
    )


def serialize_circuit(circuit: Circuit) -> dict:
    """Serialize a Circuit to JSON-compatible dict."""
    return {
        "n_qubits": circuit.n_qubits,
        "ops": [
            {"gate": op.gate.name, "qubits": list(op.qubits), "params": list(op.params)}
            if op.params else {"gate": op.gate.name, "qubits": list(op.qubits)}
            for op in circuit.ops
        ]
    }


def load_golden_data():
    """Load all golden test data."""
    with open(GOLDEN_DIR / "circuits.json") as f:
        circuits = json.load(f)
    with open(GOLDEN_DIR / "targets.json") as f:
        targets = json.load(f)
    try:
        with open(GOLDEN_DIR / "expected_outputs.json") as f:
            expected = json.load(f)
    except FileNotFoundError:
        expected = {}
    return circuits, targets, expected


# =============================================================================
# Test combinations
# =============================================================================

# Which (circuit, target) pairs to test - target must have enough qubits
GOLDEN_COMBOS = [
    # Bell state on various targets
    ("bell_2", "line_5_ibm"),
    ("bell_2", "grid_4_ibm"),
    ("bell_2", "line_5_rigetti"),
    ("bell_2", "all_to_all_4"),
    # GHZ-3 on various targets
    ("ghz_3", "line_5_ibm"),
    ("ghz_3", "grid_4_ibm"),
    ("ghz_3", "line_5_rigetti"),
    # GHZ-5 (needs routing on line)
    ("ghz_5", "line_5_ibm"),
    ("ghz_5", "line_5_rigetti"),
    # QFT-4 (needs decomposition + routing)
    ("qft_4", "grid_4_ibm"),
    ("qft_4", "all_to_all_4"),
    # Variational circuit
    ("variational_4", "grid_4_ibm"),
    ("variational_4", "line_5_ibm"),
    # Circuits needing SWAPs
    ("swap_needed_3", "line_5_ibm"),
    ("multi_swap_5", "line_5_ibm"),
]


# =============================================================================
# Golden tests
# =============================================================================

@pytest.mark.parametrize("circuit_name,target_name", GOLDEN_COMBOS)
def test_golden_output_matches(circuit_name, target_name):
    """Test that transpile output matches golden expected output."""
    circuits, targets, expected = load_golden_data()

    circuit = load_circuit(circuits[circuit_name])
    target = load_target(targets[target_name])

    result = transpile(circuit, target)
    actual = serialize_circuit(result)

    key = f"{circuit_name}@{target_name}"
    if key not in expected:
        pytest.skip(f"No golden output for {key}. Run scripts/update_golden.py to generate.")

    assert actual == expected[key], (
        f"Output changed for {key}!\n"
        f"Expected {len(expected[key]['ops'])} ops, got {len(actual['ops'])} ops.\n"
        f"Run `python scripts/update_golden.py` to update if this change is intentional."
    )


def test_determinism_multiple_runs():
    """Same input produces exact same output across multiple runs."""
    circuits, targets, _ = load_golden_data()

    circuit = load_circuit(circuits["qft_4"])
    target = load_target(targets["grid_4_ibm"])

    results = [serialize_circuit(transpile(circuit, target)) for _ in range(5)]

    for i, r in enumerate(results[1:], 1):
        assert r == results[0], f"Run {i} differs from run 0"


def test_determinism_fresh_circuit():
    """Fresh circuit instances produce same output."""
    circuits, targets, _ = load_golden_data()

    target = load_target(targets["line_5_ibm"])

    # Create fresh circuits each time
    results = []
    for _ in range(3):
        c = load_circuit(circuits["ghz_5"])
        results.append(serialize_circuit(transpile(c, target)))

    for i, r in enumerate(results[1:], 1):
        assert r == results[0], f"Fresh circuit {i} differs from circuit 0"
