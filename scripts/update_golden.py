#!/usr/bin/env python
"""
Regenerate golden test expected outputs.

Run this when you intentionally change transpile behavior:
    python scripts/update_golden.py

This will update tests/golden/expected_outputs.json with current outputs.
Review the diff carefully before committing!
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit, Gate, Operation
from tinyqubit.target import Target
from tinyqubit.compile import transpile

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden"


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


# Which (circuit, target) pairs to generate
GOLDEN_COMBOS = [
    ("bell_2", "line_5_ibm"),
    ("bell_2", "grid_4_ibm"),
    ("bell_2", "line_5_rigetti"),
    ("bell_2", "all_to_all_4"),
    ("ghz_3", "line_5_ibm"),
    ("ghz_3", "grid_4_ibm"),
    ("ghz_3", "line_5_rigetti"),
    ("ghz_5", "line_5_ibm"),
    ("ghz_5", "line_5_rigetti"),
    ("qft_4", "grid_4_ibm"),
    ("qft_4", "all_to_all_4"),
    ("variational_4", "grid_4_ibm"),
    ("variational_4", "line_5_ibm"),
    ("swap_needed_3", "line_5_ibm"),
    ("multi_swap_5", "line_5_ibm"),
]


def main():
    print("Loading golden test data...")
    with open(GOLDEN_DIR / "circuits.json") as f:
        circuits = json.load(f)
    with open(GOLDEN_DIR / "targets.json") as f:
        targets = json.load(f)

    print(f"Generating {len(GOLDEN_COMBOS)} golden outputs...")
    expected = {}

    for circuit_name, target_name in GOLDEN_COMBOS:
        key = f"{circuit_name}@{target_name}"
        print(f"  {key}...", end=" ")

        circuit = load_circuit(circuits[circuit_name])
        target = load_target(targets[target_name])

        result = transpile(circuit, target)
        expected[key] = serialize_circuit(result)

        print(f"{len(result.ops)} ops")

    output_path = GOLDEN_DIR / "expected_outputs.json"
    with open(output_path, "w") as f:
        json.dump(expected, f, indent=2)

    print(f"\nWrote {output_path}")
    print("Review the diff carefully before committing!")


if __name__ == "__main__":
    main()
