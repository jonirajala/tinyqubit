# TinyQubit

> Minimal, deterministic quantum circuit compiler with explainable routing.

## Why?

Existing quantum SDKs have problems:

- **API Churn** — Qiskit breaks everything between versions
- **Non-Deterministic** — Same circuit, different output every run
- **Black-Box** — No explanation of routing decisions
- **Bloat** — 1+ second imports, heavy dependencies
- **Vendor Lock-in** — Best features tied to one provider

TinyQubit: tiny, no dependencies, deterministic, explainable.

## Usage

```python
from tinyqubit import Circuit, to_openqasm2

c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
print(to_openqasm2(c))
```

## Gates

16 primitives: `X` `Y` `Z` `H` `S` `T` `SDG` `TDG` `RX` `RY` `RZ` `CX` `CZ` `CP` `SWAP` `MEASURE`

## Dependencies

| Dependency | Used for | Plan |
|------------|----------|------|
| `numpy` | Simulator (statevector math) | Replace with pure Python or make optional |

**Dev/test:**
- `pytest` — test framework
- `hypothesis` — property-based tests

**Optional** (for hardware submission):
- `qiskit` — `to_qiskit()` export
- `qiskit-ibm-runtime` — IBM hardware submission
- `amazon-braket-sdk` — AWS Braket submission

Goal: Core compiler (circuit → routed → decomposed → QASM) runs with **zero dependencies**.

## Tests & Commands

```bash
# Run examples
python examples/circuit.py       # Basic circuit + transpile
python examples/grover.py        # Grover's algorithm
python examples/submit_to_ibm.py # IBM hardware submission (requires qiskit-ibm-runtime)

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_determinism.py -v   # Golden/determinism tests
python -m pytest tests/test_properties.py -v    # Property-based tests (hypothesis)
python -m pytest tests/test_metrics.py -v       # Routing/optimization metrics
python -m pytest tests/test_performance.py -v   # Import time, transpile speed

# Run benchmarks (vs Qiskit)
python benchmarks/run_all.py

# Update golden test baselines (after intentional changes)
python scripts/update_golden.py
```
