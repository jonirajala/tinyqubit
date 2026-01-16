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
from tinyqubit.ir import Circuit

c = Circuit(2).h(0).cx(0, 1).measure(0).measure(1)
print(c.to_openqasm())
```

## Gates

12 primitives: `X` `Y` `Z` `H` `S` `T` `RX` `RY` `RZ` `CX` `CZ` `MEASURE`
