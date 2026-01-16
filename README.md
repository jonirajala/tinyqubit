# TinyQubit

> A tiny, stable, deterministic quantum circuit compiler with explainable routing — export anywhere.

## What is TinyQubit?

TinyQubit is a minimal quantum computing library that prioritizes **stability**, **predictability**, and **explainability** over feature bloat. Think of it as "LLVM for quantum" — a stable intermediate representation with deterministic compilation that exports to any target.

We want to minimize locs and not include any dependency library.

## The Problem

Existing quantum SDKs have serious pain points:

- **API Churn**: Qiskit 0.x → 1.0 → 2.0 broke everything. All educational content instantly obsoleted. Gate names changed, `execute()` removed, entire modules deleted.
- **Non-Deterministic Transpilation**: Same circuit → different output every run. Qiskit uses random seeds by default, and results vary by CPU count. Debugging becomes impossible.
- **Black-Box Compilation**: No explanation of routing decisions. When your circuit fails, good luck figuring out why.
- **Bloat**: 1+ second import times. Heavy dependencies pulled in at top-level. Backend loading can take 5+ minutes.
- **Vendor Lock-in**: Qiskit's best features only work with IBM. Cirq has no real hardware access at all.

The quantum community accepts this as "just how quantum is" — but most of this pain is **accidental complexity** from poor software engineering, not inherent quantum difficulty.

## Design Principles

### Boringly Stable
- One stable public API, everything else internal
- Semantic versioning that we actually follow
- "Write once, runs for years"
- **Why**: Qiskit's API churn obsoleted every tutorial, course, and YouTube video. We won't do that.

### Predictable
- Deterministic compilation (no hidden randomness unless user passes a seed)
- Compilation output includes an explanation report
- Fewer passes, but each pass is understandable
- **Why**: Qiskit's stochastic transpiler makes reproducibility impossible. Same input should always produce same output.

### Portable
- Vendor-neutral design
- Export targets: OpenQASM 2/3, Qiskit circuit, Cirq circuit
- **Why**: OpenQASM is the universal interchange format. Export once, run on IBM, Braket, IonQ, Rigetti, or any future hardware.

## Architecture

TinyQubit consists of 4 core modules:

### 1. `ir.py` — Circuit IR
- `Qubit`, `Bit` primitives
- `Gate` definitions (enum or small classes)
- `Operation(gate, qubits, params)`
- `Circuit(n_qubits)` with `append(op)`
- Measurement support

**Supported Gates:**
- Single-qubit: X, Y, Z, H, S, T
- Rotation: RX, RY, RZ
- Two-qubit: CX (CNOT), CZ
- Measurement: MEASURE

### 2. `passes/` — Deterministic Transpiler Passes
- Decompose to basis gates
- Layout (map logical → physical qubits)
- Route (insert SWAPs for connectivity)
- Optimize (local cancellation/merge)
- Schedule (optional ASAP/ALAP)

### 3. `backends/` — Simulation & Adapters
- Statevector simulator for correctness testing
- Shot-based simulator for sampling
- Adapters:
  - `to_qiskit(circuit)`
  - `to_openqasm(circuit)`

### 4. `reporting/` — Explainability
Every compilation returns:
- Compiled circuit
- Metrics: depth, 2Q gate count, SWAP count
- Routing path explanation
- Pass-by-pass diffs (optional)

## The Killer Feature: Explainable Compilation

Quantum debugging is fundamentally hard — you can't step through code (observation collapses state), you can't inspect variables (no-cloning theorem), and you can't add assertions mid-circuit. When something goes wrong, you need to understand what the compiler did.

Every `compile()` call returns a `CompileReport` containing:
- Chosen initial layout and why (score)
- Each routing SWAP insertion with explanation of which interaction forced it
- Before/after metrics for each pass
- Pass-by-pass diffs so you can trace exactly what changed

This directly attacks the "transpiler is magic" problem. No more guessing why your circuit was mangled.

## Why TinyQubit?

| Pain Point | The Problem | TinyQubit Solution |
|------------|-------------|-------------------|
| API churn | Qiskit 0.x→1.0→2.0 broke all code and tutorials | Tiny, stable API with real semver |
| Non-determinism | Same input → different output, varies by CPU count | Deterministic by default |
| Black-box transpiler | No explanation of routing decisions | CompileReport explains every SWAP |
| Bloat | 1+ sec imports, 5+ min backend loading | 4 modules, minimal dependencies |
| Vendor lock-in | Qiskit=IBM only, Cirq=no hardware | OpenQASM export works everywhere |
| Debugging difficulty | Can't step through quantum state | Pass-by-pass diffs and metrics |
| Educational rot | Tutorials obsolete within months | Stable API = content stays valid |

## What We Don't Do

- Full algorithm library (VQE/QAOA/etc.)
- Cloud provider integration
- Every gate under the sun
- Pulse-level scheduling
- "Best possible" optimization

**Our advantage is simplicity + determinism + explainability.**

We're not trying to beat Qiskit at optimization benchmarks. We're solving the pain points that make quantum development frustrating.

## License

MIT
# tinyqubit
