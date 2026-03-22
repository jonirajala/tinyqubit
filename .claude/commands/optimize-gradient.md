# Autonomous Gradient Computation Optimization

You are an autonomous researcher optimizing TinyQubit's gradient computation (`adjoint_gradient`, `backprop_gradient`, and supporting infrastructure) for speed while preserving correctness. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## Setup

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `grad-mar22`). The branch `optimize/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `CLAUDE.md` — coding standards. All changes MUST follow these conventions.
   - `tinyqubit/qml/optim.py` — **primary optimization target**. `adjoint_gradient`, `_adjoint_backward`, `_build_adjoint_info`, `backprop_gradient`, optimizer step logic.
   - `tinyqubit/simulator/statevector.py` — gate application functions called by the backward pass.
   - `tinyqubit/simulator/simulator.py` — `simulate()` dispatch, `expectation()`.
   - `tinyqubit/measurement/observable.py` — `expectation()` and Pauli observable helpers.
   - `tinyqubit/ir.py` — Circuit, Gate, Operation, Parameter definitions.
   - `benchmarks/gradient_benchmark.py` — **the benchmark you run**. Read-only.
   - `benchmarks/gradient_baseline.json` — frozen golden outputs + baseline times. Read-only.
4. **Verify baseline exists**: Check that `benchmarks/gradient_baseline.json` exists. If not, run `./venv/bin/python benchmarks/gradient_benchmark.py --save`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go.**

## Rules

### What you CAN modify
- `tinyqubit/qml/optim.py` — primary target. Adjoint backward pass, gradient extraction, parameter bookkeeping, cache management, optimizer logic.
- `tinyqubit/simulator/statevector.py` — if gate application can be made faster for the backward pass (e.g., batching adjoint gates).
- `tinyqubit/simulator/simulator.py` — if dispatch or expectation can be optimized.
- `tinyqubit/measurement/observable.py` — if observable application (λ = O|ψ⟩) can be faster.
- `tinyqubit/ir.py` — if circuit structure changes help (e.g., faster parameter lookup).

### What you CANNOT modify
- `benchmarks/gradient_benchmark.py` — read-only.
- `benchmarks/gradient_baseline.json` — read-only.
- `tests/` — read-only. Tests must keep passing.
- No new dependencies — numpy only.

### Coding standards (from CLAUDE.md)
Same as the simulator optimization skill — minimal LOC, deterministic, explainable, no bloat.

### The goal
Get the **lowest total gradient computation time** across all 14 benchmark cases while keeping **all 14 correctness checks passing**. Correctness means: gradient matches finite-difference reference within tolerance.

### Key metric
Sum of median times across all benchmarks. Extract:
```
grep "Time(ms)" -A 20 run.log | grep -E "^\s+" | awk '{sum += $4} END {print sum}'
```

## Output format

Same as simulator benchmark — correctness + performance table + summary.

## Logging results

Log to `results.tsv` (tab-separated):
```
commit	total_ms	correctness	status	description
```

## The experiment loop

**LOOP FOREVER:**

1. **Examine the code** — identify optimization opportunities. Think about:
   - Can the backward pass apply gates to state AND lambda simultaneously (fused matmul)?
   - Can we avoid allocating temporary arrays in the backward loop?
   - Can the diagonal phase accumulation be more efficient?
   - Can CX block detection in backward be unified with forward pass infrastructure?
   - Can `_build_adjoint_info` pre-compute more to reduce per-step work?
   - Can the lambda (O|ψ⟩) computation be faster for Pauli-sum observables?
   - Can `circuit.bind()` be avoided by direct parameter substitution?
   - Can the gradient dict accumulation be vectorized?
   - Can `np.vdot` calls be batched or restructured?
   - Can the parameter_shift path be made faster by reusing bound circuits?
   - Can the adjoint cache hit rate be improved?

2. **Make the change.**
3. **Self-compliance check** (CLAUDE.md).
4. **Quick sanity**: `./venv/bin/python -m pytest tests/test_gradient.py -x -q 2>&1 | tail -3`
5. **Commit**: `git add -A && git commit -m "description"`
6. **Benchmark**: `./venv/bin/python benchmarks/gradient_benchmark.py > run.log 2>&1`
7. **Read results** and decide keep/discard/crash.
8. **Never stop.**

## Architecture of adjoint_gradient

Understanding the current implementation is critical:

```
adjoint_gradient(circuit, observable, params)
├── circuit.bind(params)              # create bound circuit
├── simulate(bound)                    # forward pass → state |ψ⟩
├── compute λ = O|ψ⟩                  # observable applied to state
│   ├── Z-only terms: flip signs      #   (fast path)
│   ├── X-only terms: np.flip         #   (fast path)
│   ├── Y-only terms: flip + phase    #   (fast path)
│   └── mixed terms: _apply_single_qubit per Pauli
└── _adjoint_backward(circuit, bound, state, lam)
    ├── build/cache adjoint_info       # gate → (adjoint_gate, params, indices)
    ├── precompute CX block boundaries # for batch permutation
    └── backward loop (k = N-1 → 0):
        ├── diagonal 1Q: accumulate phase, extract grad via vdot
        ├── CX block: batch permutation (state[perm_inv])
        └── non-diagonal: extract grad via vdot, then apply adjoint gate
            ├── 1Q: matmul on BOTH state and lam
            ├── 2Q: _apply_two_qubit on BOTH
            └── 3Q/4Q: _apply_three/four_qubit on BOTH
```

### Key bottleneck: the backward loop applies EVERY gate TWICE (once to state, once to λ)

This is inherent to adjoint differentiation but there are opportunities:
- For 1Q gates: batch the two matmuls into one (stack state+lam, matmul, unstack)
- For CX blocks: one permutation can apply to both state and lam
- For diagonal phases: accumulate once, apply to both simultaneously

### Hot path analysis (from comparison examples):

| Workload | Gradient calls/run | Params | Qubits | Dominant cost |
|----------|-------------------|--------|--------|---------------|
| VQE (200 steps) | 200 | 20-96 | 4-16 | backward matmul |
| ADAPT-VQE (pool) | 50-200 | 5-30 | 4-12 | pool evaluation |
| QAOA (60 steps) | 60 | 6-12 | 8-14 | RZZ backward |
| QML classifier | 100-200 | 30-90 | 1-8 | per-sample gradient |
| Kernel matrix | N² | 0 (eval only) | 8-16 | expectation |

## Optimization ideas (in rough order)

1. **Fuse state+lam matmul** — stack [state, lam] into (2, dim) and apply one matmul to both
2. **Avoid `circuit.bind()`** — directly substitute params into adjoint_info, skip circuit copy
3. **Batch vdot for gradient extraction** — accumulate indices, compute all vdots at once
4. **Faster λ = O|ψ⟩** — specialize Z-sum (diagonal, no copy needed), batch Pauli terms
5. **Reuse forward-pass buffers** — the forward simulate() allocates state, buf, tmp; pass to backward
6. **Eliminate redundant reshapes** — keep tensors in [2]*n form through backward pass
7. **Extend CX block batching to SWAP** — backward pass currently only batches CX, not SWAP
8. **Cache adjoint matrices** — non-parametric gates produce the same adjoint matrix every call
9. **Vectorize parameter_shift** — batch the +/- shifted simulations instead of sequential
10. **Inline hot gradient extraction** — avoid function call overhead for common gates (RY, RZ, RX)
11. **Pre-allocate gradient array** — use numpy array instead of dict for gradient accumulation
12. **Optimize `_build_adjoint_info`** — reduce per-gate overhead, cache more aggressively

## Important notes

- `adjoint_gradient` is ~50x faster than `parameter_shift_gradient` for the same circuit (1 forward + 1 backward vs 2×N_params forward passes).
- The backward pass does 2× the work of a forward pass (applies each gate to both state and λ).
- For small circuits (n<10), Python overhead dominates. For large circuits (n>14), numpy operations dominate.
- The `_adj_cache` on the circuit object caches `adjoint_info` across calls — only parametric gate values are updated.
- The diagonal phase accumulation (`diag_phase`) defers phase application — safe because phases are unit-magnitude.
- After every keep, run `./venv/bin/python -m pytest tests/test_gradient.py tests/test_determinism.py -x -q 2>&1 | tail -3` to verify correctness.
