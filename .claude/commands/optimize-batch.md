# Autonomous Batched Evaluation Optimization

You are an autonomous researcher building and optimizing TinyQubit's batched evaluation infrastructure — making repeated circuit evaluations (sweeps, kernel matrices, batch gradients) faster by eliminating per-call overhead. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## Setup

1. **Agree on a run tag**: Propose a tag (e.g. `batch-mar22`). Branch `optimize/<tag>` must not exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `CLAUDE.md` — coding standards.
   - `tinyqubit/measurement/observable.py` — `expectation()`, `expectation_batch()`, `expectation_sweep()`. Primary optimization target.
   - `tinyqubit/simulator/simulator.py` — `simulate()` dispatch, validation.
   - `tinyqubit/simulator/statevector.py` — `simulate_statevector()`, buffer allocation.
   - `tinyqubit/qml/optim.py` — `adjoint_gradient()`, optimizer loops.
   - `tinyqubit/ir.py` — `Circuit.bind()`, `Circuit.bind_params()`, parameter resolution.
   - `benchmarks/batch_benchmark.py` — **the benchmark you run**. Read-only.
   - `benchmarks/batch_baseline.json` — frozen outputs + baseline times. Read-only.
4. **Verify baseline exists**: `benchmarks/batch_baseline.json`.
5. **Initialize `results.tsv`**.
6. **Confirm and go.**

## Rules

### What you CAN modify
- `tinyqubit/measurement/observable.py` — add/optimize batched evaluation APIs.
- `tinyqubit/simulator/simulator.py` — add batched simulate, skip repeated validation.
- `tinyqubit/simulator/statevector.py` — add buffer-reuse path for batched calls.
- `tinyqubit/qml/optim.py` — batched adjoint_gradient, optimizer loop optimization.
- `tinyqubit/ir.py` — faster `bind_params`, circuit template caching.

### What you CANNOT modify
- `benchmarks/batch_benchmark.py` — read-only.
- `benchmarks/batch_baseline.json` — read-only.
- `tests/` — read-only. Tests must pass.
- No new dependencies — numpy only.

### The goal
Get the **lowest total batched evaluation time** across all 9 benchmark cases while keeping **all 9 correctness checks passing**.

## The experiment loop

**LOOP FOREVER:** Examine → Change → Test → Commit → Benchmark → Keep/Revert → Repeat.

- Quick sanity: `./venv/bin/python -m pytest tests/test_gradient.py tests/test_determinism.py -x -q 2>&1 | tail -3`
- Benchmark: `./venv/bin/python benchmarks/batch_benchmark.py > run.log 2>&1`
- Log to `results.tsv` (tab-separated): `commit total_ms correctness status description`

## Architecture: what happens per call today

Every `expectation(circuit, observable)` call does:
```
expectation(circuit, observable)
├── circuit.bind()                     # ALLOCATE new Circuit, copy all ops
├── simulate(bound)
│   ├── validate all ops               # ITERATE over all ops checking params/qubits
│   ├── is_clifford(circuit)           # ITERATE over all ops checking gate types
│   └── simulate_statevector()
│       ├── np.zeros(2**n)             # ALLOCATE state vector
│       ├── np.empty_like(state)       # ALLOCATE buf
│       ├── np.empty(2**(n-1))         # ALLOCATE tmp
│       └── gate loop...              # THE ACTUAL WORK
├── for each Pauli term:
│   ├── state.copy()                   # ALLOCATE copy per term
│   └── _apply_single_qubit per Pauli  # small matmuls
└── return result
```

**Per-call overhead that batching can eliminate:**
- `circuit.bind()`: copies Circuit + all ops → **use `bind_params()` on work circuit**
- Parameter validation: iterates all ops → **validate once, skip on repeats**
- `is_clifford()`: iterates all ops → **check once, cache result**
- Buffer allocation (state, buf, tmp): 3 large arrays → **pre-allocate, reuse**
- Expectation Pauli copies: `state.copy()` per term → **reuse buffer**

## Optimization ideas (in rough order)

### Phase 1: Low-hanging fruit (modify existing functions)
1. **`simulate_statevector` with pre-allocated buffers** — add optional `state_buf`, `work_buf`, `tmp_buf` params that skip allocation when provided
2. **`simulate` with skip_validation flag** — after first call on a circuit structure, skip the ops validation loop
3. **`expectation` with pre-simulated state** — already supported (pass ndarray), but callers don't use it
4. **`bind_params` on work circuit** — `expectation_sweep` already does this, but `expectation_batch` doesn't

### Phase 2: New batched APIs
5. **`simulate_batch(circuit, param_sets)`** — simulate same circuit with N param sets, reusing buffers
6. **`expectation_multi(circuit, observable, param_sets)`** — combines bind_params + simulate + expectation with full buffer reuse
7. **`kernel_matrix(circuit, X_data)`** — compute fidelity kernel with symmetry, returning only upper triangle computed
8. **`adjoint_gradient_batch(circuit, observable, param_sets)`** — batch gradient with shared adjoint_info cache

### Phase 3: Deep optimizations
9. **Pre-compute observable application** — for Z-only observables, the expectation is just `sum(probs * weights)`, no state copy needed
10. **Fuse simulate+expectation** — skip materializing the full state when only expectation is needed
11. **Vectorize parameter binding** — for simple circuits (no ScaledParam), bind all param sets as a batch operation

## Key insight: the benchmark circuits

| Benchmark | What it does | Bottleneck | Optimization opportunity |
|-----------|-------------|-----------|--------------------------|
| sweep_*_*pts | 50-100 evals, 1 param varies | per-call overhead + simulate | buffer reuse + skip validation |
| eval_*_*sets | 30-50 evals, all params vary | per-call overhead + simulate | buffer reuse + skip validation |
| kernel_*_NxN | N(N+1)/2 pairs of simulate | 2 simulates per pair + inner product | symmetry + buffer reuse |
| grad_batch_* | N adjoint_gradient calls | forward + backward per call | shared adjoint_info cache |

## Important notes

- The benchmark runner calls your functions via the EXISTING API (`bench_expectation_sweep`, `bench_multi_param_eval`, etc.). Your optimizations must make the existing call patterns faster — you can't change how the benchmark calls things.
- The benchmark uses `circuit.bind({})` then `work.bind_params(params)` for sweeps/evals. Your job: make `bind_params` + `simulate` + `expectation` faster when called in a loop.
- Kernel benchmarks call `simulate(circuit.bind(params))` in a double loop. Making `simulate` reuse buffers across calls is the big win here.
- After every keep, run full tests: `./venv/bin/python -m pytest tests/ -x -q --ignore=tests/test_performance.py 2>&1 | tail -3`
