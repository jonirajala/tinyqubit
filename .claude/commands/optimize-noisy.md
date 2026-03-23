# Autonomous Noisy Simulation Optimization

You are an autonomous researcher optimizing TinyQubit's noisy simulation — making Monte Carlo trajectory simulation, noise channel application, and ZNE error mitigation faster. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## Setup

1. **Agree on a run tag**: Propose a tag (e.g. `noisy-mar23`). Branch `optimize/<tag>` must not exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `CLAUDE.md` — coding standards.
   - `tinyqubit/simulator/noise.py` — **primary target**. `NoiseModel`, `_apply_gate_noise`, noise channel functions (`depolarizing`, `amplitude_damping`, `phase_damping`), Kraus operators, `realistic_noise`.
   - `tinyqubit/simulator/statevector.py` — `_apply_gate_noise()` called per gate, `_apply_single_qubit()` used by noise channels.
   - `tinyqubit/measurement/mitigation.py` — `zne()`, `_fold_circuit()`, `_noisy_exp()`.
   - `tinyqubit/simulator/simulator.py` — `simulate()` dispatch.
   - `tinyqubit/ir.py` — `Gate`, `Operation` definitions.
   - `benchmarks/noisy_benchmark.py` — **the benchmark you run**. Read-only.
   - `benchmarks/noisy_baseline.json` — frozen outputs + baseline times. Read-only.
4. **Verify baseline exists**: `benchmarks/noisy_baseline.json`.
5. **Initialize `results.tsv`**.
6. **Confirm and go.**

## Rules

### What you CAN modify
- `tinyqubit/simulator/noise.py` — primary target. Noise channel implementations, NoiseModel, Kraus operator caching.
- `tinyqubit/simulator/statevector.py` — `_apply_gate_noise()`, the noise application path in the main loop.
- `tinyqubit/measurement/mitigation.py` — `zne()`, `_fold_circuit()`, `_noisy_exp()`.
- `tinyqubit/simulator/simulator.py` — if dispatch changes help noisy path.
- `tinyqubit/ir.py` — if Operation/Gate changes help.

### What you CANNOT modify
- `benchmarks/noisy_benchmark.py` — read-only.
- `benchmarks/noisy_baseline.json` — read-only.
- `tests/` — read-only. Tests must pass.
- No new dependencies — numpy only.

### The goal
Get the **lowest total noisy simulation time** across all 11 benchmark cases while keeping **all 11 correctness checks passing**. Correctness is stochastic — mean must match within tolerance (0.05).

## The experiment loop

**LOOP FOREVER:** Examine → Change → Test → Commit → Benchmark → Keep/Revert → Repeat.

- Quick sanity: `./venv/bin/python -m pytest tests/test_noise.py tests/test_determinism.py -x -q 2>&1 | tail -3`
- Benchmark: `./venv/bin/python benchmarks/noisy_benchmark.py > run.log 2>&1`
- Log to `results.tsv` (tab-separated): `commit total_ms correctness status description`

## Architecture: how noisy simulation works today

```
simulate(circuit, noise_model=noise)
└── simulate_statevector(circuit, n, seed, noise_model, batch_ops=False)
    └── for each gate:
        ├── apply gate (normal statevector operation)
        └── _apply_gate_noise(state, op, noise_model, n, rng)
            ├── noise_list = noise_model.gate_noise.get(gate) or default_noise
            ├── for noise_fn in noise_list:        # 1-3 channels per gate
            │   └── for q in op.qubits:            # 1-2 qubits per gate
            │       └── noise_fn(state, q, n, rng)  # Python function call
            └── each noise_fn does:
                ├── depolarizing: rng.random() < p → apply random Pauli
                ├── amplitude_damping: compute p_jump → branch
                └── phase_damping: rng.random() < lam → collapse
```

### Key bottleneck: per-gate, per-qubit, per-channel Python function calls

For a `realistic_noise` model on an 8Q HEA with 141 ops:
- Each gate has ~3 noise channels (depolarizing + amplitude_damping + phase_damping)
- Each channel is applied per qubit (1Q gate: 1 qubit, 2Q gate: 2 qubits)
- Total: 141 × 3 × ~1.5 = **~635 Python function calls** per simulate
- Each call does: rng.random() + conditional + optional _apply_single_qubit

### Noisy sim disables batch_ops

When `noise_model is not None`, `batch_ops` is forced False (line 38 in simulator.py). This means:
- No kron grouping, no diagonal fusion, no CX perm batching
- Every gate applied individually through the slow Python dispatch
- This is the #1 reason noisy sim is 3.7× slower than clean

### ZNE multiplies the cost by scale_factor

`_fold_circuit(circuit, scale=5)` creates a circuit with 5× the ops (append U†U for each pair). With 3 scale factors × 30 shots: **90 noisy simulations** on increasingly large circuits.

## Optimization ideas (in rough order)

### Phase 1: Quick wins in noise channel application
1. **Cache noise channel lookup** — `noise_model.gate_noise.get(gate, default)` is called for every gate. Pre-compute a gate→channels mapping once.
2. **Skip no-op channels** — if `rng.random() >= p`, skip immediately without function call overhead. Inline the probability check.
3. **Batch channel application** — instead of per-qubit loops, apply noise to all qubits of a gate at once.
4. **Use cached index tuples** — `_get_1q_idx(n, qubit)` already exists; noise channels rebuild indices from scratch.
5. **Pre-compute Kraus matrices** — `_get_gate_matrix(Gate.X, ())` is called inside `depolarizing()` every time; cache the Pauli matrices.

### Phase 2: Enable batching with noise
6. **Allow batch_ops with noise** — the batch_ops flag is disabled for noisy sims because noise must be applied per gate. But 1Q gate batching (fusing RY+RZ into one matmul) is still valid if noise is applied AFTER the fused gate. Modify the batch path to apply noise after each logical gate group.
7. **Vectorize depolarizing** — for depolarizing noise with small p, most gates have no error. Generate all random numbers upfront and only apply noise where needed.
8. **Fuse depolarizing channel** — instead of `apply_single_qubit(state, X/Y/Z, q, n)`, depolarizing just needs to: apply X (swap halves), Y (swap + phase), or Z (negate half). These can be done in-place without `_apply_single_qubit`.

### Phase 3: Algorithmic improvements
9. **Sparse noise application** — for low p, pre-generate which gates get noise (binomial sampling), skip all others entirely.
10. **Vectorized trajectory** — instead of applying noise gate-by-gate, pre-compute which qubits get which Pauli errors, then apply all noise as a single phase mask + permutation.
11. **Faster ZNE fold** — `_fold_circuit` creates a new circuit with 5× ops. Could instead run the original circuit + replay backward/forward in the simulator without building the folded circuit.
12. **Amplitude damping fast path** — for small gamma, the jump probability is tiny. Skip the norm computation when no jump occurs.

### Phase 4: Deep optimizations
13. **Density matrix for noise** — for small circuits (n<8), density matrix simulation handles noise analytically (no Monte Carlo). Already available via `simulate_density`.
14. **Re-enable batch_ops for 1Q noise** — when noise is only on 1Q gates, kron grouping of consecutive 1Q gates is valid if noise is applied per-gate before fusion. This requires interleaving noise within the batch.

## Real-world impact

From the comparison examples:
- **chemistry/06_zne_noisy_h2o.py**: 412s total. Runs 3 noise levels × 3 scale factors × 200 VQE steps = ~1800 noisy simulations. Each sim ~2.5ms for 8Q.
- **chemistry/10_full_pipeline_n2.py**: 180s. ADAPT-VQE + noisy evaluation + ZNE.
- **finance/08_derivative_pricing_pde.py**: Uses 9Q noisy VQE on 512-point grid Hamiltonian.

A 2× speedup on noisy simulation saves **200+ seconds** on the ZNE example alone.

## Important notes

- Noisy simulation is STOCHASTIC — results vary between runs even with the same seed (Monte Carlo trajectories). Correctness tolerance is 0.05, not 1e-10.
- The `realistic_noise()` model applies 3 channels per gate (depolarizing + amplitude_damping + phase_damping). This is the worst case for per-channel overhead.
- `batch_ops` is currently forced False for noisy simulation. Re-enabling it for the forward gate application (while keeping per-gate noise) is the single biggest potential win.
- The `_apply_gate_noise` function reshapes state to `[2]*n` and back for every gate — redundant when the gate dispatch already does this.
- After every keep, run: `./venv/bin/python -m pytest tests/test_noise.py tests/test_determinism.py -x -q 2>&1 | tail -3`
