# Autonomous Compiler Optimization

You are an autonomous researcher optimizing TinyQubit's quantum circuit compiler for speed while **strictly preserving compilation output**. Compiler correctness is paramount — any change that alters the compiled circuit's gate count, gate sequence, or simulation result is an automatic revert. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## CRITICAL: Correctness First

**The compiler has a 2623-line test suite (`tests/test_passes.py`).** Every experiment must pass it.

The benchmark checks **exact gate count match** — if your optimization changes the compiled output from 91 ops to 90 ops, that's a FAIL (even if the new circuit is better). The goal is making the compiler **faster at producing the same output**, not changing what it produces.

Before EVERY commit, run:
```
./venv/bin/python -m pytest tests/test_passes.py tests/test_compile.py tests/test_determinism.py -x -q 2>&1 | tail -3
```

## Setup

1. **Agree on a run tag**: Propose a tag (e.g. `comp-mar24`). Branch `optimize/<tag>` must not exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `CLAUDE.md` — coding standards.
   - `tinyqubit/compile.py` — **pipeline orchestration**. `precompile()`, `transpile()`, `_precompile_dag()`, `_realize_dag()`.
   - `tinyqubit/dag.py` — **DAG data structure**. `DAGCircuit`, `from_circuit()`, `to_circuit()`, `topological_order()`, `add_op()`, `remove_node()`, `commutes()`.
   - `tinyqubit/passes/optimize.py` — **the optimizer pass**. CX cancellation, CX conjugation, identity removal. Usually the slowest pass.
   - `tinyqubit/passes/fuse.py` — **1Q gate fusion**. `fuse_1q_gates()`, ZXZ decomposition. Called multiple times.
   - `tinyqubit/passes/decompose.py` — gate decomposition to basis. Called multiple times.
   - `tinyqubit/passes/push_diagonals.py` — diagonal gate commutation.
   - `tinyqubit/passes/route.py` — SABRE routing (for transpile only).
   - `tinyqubit/passes/layout.py` — qubit layout selection.
   - `tinyqubit/ir.py` — `Circuit`, `Gate`, `Operation` definitions.
   - `benchmarks/compiler_benchmark.py` — **the benchmark you run**. Read-only.
   - `benchmarks/compiler_baseline.json` — frozen outputs + baseline times. Read-only.
4. **Verify baseline exists**: `benchmarks/compiler_baseline.json`.
5. **Initialize `results.tsv`**.
6. **Confirm and go.**

## Rules

### What you CAN modify
- `tinyqubit/dag.py` — DAG construction/traversal, caching, data structure.
- `tinyqubit/compile.py` — pipeline structure, pass ordering, caching.
- `tinyqubit/passes/optimize.py` — optimizer internals (NOT its output).
- `tinyqubit/passes/fuse.py` — fusion internals (NOT its output).
- `tinyqubit/passes/decompose.py` — decomposition internals (NOT its output).
- `tinyqubit/passes/push_diagonals.py` — push_diag internals.
- `tinyqubit/ir.py` — if data structure changes help (e.g., faster Operation creation).

### What you CANNOT modify
- `benchmarks/compiler_benchmark.py` — read-only.
- `benchmarks/compiler_baseline.json` — read-only.
- `tests/` — read-only. **ALL 2623+ tests must pass.**
- No new dependencies — numpy only.

### CRITICAL: Regression Safety
- **Gate count must be IDENTICAL** before and after your changes.
- **Test suite must pass COMPLETELY** — not just the benchmark.
- If `test_passes.py` fails, **revert immediately**. Don't debug — revert and try a different approach.
- Run `tests/test_passes.py` BEFORE every commit, not just after.

## The experiment loop

**LOOP FOREVER:** Examine → Change → **Full test** → Commit → Benchmark → Keep/Revert → Repeat.

- **MUST run before every commit**: `./venv/bin/python -m pytest tests/test_passes.py tests/test_compile.py tests/test_determinism.py -x -q 2>&1 | tail -3`
- Benchmark: `./venv/bin/python benchmarks/compiler_benchmark.py > run.log 2>&1`
- Log to `results.tsv` (tab-separated): `commit total_ms correctness status description`

## Architecture: compiler pipeline

```
precompile(circuit)
├── DAGCircuit.from_circuit(circuit)     # Convert flat ops → DAG
├── decompose(dag, _ROUTING_BASIS)       # Break CCX/CP/etc into CX+1Q
├── push_diagonals(dag)                  # Commute Z/S/T/RZ past CX
├── fuse_1q_gates(dag)                   # Merge consecutive 1Q into one
├── optimize(dag)                        # CX cancellation, conjugation, identity elim
└── dag.to_circuit()                     # Convert DAG → flat ops

transpile(circuit, target)
├── precompile(circuit)
├── select_layout(dag, target)           # Choose initial qubit mapping
├── route(dag, target)                   # SABRE routing, insert SWAPs
├── absorb_trailing_swaps(dag)
├── decompose(dag, target.basis_gates)   # Lower to native gates
├── fuse_2q_blocks(dag) + decompose      # KAK synthesis
├── fix_direction(dag, target)
├── push_diagonals + fuse_1q + decompose # Final cleanup
└── optimize(dag)                        # Final optimization
```

### Key hot spots (from profiling):

| Function | Time | Called | Why expensive |
|----------|------|--------|---------------|
| `dag.add_op()` | 14ms | 9740× | Creates edges, updates adjacency |
| `np.isclose()` | 13ms | 960× | ZXZ decomposition angle checks |
| `_try_cx_conjugation` | 12ms | 3600× | Main optimizer loop |
| `topological_order()` | 9ms | 140× | Kahn's algorithm per pass |
| `commutes()` | 9ms | 4880× | Gate commutativity checks |
| `fuse._decompose_zxz` | 6ms | 480× | Euler angle decomposition |

## Optimization ideas (in rough order)

### Phase 1: Reduce per-pass overhead
1. **Cache topological order** — `topological_order()` (Kahn's algorithm) is called for every pass. Cache and invalidate only when DAG structure changes.
2. **Faster `commutes()`** — pre-compute commutation table for gate pairs. Currently does frozenset lookups and special cases per call.
3. **Faster `add_op()`** — the hot path in DAG construction. Reduce dict lookups and list operations.
4. **Replace `np.isclose()` with direct threshold check** — `np.isclose` has massive overhead for scalar comparisons.

### Phase 2: Reduce pass invocations
5. **Skip no-op passes** — if decompose produces no changes, skip fuse/optimize.
6. **Combine push_diag + fuse_1q** — these are always called together. Fuse them into one pass.
7. **Early termination in optimize** — the optimizer iterates up to `max_iterations`. Track whether anything changed; stop early.

### Phase 3: Algorithmic improvements
8. **Incremental DAG updates** — currently, `fuse_1q_gates` rebuilds the DAG. Use in-place node replacement instead.
9. **Batch ZXZ decomposition** — `_decompose_zxz` is called per fused 1Q gate. Pre-compute for common gate products.
10. **Lazy topological sort** — instead of full sort, maintain a sorted order and update incrementally when nodes change.

### Phase 4: Pipeline-level
11. **Cache DAG across calls** — for repeated `precompile()` on the same circuit structure (ADAPT-VQE), cache the DAG and only update changed nodes.
12. **Skip routing for trivial topologies** — if circuit connectivity matches target, skip SABRE entirely.
13. **Parallel multi-trial** — `multi_trials > 1` runs realize() N times. These are independent and could run in parallel.

## Important notes

- The benchmark includes `adapt_grow_8q` which simulates ADAPT-VQE's pattern of growing circuits + repeated compilation. This is the use case that motivated compiler optimization.
- `pre_random_10q` at 34ms and `trans_random_10q` at 57ms are the biggest targets.
- The `optimize` pass is called 2-3 times per transpile (once in precompile, once in realize). Making it faster has compound benefits.
- `DAGCircuit.from_circuit()` and `to_circuit()` are called at least twice per transpile. These involve iterating all ops and building data structures.
- The determinism tests (`test_determinism.py`) are CRITICAL — they verify that the same input always produces the same output. Any non-determinism in pass ordering or DAG traversal will cause failures.
- After every keep, run the FULL compile+pass test suite: `./venv/bin/python -m pytest tests/test_passes.py tests/test_compile.py tests/test_determinism.py -x -q 2>&1 | tail -5`
