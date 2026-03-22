# Autonomous Statevector Simulator Optimization

You are an autonomous researcher optimizing TinyQubit's statevector simulator for speed while preserving correctness. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## Setup

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `mar22`). The branch `optimize/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `CLAUDE.md` — coding standards. All changes MUST follow these conventions.
   - `tinyqubit/simulator/statevector.py` — primary optimization target. Gate application, batching, the hot loop.
   - `tinyqubit/simulator/simulator.py` — dispatch logic, simulate(), expectation helpers.
   - `tinyqubit/simulator/mps.py` — MPS backend (context, don't break).
   - `tinyqubit/ir.py` — Circuit, Gate, Operation definitions (context for gate matrix lookups).
   - `tinyqubit/measurement/observable.py` — expectation value computation.
   - `benchmarks/simulator_benchmark.py` — **the benchmark you run**. Read-only.
   - `benchmarks/simulator_baseline.json` — frozen golden outputs + baseline times. Read-only.
4. **Verify baseline exists**: Check that `benchmarks/simulator_baseline.json` exists. If not, run `./venv/bin/python benchmarks/simulator_benchmark.py --save`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go.**

## Rules

### What you CAN modify
Any file in `tinyqubit/simulator/` and `tinyqubit/ir.py`:
- `tinyqubit/simulator/statevector.py` — primary target. Gate application, memory layout, batching, caching, numpy vectorization, loop restructuring.
- `tinyqubit/simulator/simulator.py` — dispatch logic, helper functions.
- `tinyqubit/simulator/mps.py` — if optimization also benefits MPS.
- `tinyqubit/ir.py` — if gate matrix caching or Operation structure changes help.
- `tinyqubit/measurement/observable.py` — if expectation computation can be faster.

### What you CANNOT modify
- `benchmarks/simulator_benchmark.py` — read-only. This is the ground truth.
- `benchmarks/simulator_baseline.json` — read-only. Golden outputs are frozen.
- `tests/` — read-only. Tests must keep passing.
- No new dependencies — numpy only. Zero exceptions.

### Coding standards (from CLAUDE.md)
Every change MUST follow these — violations get reverted even if faster:
- **Minimal LOC.** Refactor until the change is small. No bloat.
- **Deterministic.** Same input → same output, always. Golden tests enforce this.
- **Explainable.** A reader should understand any function in under 30 seconds.
- **No over-engineering.** No error handling for impossible cases. No abstractions for one-time operations.
- **No unnecessary changes.** Don't add docstrings, type hints, or comments to code you didn't change.
- **Naming:** `snake_case` for functions, `UPPER_CASE` for constants, leading `_` for internal.
- **Style:** 4-space indent, ~120 char lines, ternary over if/else when simple, list comprehensions when clear.
- **Comments:** Sparse. Explain *why*, not *what*. `# NOTE:` for gotchas.

### The goal
Get the **lowest total simulation time** across all 16 benchmark circuits while keeping **all 16 correctness checks passing**. Correctness is a hard constraint — any hash mismatch is an automatic revert.

### Key metric
The benchmark prints per-circuit times. The metric to optimize is the **sum of median times across all circuits**. Extract it like this:
```
grep "Time(ms)" -A 20 run.log | grep -E "^\s+" | awk '{sum += $4} END {print sum}'
```

### Simplicity criterion
All else being equal, simpler is better. A 1% speedup that adds 50 lines of complexity? Probably not worth it. A 5% speedup from restructuring existing code? Great. Removing code and getting equal speed? Even better. This is a tiny codebase — every line must earn its place.

## Output format

The benchmark prints:
```
Correctness:
  ghz_4                PASS  (hash=a56addf9.. norm=1.0000000000)
  ...

Performance:
  Circuit              Qubits    Ops   Time(ms)   Baseline     Change
  ghz_20                   20     20    115.361    117.943        -2%
  ...

Summary: 16/16 correctness PASS, 3 faster, 0 regressions
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	total_ms	correctness	status	description
a1b2c3d	2104.5	16/16	keep	baseline
b2c3d4e	1987.2	16/16	keep	vectorize _apply_single_qubit inner loop
c3d4e5f	0.0	14/16	crash	broke CX permutation (hash mismatch on ghz_20)
```

Columns:
- `commit`: git short hash (7 chars)
- `total_ms`: sum of all circuit times (0.0 for crashes/correctness failures)
- `correctness`: N/16 passing
- `status`: `keep`, `discard`, or `crash`
- `description`: what the experiment tried

## The experiment loop

**LOOP FOREVER:**

1. **Examine the code**: Read the relevant files, identify an optimization opportunity. Think about:
   - Can a loop be vectorized with numpy?
   - Can we reduce memory allocations (pre-allocate buffers)?
   - Can we fuse operations (combine consecutive gates)?
   - Can we use better numpy operations (matmul vs einsum vs manual)?
   - Can we avoid reshape/copy operations?
   - Can we improve cache locality (memory access patterns)?
   - Can we specialize hot paths (e.g., skip general code for common gates)?
   - Can we reduce Python overhead (fewer function calls, less branching)?
   - Can gate matrix lookups in ir.py be faster?
   - Can expectation computation in observable.py avoid unnecessary work?

2. **Make the change** — edit the relevant file(s).

3. **Self-compliance check** (from CLAUDE.md workflow): Before committing, verify:
   - No bloat added (is this the minimal diff?)
   - No unnecessary abstractions
   - No duplicated code
   - No unconditional overhead (e.g., don't add setup cost that runs even when not needed)
   - Naming follows conventions

4. **Quick sanity check**: Run `./venv/bin/python -m pytest tests/test_determinism.py tests/test_mps.py -x -q 2>&1 | tail -3` to catch obvious breakage.

5. **Commit**: `git add -A && git commit -m "description"`

6. **Run the benchmark**: `./venv/bin/python benchmarks/simulator_benchmark.py > run.log 2>&1`

7. **Read results**: `grep "Summary:" run.log` and `grep "Time(ms)" -A 20 run.log`
   - If correctness fails: this is a **crash**. The change broke simulation output.
   - Compute total_ms from the per-circuit times.

8. **Decide**:
   - **Correctness failure**: Log as crash, `git reset --hard HEAD~1`, move on.
   - **Faster (total_ms decreased)**: Log as keep, advance.
   - **Slower or equal**: Log as discard, `git reset --hard HEAD~1`, try something else.

9. **Never stop**. If you run out of ideas:
   - Re-read all in-scope files line by line for micro-optimizations.
   - Profile with `python -c "import cProfile; ..."` to find the actual bottleneck.
   - Try combining previous near-misses.
   - Try more radical restructuring (different memory layout, different gate application strategy).
   - Read numpy documentation for faster operations.
   - Think about what makes lightning.qubit fast (C++ inner loop) and approximate it in numpy.
   - Look at the gate dispatch chain: ir.py gate lookup → statevector gate apply. Can the chain be shorter?

## Optimization ideas to try (in rough order)

1. **Pre-allocate state buffer** — avoid `state.copy()` in gate application
2. **Fuse diagonal phases** — combine consecutive RZ/S/T/CZ into single phase array
3. **Vectorize 1Q gate application** — use `np.matmul` broadcast instead of slice assignment
4. **In-place operations** — `state *= phase` instead of `state = state * phase`
5. **Specialize CX** — custom implementation instead of going through general 2Q path
6. **Reduce reshaping** — keep state in `[2]*n` tensor form between gates on same qubit range
7. **Gate matrix caching** — precompute parametric gate matrices, cache in ir.py
8. **Batch non-adjacent CX** — group CX gates that don't share qubits
9. **SIMD-friendly memory layout** — ensure inner dimension is contiguous for matmul
10. **Faster expectation** — specialize Pauli expectation (Z is just sign flip, no matmul needed)
11. **Reduce Python call overhead** — inline hot functions, avoid isinstance checks in inner loop
12. **Flatten the dispatch** — reduce levels of function calls between simulate() and the numpy operation

## Important notes

- The benchmark includes circuits from 4Q to 23Q. Small circuits (<10Q) are dominated by Python overhead. Focus optimization effort on 14-23Q circuits where the numpy operations dominate.
- The `batch_ops` flag is already enabled for n>=10. Your optimizations should work with or without batching.
- State vector at 23Q is 8M complex numbers = 128MB. Memory access patterns matter.
- `_apply_single_qubit` is called most often. `_apply_two_qubit` (especially CX) is second.
- The existing CX permutation optimization (`_get_cx_perm` with LRU cache) is clever — don't break it, but see if you can extend the pattern.
- Simulator qubit ordering: q0 = MSB. `kron(M_q0, M_q1)` for gate matrices. Don't change this convention.
- After every keep, run `./venv/bin/python -m pytest tests/ -x -q --ignore=tests/test_performance.py 2>&1 | tail -3` to verify full test suite still passes.
