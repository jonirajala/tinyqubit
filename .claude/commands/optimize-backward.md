# Autonomous Adjoint Backward Pass Optimization

You are an autonomous researcher optimizing TinyQubit's adjoint backward pass — the single most expensive operation in VQE/QML workloads. You run experiments in a loop, keep improvements, revert failures, and never stop until interrupted.

## Context: Why This Matters

The adjoint backward pass is **3.1x slower than PennyLane's C++ lightning.qubit** at 16Q. This is the last major gap. The forward simulation is already FASTER than lightning.qubit (our kron grouping wins). Closing this gap in pure Python would be a significant achievement.

Current 16Q HEA breakdown:
- Forward simulate: **6ms** (kron-grouped, 4 matmuls)
- Backward pass: **~20-25ms** (fused RZ-RY pairs, kron-grouped, but still ~10 matmuls + 96 vdots + 6 perm takes)
- Lambda computation: **~2ms**
- PennyLane total: **~12ms** (all in C++)

## Setup

1. **Tag**: e.g. `back-mar24`. Branch `optimize/<tag>` must not exist.
2. **Create branch** from current HEAD.
3. **Read**:
   - `tinyqubit/qml/optim.py` — **primary target**: `_adjoint_backward()`, `_build_adjoint_info()`, `adjoint_gradient()`
   - `tinyqubit/simulator/statevector.py` — gate application functions used by backward
   - `tinyqubit/measurement/observable.py` — lambda (O|ψ⟩) computation
   - `tinyqubit/ir.py` — Circuit, Gate, Operation
   - `benchmarks/backward_benchmark.py` — **read-only benchmark**
   - `benchmarks/backward_baseline.json` — **read-only baseline**
4. **Verify baseline**, init `results.tsv`, go.

## Rules

- CAN modify: `tinyqubit/qml/optim.py`, `tinyqubit/simulator/statevector.py`, `tinyqubit/measurement/observable.py`, `tinyqubit/ir.py`
- CANNOT modify: benchmarks, tests. No new dependencies.
- Quick sanity: `./venv/bin/python -m pytest tests/test_gradient.py tests/test_determinism.py -x -q 2>&1 | tail -3`
- Benchmark: `./venv/bin/python benchmarks/backward_benchmark.py > run.log 2>&1`

## What the backward pass does today

```
_adjoint_backward(circuit, bound, state, lam):
├── Pack state+lam into sl = (2, dim) pair
├── Pre-build perm_block_start for CX/SWAP batch boundaries
├── Loop k = N-1 → 0:
│   ├── RZ-RY FUSED PAIR (when RZ at k, RY at k-1, same qubit, no pending diag):
│   │   ├── Scan backward collecting consecutive fused pairs on different qubits
│   │   ├── Extract ALL gradients from same state (phase-corrected RY formula)
│   │   ├── Kron-group combined RY†RZ† matrices (groups of up to 5)
│   │   └── Apply via matmul on sl pair
│   ├── DIAGONAL 1Q (RZ, S, T, etc. — not fused):
│   │   ├── Extract gradient via vdot (phase-invariant)
│   │   └── Accumulate phase into diag_phase
│   ├── CX/SWAP PERM BLOCK:
│   │   ├── Flush pending diagonal phase
│   │   └── Apply inverse permutation via np.take on both state+lam
│   ├── NON-DIAGONAL 1Q (RY, RX, H — not part of RZ-RY pair):
│   │   ├── Flush diagonal phase
│   │   ├── Collect consecutive non-diagonal 1Q on different qubits
│   │   ├── Extract gradients, kron-group, apply matmul
│   │   └── OR: single gate matmul with 2D GEMM edge cases
│   └── 2Q/3Q/4Q gates:
│       └── Apply adjoint to both state and lam
└── Final flush of any pending diagonal phase
```

## Key bottlenecks (from profiling 16Q HEA)

| Operation | Time | Count/run | What |
|-----------|------|-----------|------|
| RZ-RY fused matmul | ~10ms | ~10 kron groups | Kron-grouped matmul on (2, dim) sl pair |
| Gradient vdots | ~6ms | 96 | np.vdot on 32K-element sub-arrays |
| CX perm takes | ~3ms | 6 | np.take random access on 65K elements |
| np.exp for phases | ~1ms | 48 | Phase angle computation |
| np.array for matrices | ~1ms | ~50 | Constructing 2×2 diagonal matrices |
| Python overhead | ~5ms | — | Loop dispatch, dict lookups, branches |

## Optimization ideas

### Phase 1: Reduce numpy call overhead
1. **Pre-compute RZ diagonal matrices** — `np.array([[e0,0],[0,e1]])` is rebuilt every fused pair. Cache by angle or use in-place construction.
2. **Batch vdot calls** — 96 individual `np.vdot` calls on 32K sub-arrays. Could reshape into one matrix multiply.
3. **Use `cmath.exp` for scalar phase** — `np.exp(1j * theta)` has numpy dispatch overhead. `cmath.exp` is 10x faster for scalars.
4. **Cache `_get_1q_idx` results per backward call** — pre-build all (n, qubit) → (i0, i1) pairs once.

### Phase 2: Reduce number of operations
5. **Extend kron group size to 5+** — currently limited to 5 (32×32 matmul). For 16Q with 16 fused pairs, groups of 8 (256×256) would give 2 matmuls instead of 4.
6. **Fuse the phase flush with the next matmul** — when diag_phase is pending and the next op is a matmul, absorb the phase into the matrix instead of a separate state multiplication.
7. **Batch CX perm with adjacent non-param gates** — if CX blocks have non-parametric 1Q gates nearby, batch them.

### Phase 3: Algorithmic improvements
8. **Vectorized gradient extraction** — instead of 96 individual vdots, express all gradients as a single matrix operation on the (2, dim) sl pair.
9. **Fuse lambda computation into backward** — the lambda (O|ψ⟩) for Z-sum observables is just a diagonal scaling. Could fold it into the first backward step.
10. **Skip zero-gradient gates** — gates whose gradient is provably zero (e.g., RZ with θ=0) can be skipped entirely.

### Phase 4: Memory and layout
11. **Keep sl in tensor form [2, 2, ..., 2]** — avoid reshape between operations.
12. **Pre-allocate all intermediate buffers** — the kron-grouped matmul builds combined matrices that are allocated per group. Pre-allocate the max-size matrix.
13. **Avoid `np.conj(e_phase)` in gradient formula** — compute both `e_phase` and its conjugate once.

## Benchmark circuits and what they test

| Benchmark | Pattern | Backward bottleneck |
|-----------|---------|-------------------|
| hea_*q_*l | RY+RZ+CX layers | RZ-RY fusion + kron grouping (main target) |
| hea_20q_2l | Large state (16MB) | Memory bandwidth in matmul |
| hea_8q_heisen | Mixed Pauli lambda | Lambda computation for non-Z terms |
| qaoa_*q_p3 | RZZ+RX layers | RZZ gradient (4 vdots) + non-fused RX |
| uccsd_8q | CX-RY-CX ladders | No RZ-RY pairs, individual gate processing |
| random_*q_d5 | Mixed RX/RY/RZ | Partial fusion, mixed gate types |
| swap_10q | SWAP+CX routing | Perm batching in backward |
| hea_8q_bigH | 100-term Hamiltonian | Lambda computation overhead |

## Important notes

- The sl (2, dim) pair fusion means EVERY matmul operates on 2× the data. This is intentional — one BLAS call instead of two.
- The RZ-RY phase-corrected gradient formula: `Re(e^{iθ}⟨λ₁|ψ₀⟩ - e^{-iθ}⟨λ₀|ψ₁⟩)` — this was derived and verified. Don't change the math.
- The kron-grouped backward extracts ALL gradients from the SAME state before applying any adjoint gates. This works because gates on different qubits commute in the gradient formula.
- After every keep: `./venv/bin/python -m pytest tests/test_gradient.py tests/test_chemistry.py tests/test_determinism.py -x -q 2>&1 | tail -3`
- The 20Q benchmark at 932ms is dominated by memory bandwidth (16MB state × 2 for sl pair). Optimizations here need to reduce state passes, not Python overhead.
