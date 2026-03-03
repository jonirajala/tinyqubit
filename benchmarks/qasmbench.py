"""
Benchmark: tinyqubit vs Qiskit on QASMBench circuits.

Downloads small circuits from github.com/pnnl/QASMBench, transpiles with both
compilers to the same line topology and basis gates, then compares 2Q gate count
(quality) and compile time (speed).

Requires: pip install qiskit (optional, runs tinyqubit-only if missing)

Run: python benchmarks/qasmbench.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from urllib.request import urlopen
from urllib.error import URLError

from tinyqubit.ir import Gate, Circuit
from tinyqubit.export.qasm import from_openqasm2
from tinyqubit.target import Target
from tinyqubit.compile import transpile
from tinyqubit.simulator import verify

try:
    from qiskit import QuantumCircuit, transpile as qk_transpile
    from qiskit.transpiler import CouplingMap
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

_BASE = "https://raw.githubusercontent.com/pnnl/QASMBench/master/small"
_CIRCUITS = [
    "deutsch_n2",
    "iswap_n2",
    "fredkin_n3",
    "teleportation_n3",
    "qaoa_n3",
    "basis_change_n3",
    "cat_state_n4",
    "adder_n4",
    "hs4_n4",
    "variational_n4",
    "lpn_n5",
    "dnn_n2",
    "vqe_uccsd_n4",
    "linearsolver_n3",
]


def _fetch(name: str) -> str | None:
    url = f"{_BASE}/{name}/{name}.qasm"
    try:
        with urlopen(url, timeout=10) as resp:
            return resp.read().decode()
    except (URLError, TimeoutError):
        return None


def _line_edges(n):
    return frozenset((i, i + 1) for i in range(n - 1)) | frozenset((i + 1, i) for i in range(n - 1))


def _count_2q(ops):
    return sum(1 for op in ops if op.gate.n_qubits == 2)


def _qk_count_2q(qc):
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def _strip_measures(circuit):
    """Return copy without MEASURE/RESET for unitary verification."""
    c = Circuit(circuit.n_qubits)
    c.ops = [op for op in circuit.ops if op.gate not in (Gate.MEASURE, Gate.RESET) and op.condition is None]
    return c


def run_benchmark():
    if not HAS_QISKIT:
        print("Qiskit not installed — running tinyqubit-only. Install with: pip install qiskit\n")

    print("QASMBench: tinyqubit vs Qiskit transpile comparison")
    print("Target: line topology, basis={CX, H, RZ, SX}, optimization_level=3")
    print()

    hdr_tq = "tinyqubit"
    hdr_qk = "Qiskit-O3" if HAS_QISKIT else ""
    w = 100 if HAS_QISKIT else 62
    print(f"{'Circuit':<18} {'n':>2}  {'2Q in':>5}  {'2Q tq':>5} {'ms tq':>6} {'ok':>3}", end="")
    if HAS_QISKIT:
        print(f"  {'2Q qk':>5} {'ms qk':>6}  {'2Q winner':>9} {'speed winner':>12}", end="")
    print()
    print("-" * w)

    tq_wins_2q, qk_wins_2q, ties_2q = 0, 0, 0
    tq_wins_speed, qk_wins_speed = 0, 0
    total, ok, wrong = 0, 0, 0

    for name in _CIRCUITS:
        total += 1
        qasm = _fetch(name)
        if qasm is None:
            print(f"{name:<18} {'':>2}  {'SKIP — no network':>20}")
            continue

        try:
            circuit = from_openqasm2(qasm)
        except (ValueError, KeyError) as e:
            print(f"{name:<18} {'':>2}  PARSE ERR: {e}")
            continue

        n = circuit.n_qubits
        orig_2q = _count_2q(circuit.ops)

        # --- tinyqubit ---
        target = Target(n_qubits=n, edges=_line_edges(n),
                        basis_gates=frozenset({Gate.CX, Gate.H, Gate.RZ, Gate.SX}),
                        name=f"line_{n}")
        try:
            t0 = time.perf_counter()
            compiled = transpile(circuit, target)
            tq_ms = (time.perf_counter() - t0) * 1000
            tq_2q = _count_2q(compiled.ops)

            tracker = getattr(compiled, '_tracker', None)
            correct = verify(_strip_measures(circuit), _strip_measures(compiled), tracker=tracker)
            v_str = "Y" if correct else "N"
            if not correct: wrong += 1
            ok += 1
        except Exception as e:
            print(f"{name:<18} {n:>2}  {orig_2q:>5}  ERR: {e}")
            continue

        # --- Qiskit ---
        qk_2q = qk_ms = None
        if HAS_QISKIT:
            try:
                qc = QuantumCircuit.from_qasm_str(qasm)
                coupling = CouplingMap.from_line(n)
                t0 = time.perf_counter()
                qk_compiled = qk_transpile(qc, coupling_map=coupling,
                                           basis_gates=['cx', 'h', 'rz', 'sx'],
                                           optimization_level=3)
                qk_ms = (time.perf_counter() - t0) * 1000
                qk_2q = _qk_count_2q(qk_compiled)
            except Exception:
                pass

        # --- Print row ---
        print(f"{name:<18} {n:>2}  {orig_2q:>5}  {tq_2q:>5} {tq_ms:>5.1f}  {v_str:>2}", end="")
        if HAS_QISKIT and qk_2q is not None:
            if tq_2q < qk_2q: w2q = "tinyqubit"; tq_wins_2q += 1
            elif tq_2q > qk_2q: w2q = "Qiskit"; qk_wins_2q += 1
            else: w2q = "tie"; ties_2q += 1
            if tq_ms < qk_ms: wspeed = "tinyqubit"; tq_wins_speed += 1
            else: wspeed = "Qiskit"; qk_wins_speed += 1
            print(f"  {qk_2q:>5} {qk_ms:>5.1f}   {w2q:>9} {wspeed:>12}", end="")
        print()

    print("-" * w)
    print(f"Verified: {ok - wrong}/{ok}")
    if HAS_QISKIT:
        print(f"2Q gates — tinyqubit wins: {tq_wins_2q}, Qiskit wins: {qk_wins_2q}, ties: {ties_2q}")
        print(f"Speed    — tinyqubit wins: {tq_wins_speed}, Qiskit wins: {qk_wins_speed}")


if __name__ == "__main__":
    run_benchmark()
