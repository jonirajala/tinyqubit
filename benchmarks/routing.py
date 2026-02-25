"""
Benchmark: tinyqubit vs Qiskit routing.

Compares 2Q gate counts after routing for multiple topologies.
Requires: pip install qiskit

Run: python benchmarks/routing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyqubit.ir import Circuit, Gate
from tinyqubit.dag import DAGCircuit
from tinyqubit.target import Target
from tinyqubit.passes.route import route
from tinyqubit.passes.layout import select_layout

from benchmarks.topologies import (
    line_topology,
    grid_topology,
    heavy_hex_topology,
    all_to_all_topology,
    create_target,
    create_coupling_map,
)
from benchmarks.metrics import count_swaps, count_cx_equivalent

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Install with: pip install qiskit\n")


def count_2q(circuit):
    return sum(1 for op in circuit.ops if op.gate in {Gate.CX, Gate.CZ, Gate.SWAP})


def create_test_circuits(n_qubits: int) -> list[tuple[str, Circuit, "QuantumCircuit | None", int]]:
    """Create test circuits for a given number of qubits."""
    tests = []

    # CX between non-adjacent qubits (0 to n-1)
    tq = Circuit(n_qubits).h(0).cx(0, n_qubits - 1)
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n_qubits)
        qk.h(0)
        qk.cx(0, n_qubits - 1)
    tests.append(("cx_far", tq, qk, 1))

    # Multiple non-adjacent CX
    if n_qubits >= 5:
        tq = Circuit(n_qubits).cx(0, n_qubits - 1).cx(1, n_qubits - 2)
        if n_qubits >= 6:
            tq.cx(0, n_qubits // 2)
        qk = None
        if HAS_QISKIT:
            qk = QuantumCircuit(n_qubits)
            qk.cx(0, n_qubits - 1)
            qk.cx(1, n_qubits - 2)
            if n_qubits >= 6:
                qk.cx(0, n_qubits // 2)
        tests.append(("multi_cx", tq, qk, 3 if n_qubits >= 6 else 2))

    # GHZ pattern (star from qubit 0)
    tq = Circuit(n_qubits).h(0)
    for i in range(1, min(n_qubits, 5)):
        tq.cx(0, i)
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n_qubits)
        qk.h(0)
        for i in range(1, min(n_qubits, 5)):
            qk.cx(0, i)
    tests.append(("ghz_star", tq, qk, min(n_qubits - 1, 4)))

    # Linear GHZ (chain)
    tq = Circuit(n_qubits).h(0)
    for i in range(n_qubits - 1):
        tq.cx(i, i + 1)
    qk = None
    if HAS_QISKIT:
        qk = QuantumCircuit(n_qubits)
        qk.h(0)
        for i in range(n_qubits - 1):
            qk.cx(i, i + 1)
    tests.append(("ghz_chain", tq, qk, n_qubits - 1))

    # QFT-like pattern
    if n_qubits >= 4:
        tq = Circuit(n_qubits).h(0)
        for j in range(1, min(4, n_qubits)):
            tq.cx(0, j)
        tq.h(1)
        for j in range(2, min(4, n_qubits)):
            tq.cx(1, j)
        qk = None
        if HAS_QISKIT:
            qk = QuantumCircuit(n_qubits)
            qk.h(0)
            for j in range(1, min(4, n_qubits)):
                qk.cx(0, j)
            qk.h(1)
            for j in range(2, min(4, n_qubits)):
                qk.cx(1, j)
        orig_2q = min(3, n_qubits - 1) + min(2, n_qubits - 2)
        tests.append(("qft_like", tq, qk, orig_2q))

    return tests


def run_topology_benchmark(topology_name: str, n_qubits: int, edges: frozenset, verbose: bool = True):
    """Run benchmark for a specific topology."""
    target = create_target(topology_name, n_qubits, edges)
    coupling = create_coupling_map(edges) if HAS_QISKIT else None

    tests = create_test_circuits(n_qubits)

    results = []
    for name, tq_c, qk_c, orig_2q in tests:
        # Skip if circuit requires more qubits than available
        if tq_c.n_qubits > n_qubits:
            continue

        dag = DAGCircuit.from_circuit(tq_c)
        layout = select_layout(dag, target)
        routed = route(tq_c, target, initial_layout=layout)
        tq_2q = count_2q(routed)
        tq_swaps = count_swaps(routed)
        tq_cx_eq = count_cx_equivalent(routed)

        if HAS_QISKIT and qk_c is not None and coupling is not None:
            try:
                qk_routed = transpile(qk_c, coupling_map=coupling, optimization_level=3)
                qk_2q = qk_routed.count_ops().get('cx', 0) + qk_routed.count_ops().get('swap', 0) * 3
            except Exception:
                qk_2q = "-"
        else:
            qk_2q = "-"

        if isinstance(qk_2q, int):
            if tq_cx_eq < qk_2q:
                winner = "tinyqubit"
            elif tq_cx_eq > qk_2q:
                winner = "Qiskit"
            else:
                winner = "tie"
        else:
            winner = "-"

        results.append({
            "name": name,
            "orig_2q": orig_2q,
            "tq_2q": tq_2q,
            "tq_swaps": tq_swaps,
            "tq_cx_eq": tq_cx_eq,
            "qk_2q": qk_2q,
            "winner": winner,
        })

    return results


def run_benchmark():
    """Run routing benchmark across multiple topologies."""

    # Define topologies to test
    # heavy_hex returns (edges, actual_n_qubits)
    heavy_hex_8_edges, heavy_hex_8_n = heavy_hex_topology(8)
    heavy_hex_11_edges, heavy_hex_11_n = heavy_hex_topology(11)

    topologies = [
        ("line_5", 5, line_topology(5)),
        ("line_10", 10, line_topology(10)),
        ("grid_2x3", 6, grid_topology(2, 3)),
        ("grid_3x3", 9, grid_topology(3, 3)),
        ("heavy_hex_8", heavy_hex_8_n, heavy_hex_8_edges),
        ("heavy_hex_11", heavy_hex_11_n, heavy_hex_11_edges),
        ("all_to_all_5", 5, all_to_all_topology(5)),
    ]

    for topo_name, n_qubits, edges in topologies:
        print(f"\n{'='*75}")
        print(f"  Topology: {topo_name} ({n_qubits} qubits, {len(edges)//2} edges)")
        print(f"{'='*75}")
        print()
        print("2Q gate counts after routing (lower is better)")
        print("-" * 75)
        print(f"{'Circuit':<12} {'Orig':>6} {'TQ-2Q':>8} {'TQ-SWAP':>8} {'TQ-CXeq':>8} {'QK-CXeq':>8} {'Winner':>10}")
        print("-" * 75)

        results = run_topology_benchmark(topo_name, n_qubits, edges)

        for r in results:
            print(f"{r['name']:<12} {r['orig_2q']:>6} {r['tq_2q']:>8} {r['tq_swaps']:>8} {r['tq_cx_eq']:>8} {r['qk_2q']:>8} {r['winner']:>10}")

        print("-" * 75)

    # Summary table
    print("\n" + "="*75)
    print("  SUMMARY: Wins by topology")
    print("="*75)
    print(f"{'Topology':<16} {'tinyqubit':>12} {'Qiskit':>12} {'Tie':>12}")
    print("-" * 52)

    for topo_name, n_qubits, edges in topologies:
        results = run_topology_benchmark(topo_name, n_qubits, edges, verbose=False)
        wins = {"tinyqubit": 0, "Qiskit": 0, "tie": 0, "-": 0}
        for r in results:
            wins[r["winner"]] = wins.get(r["winner"], 0) + 1
        print(f"{topo_name:<16} {wins['tinyqubit']:>12} {wins['Qiskit']:>12} {wins['tie']:>12}")

    print("-" * 52)


if __name__ == "__main__":
    run_benchmark()
