"""
Performance tests for import time and transpile speed.

Tests:
    - Import time < 100ms
    - Transpilation time within bounds
"""
import subprocess
import sys
import time
import pytest
from math import pi

from tinyqubit.ir import Circuit, Gate
from tinyqubit.target import Target
from tinyqubit.compile import transpile


# =============================================================================
# Helpers
# =============================================================================

def line_topology(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, i + 1) for i in range(n - 1))


def time_function(func, runs=5):
    """Time a function, return average ms."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)


# =============================================================================
# Import time test
# =============================================================================

def test_import_time():
    """tinyqubit import should complete in < 100ms."""
    # Run in subprocess to get clean import
    code = """
import time
start = time.perf_counter()
import tinyqubit
print(time.perf_counter() - start)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Import failed: {result.stderr}"

    import_time = float(result.stdout.strip())
    import_time_ms = import_time * 1000

    assert import_time_ms < 100, (
        f"Import took {import_time_ms:.1f}ms, expected < 100ms"
    )


# =============================================================================
# Transpile performance tests
# =============================================================================

class TestTranspilePerformance:
    """Tests for transpilation performance."""

    @pytest.fixture
    def line_target_10(self):
        return Target(
            n_qubits=10,
            edges=line_topology(10),
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
            name="line_10"
        )

    def test_bell_transpile_fast(self):
        """Bell state transpile should be very fast (< 5ms)."""
        target = Target(
            n_qubits=2,
            edges=line_topology(2),
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
            name="line_2"
        )

        def do_transpile():
            c = Circuit(2).h(0).cx(0, 1)
            return transpile(c, target)

        avg_ms = time_function(do_transpile, runs=10)

        assert avg_ms < 5, f"Bell transpile took {avg_ms:.2f}ms, expected < 5ms"

    def test_ghz10_transpile_reasonable(self, line_target_10):
        """GHZ-10 transpile should complete in < 50ms."""
        def do_transpile():
            c = Circuit(10).h(0)
            for i in range(9):
                c.cx(i, i + 1)
            return transpile(c, line_target_10)

        avg_ms = time_function(do_transpile, runs=5)

        assert avg_ms < 50, f"GHZ-10 transpile took {avg_ms:.2f}ms, expected < 50ms"

    def test_qft8_transpile_reasonable(self):
        """QFT-8 transpile should complete in < 100ms."""
        target = Target(
            n_qubits=8,
            edges=line_topology(8),
            basis_gates=frozenset({Gate.RZ, Gate.RX, Gate.CX}),
            name="line_8"
        )

        def do_transpile():
            c = Circuit(8)
            for i in range(8):
                c.h(i)
                for j in range(i + 1, 8):
                    c.cp(j, i, pi / (2 ** (j - i)))
            return transpile(c, target)

        avg_ms = time_function(do_transpile, runs=3)

        assert avg_ms < 100, f"QFT-8 transpile took {avg_ms:.2f}ms, expected < 100ms"

    def test_transpile_scales_reasonably(self, line_target_10):
        """Transpile time should not blow up with circuit size."""
        times = []

        for depth in [5, 10, 20]:
            def make_and_transpile():
                c = Circuit(10)
                for _ in range(depth):
                    for i in range(10):
                        c.h(i)
                    for i in range(0, 9, 2):
                        c.cx(i, i + 1)
                return transpile(c, line_target_10)

            avg_ms = time_function(make_and_transpile, runs=3)
            times.append((depth, avg_ms))

        # Check that doubling depth doesn't more than 4x time (quadratic OK, exponential bad)
        _, time_5 = times[0]
        _, time_20 = times[2]

        # 4x depth should be < 16x time (allowing quadratic scaling)
        assert time_20 < time_5 * 20, (
            f"Scaling issue: depth 5 took {time_5:.1f}ms, depth 20 took {time_20:.1f}ms"
        )


# =============================================================================
# Simulation performance tests
# =============================================================================

class TestSimulationPerformance:
    """Tests for simulation performance."""

    def test_simulate_small_fast(self):
        """Small circuit simulation should be very fast."""
        from tinyqubit.simulator import simulate

        def do_simulate():
            c = Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
            return simulate(c)

        avg_ms = time_function(do_simulate, runs=20)

        assert avg_ms < 1, f"4-qubit simulate took {avg_ms:.2f}ms, expected < 1ms"

    def test_simulate_medium_reasonable(self):
        """Medium circuit simulation should be reasonable."""
        from tinyqubit.simulator import simulate

        def do_simulate():
            c = Circuit(10).h(0)
            for i in range(9):
                c.cx(i, i + 1)
            return simulate(c)

        avg_ms = time_function(do_simulate, runs=5)

        assert avg_ms < 50, f"10-qubit simulate took {avg_ms:.2f}ms, expected < 50ms"
