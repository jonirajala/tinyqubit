"""
Run all tinyqubit benchmarks and generate summary report.

Run: python benchmarks/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS = ["optimize", "routing", "depth", "compile_time", "simulation"]


def main():
    benchmarks_dir = Path(__file__).parent

    start_time = time.time()
    results = {}

    for name in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}\n", flush=True)

        bench_start = time.time()
        result = subprocess.run(
            [sys.executable, benchmarks_dir / f"{name}.py"],
            capture_output=False,
        )
        bench_time = time.time() - bench_start

        results[name] = {
            "returncode": result.returncode,
            "time": bench_time,
        }

    # Summary
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print()
    print(f"{'Benchmark':<16} {'Status':>10} {'Time (s)':>12}")
    print("-" * 40)

    for name, data in results.items():
        status = "PASS" if data["returncode"] == 0 else "FAIL"
        print(f"{name:<16} {status:>10} {data['time']:>12.2f}")

    print("-" * 40)
    print(f"{'Total':<16} {'':<10} {total_time:>12.2f}")
    print()

    # Check if all passed
    all_passed = all(r["returncode"] == 0 for r in results.values())
    if all_passed:
        print("All benchmarks completed successfully.")
    else:
        failed = [n for n, r in results.items() if r["returncode"] != 0]
        print(f"Some benchmarks failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
