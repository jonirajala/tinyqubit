"""
Run all tinyqubit benchmarks.

Run: python benchmarks/run_all.py
"""
import subprocess
import sys
from pathlib import Path

BENCHMARKS = ["optimize", "routing", "depth", "compile_time", "simulation"]

def main():
    benchmarks_dir = Path(__file__).parent

    for name in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}\n", flush=True)
        subprocess.run([sys.executable, benchmarks_dir / f"{name}.py"])

if __name__ == "__main__":
    main()
