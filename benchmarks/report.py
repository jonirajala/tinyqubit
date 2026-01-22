"""
Results formatting and baseline management for benchmarks.

Provides:
- generate_markdown_table: Create README-compatible results table
- save_baseline: Save results as JSON baseline
- compare_to_baseline: Detect regressions (>5% worse)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_markdown_table(results: list[dict], columns: list[str], title: str = "") -> str:
    """
    Generate a README-compatible markdown table.

    Args:
        results: List of result dictionaries
        columns: List of column keys to include
        title: Optional title for the table

    Returns:
        Markdown-formatted table string
    """
    if not results:
        return ""

    lines = []

    if title:
        lines.append(f"### {title}")
        lines.append("")

    # Header
    header = "| " + " | ".join(columns) + " |"
    lines.append(header)

    # Separator
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(sep)

    # Rows
    for r in results:
        row_vals = [str(r.get(c, "-")) for c in columns]
        row = "| " + " | ".join(row_vals) + " |"
        lines.append(row)

    return "\n".join(lines)


def save_baseline(results: dict, path: str | Path) -> None:
    """
    Save benchmark results as a JSON baseline for regression detection.

    Args:
        results: Dictionary of benchmark results
        path: Output file path
    """
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def load_baseline(path: str | Path) -> dict | None:
    """
    Load a baseline from JSON file.

    Args:
        path: Baseline file path

    Returns:
        Baseline dictionary or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def compare_to_baseline(current: dict, baseline: dict, threshold: float = 0.05) -> list[str]:
    """
    Compare current results to baseline and detect regressions.

    Args:
        current: Current benchmark results
        baseline: Baseline results to compare against
        threshold: Regression threshold (default 5%)

    Returns:
        List of regression messages (empty if no regressions)
    """
    regressions = []
    baseline_results = baseline.get("results", {})

    for category, cat_results in current.items():
        if category not in baseline_results:
            continue

        baseline_cat = baseline_results[category]

        if isinstance(cat_results, dict):
            for key, value in cat_results.items():
                if key not in baseline_cat:
                    continue

                baseline_val = baseline_cat[key]

                # Skip non-numeric values
                if not isinstance(value, (int, float)) or not isinstance(baseline_val, (int, float)):
                    continue

                # Skip zero baselines
                if baseline_val == 0:
                    continue

                # Check for regression (higher is worse for gate counts)
                change = (value - baseline_val) / baseline_val
                if change > threshold:
                    regressions.append(
                        f"{category}.{key}: {baseline_val} -> {value} "
                        f"(+{change*100:.1f}%, threshold {threshold*100:.0f}%)"
                    )

        elif isinstance(cat_results, list):
            # Handle list results (e.g., per-circuit results)
            for i, item in enumerate(cat_results):
                if i >= len(baseline_cat):
                    continue

                baseline_item = baseline_cat[i]

                if isinstance(item, dict) and isinstance(baseline_item, dict):
                    for key, value in item.items():
                        if key not in baseline_item:
                            continue

                        baseline_val = baseline_item[key]

                        if not isinstance(value, (int, float)) or not isinstance(baseline_val, (int, float)):
                            continue

                        if baseline_val == 0:
                            continue

                        change = (value - baseline_val) / baseline_val
                        if change > threshold:
                            name = item.get("name", f"item_{i}")
                            regressions.append(
                                f"{category}[{name}].{key}: {baseline_val} -> {value} "
                                f"(+{change*100:.1f}%)"
                            )

    return regressions


def format_summary(
    optimize_results: list[dict] | None = None,
    routing_results: dict[str, list[dict]] | None = None,
) -> str:
    """
    Format a summary of all benchmark results.

    Args:
        optimize_results: Results from optimization benchmark
        routing_results: Results from routing benchmark (by topology)

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  BENCHMARK SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Optimization summary
    if optimize_results:
        wins = {"tinyqubit": 0, "Qiskit": 0, "tie": 0}
        for r in optimize_results:
            winner = r.get("winner", "-")
            if winner in wins:
                wins[winner] += 1

        lines.append("Optimization:")
        lines.append(f"  tinyqubit wins: {wins['tinyqubit']}")
        lines.append(f"  Qiskit wins:    {wins['Qiskit']}")
        lines.append(f"  Ties:           {wins['tie']}")
        lines.append("")

    # Routing summary
    if routing_results:
        lines.append("Routing (by topology):")
        total_tq = 0
        total_qk = 0
        total_tie = 0

        for topo_name, results in routing_results.items():
            wins = {"tinyqubit": 0, "Qiskit": 0, "tie": 0}
            for r in results:
                winner = r.get("winner", "-")
                if winner in wins:
                    wins[winner] += 1

            lines.append(f"  {topo_name}: TQ={wins['tinyqubit']}, QK={wins['Qiskit']}, tie={wins['tie']}")
            total_tq += wins["tinyqubit"]
            total_qk += wins["Qiskit"]
            total_tie += wins["tie"]

        lines.append(f"  Total: TQ={total_tq}, QK={total_qk}, tie={total_tie}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def collect_results_for_baseline(
    optimize_results: list[dict] | None = None,
    routing_results: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """
    Collect all benchmark results into a single dict for baseline saving.

    Args:
        optimize_results: Results from optimization benchmark
        routing_results: Results from routing benchmark

    Returns:
        Combined results dictionary
    """
    return {
        "optimization": optimize_results or [],
        "routing": routing_results or {},
    }
