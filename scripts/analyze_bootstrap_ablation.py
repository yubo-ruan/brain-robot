#!/usr/bin/env python3
"""Analyze bootstrap ablation results: Oracle vs Cold-Start."""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path("logs/bootstrap_ablation")

    if not results_dir.exists():
        print(f"No results found at {results_dir}")
        return

    oracle_results = {}  # task_id -> {successes, total}
    cold_results = {}  # task_id -> {successes, total}

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            data = json.load(f)

        task_id = data["task_id"]
        successes = data["successes"]
        total = data["n_episodes"]
        bootstrap = data.get("config", {}).get("bootstrap", "oracle")

        if "cold" in run_dir.name or bootstrap == "cold":
            cold_results[task_id] = {"successes": successes, "total": total}
        else:
            oracle_results[task_id] = {"successes": successes, "total": total}

    print("=" * 70)
    print("BOOTSTRAP ABLATION: ORACLE vs COLD-START")
    print("=" * 70)
    print()
    print(f"{'Task':<6} {'Oracle':<15} {'Cold-Start':<15} {'Diff':<8} {'Impact'}")
    print("-" * 60)

    oracle_total_success = 0
    oracle_total = 0
    cold_total_success = 0
    cold_total = 0

    for task_id in sorted(set(oracle_results.keys()) | set(cold_results.keys())):
        oracle = oracle_results.get(task_id, {"successes": 0, "total": 0})
        cold = cold_results.get(task_id, {"successes": 0, "total": 0})

        oracle_rate = oracle["successes"] / oracle["total"] * 100 if oracle["total"] > 0 else 0
        cold_rate = cold["successes"] / cold["total"] * 100 if cold["total"] > 0 else 0
        diff = cold_rate - oracle_rate

        oracle_total_success += oracle["successes"]
        oracle_total += oracle["total"]
        cold_total_success += cold["successes"]
        cold_total += cold["total"]

        # Impact assessment
        if diff < -30:
            impact = "SEVERE"
        elif diff < -10:
            impact = "MODERATE"
        elif diff < 0:
            impact = "MINOR"
        else:
            impact = "OK"

        print(f"{task_id:<6} {oracle_rate:>5.0f}% ({oracle['successes']:2}/{oracle['total']:2})  "
              f"{cold_rate:>5.0f}% ({cold['successes']:2}/{cold['total']:2})  "
              f"{diff:+6.0f}%  {impact}")

    print("-" * 60)
    oracle_overall = oracle_total_success / oracle_total * 100 if oracle_total > 0 else 0
    cold_overall = cold_total_success / cold_total * 100 if cold_total > 0 else 0
    diff_overall = cold_overall - oracle_overall

    print(f"{'TOTAL':<6} {oracle_overall:>5.1f}% ({oracle_total_success:2}/{oracle_total:2})  "
          f"{cold_overall:>5.1f}% ({cold_total_success:2}/{cold_total:2})  "
          f"{diff_overall:+6.1f}%")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(diff_overall) > 20:
        print("Oracle bootstrap provides SIGNIFICANT advantage over cold-start.")
        print(f"Cold-start drops overall success by {abs(diff_overall):.0f}%.")
    elif abs(diff_overall) > 10:
        print("Oracle bootstrap provides MODERATE advantage over cold-start.")
    else:
        print("Oracle bootstrap provides MINIMAL advantage over cold-start.")

    print()
    print("Per-Task Analysis:")
    for task_id in sorted(cold_results.keys()):
        oracle = oracle_results.get(task_id, {"successes": 0, "total": 0})
        cold = cold_results.get(task_id, {"successes": 0, "total": 0})
        oracle_rate = oracle["successes"] / oracle["total"] * 100 if oracle["total"] > 0 else 0
        cold_rate = cold["successes"] / cold["total"] * 100 if cold["total"] > 0 else 0
        diff = cold_rate - oracle_rate

        if diff < -30:
            print(f"  Task {task_id}: CRITICAL - cold-start drops {abs(diff):.0f}%")
            print(f"           (likely due to occluded/hard-to-detect objects)")


if __name__ == "__main__":
    main()
