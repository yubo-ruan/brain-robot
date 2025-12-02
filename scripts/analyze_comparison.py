#!/usr/bin/env python3
"""Analyze oracle vs learned perception comparison results."""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path("logs/phase4_comparison")

    oracle_results = defaultdict(dict)  # task_id -> {successes, total}
    learned_results = defaultdict(dict)  # task_id -> {successes, total}

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            data = json.load(f)

        task_id = data["task_id"]
        perception = data.get("perception", "oracle")
        successes = data["successes"]
        total = data["n_episodes"]

        if "learned" in run_dir.name:
            learned_results[task_id] = {"successes": successes, "total": total}
        else:
            oracle_results[task_id] = {"successes": successes, "total": total}

    print("=" * 60)
    print("ORACLE vs LEARNED PERCEPTION COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Task':<6} {'Oracle':<12} {'Learned':<12} {'Diff':<8}")
    print("-" * 38)

    oracle_total_success = 0
    oracle_total = 0
    learned_total_success = 0
    learned_total = 0

    for task_id in sorted(set(oracle_results.keys()) | set(learned_results.keys())):
        oracle = oracle_results.get(task_id, {"successes": 0, "total": 0})
        learned = learned_results.get(task_id, {"successes": 0, "total": 0})

        oracle_rate = oracle["successes"] / oracle["total"] * 100 if oracle["total"] > 0 else 0
        learned_rate = learned["successes"] / learned["total"] * 100 if learned["total"] > 0 else 0
        diff = learned_rate - oracle_rate

        oracle_total_success += oracle["successes"]
        oracle_total += oracle["total"]
        learned_total_success += learned["successes"]
        learned_total += learned["total"]

        print(f"{task_id:<6} {oracle_rate:>5.0f}%       {learned_rate:>5.0f}%       {diff:+.0f}%")

    print("-" * 38)
    oracle_overall = oracle_total_success / oracle_total * 100 if oracle_total > 0 else 0
    learned_overall = learned_total_success / learned_total * 100 if learned_total > 0 else 0
    diff_overall = learned_overall - oracle_overall

    print(f"{'TOTAL':<6} {oracle_overall:>5.1f}%       {learned_overall:>5.1f}%       {diff_overall:+.1f}%")
    print(f"       ({oracle_total_success}/{oracle_total})      ({learned_total_success}/{learned_total})")
    print()

    # Statistical note
    print("=" * 60)
    print("STATISTICAL NOTE")
    print("=" * 60)
    n = oracle_total  # same as learned_total
    if n > 0:
        # Standard error of difference between two proportions
        import math
        p1 = oracle_overall / 100
        p2 = learned_overall / 100
        se = math.sqrt((p1 * (1 - p1) / n) + (p2 * (1 - p2) / n))
        print(f"Sample size: {n} episodes total")
        print(f"SE of difference: ±{se*100:.1f}%")
        print(f"95% CI for difference: [{diff_overall - 1.96*se*100:.1f}%, {diff_overall + 1.96*se*100:.1f}%]")
        if abs(diff_overall) < 1.96 * se * 100:
            print("Conclusion: Difference NOT statistically significant at p<0.05")
        else:
            print("Conclusion: Difference IS statistically significant at p<0.05")

if __name__ == "__main__":
    main()
