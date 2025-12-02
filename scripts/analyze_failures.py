#!/usr/bin/env python3
"""Analyze failure modes from Phase 5 evaluation logs.

Usage:
    python scripts/analyze_failures.py --log-dir logs/phase5_full_evaluation
    python scripts/analyze_failures.py --log-dir logs/phase5_full_evaluation --by-task
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_robot.analysis.failure_classifier import FailureClassifier, FailureMode
import json
import argparse


def analyze_by_task(log_dir: Path, classifier: FailureClassifier) -> dict:
    """Analyze failures separately for each task.

    Args:
        log_dir: Root log directory.
        classifier: FailureClassifier instance.

    Returns:
        Dictionary with per-task analysis.
    """
    results = {}

    # Find all task directories
    task_dirs = sorted(log_dir.glob("*_task*"))

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue

        # Extract perception mode and task ID from directory name
        dir_name = task_dir.name
        if "oracle" in dir_name:
            perception = "oracle"
        elif "learned" in dir_name:
            perception = "learned"
        else:
            continue

        # Extract task ID
        task_id = dir_name.split("task")[-1].split("_")[0]

        key = f"{perception}_task{task_id}"

        # Analyze this task directory
        summary = classifier.analyze_log_directory(task_dir)
        results[key] = summary

    return results


def print_task_summary(results: dict):
    """Print per-task summary table."""
    print("\n" + "=" * 80)
    print("PER-TASK FAILURE ANALYSIS")
    print("=" * 80)

    # Group by perception mode
    oracle_results = {k: v for k, v in results.items() if k.startswith("oracle")}
    learned_results = {k: v for k, v in results.items() if k.startswith("learned")}

    # Print table header
    print("\n--- Oracle Perception ---")
    print(f"{'Task':<12} {'Success':<10} {'Approach':<12} {'Grasp':<12} {'Move':<12} {'Other':<12}")
    print("-" * 70)

    for key in sorted(oracle_results.keys()):
        summary = oracle_results[key]
        task_id = key.split("task")[-1]
        success_rate = summary["success_rate"] * 100
        fm = summary["failure_modes"]

        approach = fm.get("approach_timeout", {}).get("count", 0)
        grasp = fm.get("grasp_miss", {}).get("count", 0)
        move = fm.get("move_timeout", {}).get("count", 0)
        other = summary["failures"] - approach - grasp - move

        print(f"Task {task_id:<6} {success_rate:>6.1f}%   {approach:>8}     {grasp:>8}     {move:>8}     {other:>8}")

    print("\n--- Learned Perception (Cold Bootstrap) ---")
    print(f"{'Task':<12} {'Success':<10} {'Approach':<12} {'Grasp':<12} {'Move':<12} {'Percept':<12}")
    print("-" * 70)

    for key in sorted(learned_results.keys()):
        summary = learned_results[key]
        task_id = key.split("task")[-1]
        success_rate = summary["success_rate"] * 100
        fm = summary["failure_modes"]

        approach = fm.get("approach_timeout", {}).get("count", 0)
        grasp = fm.get("grasp_miss", {}).get("count", 0)
        move = fm.get("move_timeout", {}).get("count", 0)
        perception = (
            fm.get("perception_miss", {}).get("count", 0) +
            fm.get("perception_wrong", {}).get("count", 0)
        )

        print(f"Task {task_id:<6} {success_rate:>6.1f}%   {approach:>8}     {grasp:>8}     {move:>8}     {perception:>8}")


def main():
    parser = argparse.ArgumentParser(description="Analyze failure modes from evaluation logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/phase5_full_evaluation",
        help="Directory containing evaluation logs"
    )
    parser.add_argument(
        "--by-task",
        action="store_true",
        help="Analyze separately for each task"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for analysis results"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    classifier = FailureClassifier()

    if args.by_task:
        results = analyze_by_task(log_dir, classifier)
        print_task_summary(results)

        # Also print overall summary
        print("\n--- Overall Summary ---")
        all_summary = classifier.analyze_log_directory(log_dir)
        classifier.print_summary(all_summary)

        if args.output:
            output_data = {
                "by_task": results,
                "overall": all_summary
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")
    else:
        summary = classifier.analyze_log_directory(log_dir)
        classifier.print_summary(summary)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
