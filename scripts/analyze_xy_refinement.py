#!/usr/bin/env python3
"""Analyze XY refinement distribution from logged episodes.

Extracts xy_error_before and xy_error_after from GraspSkill logs
to validate that XY refinement is working correctly.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_xy_refinement_data(log_dir: Path) -> dict:
    """Extract XY refinement data from all episode logs."""
    data = {
        "xy_error_before": [],
        "xy_error_after": [],
        "steps": [],
        "converged": [],
        "skipped": [],
        "grasp_success": [],
        "episode_success": [],
    }

    episode_files = list(log_dir.rglob("episode_*.json"))
    print(f"Found {len(episode_files)} episode files in {log_dir}")

    for ep_file in sorted(episode_files):
        try:
            with open(ep_file) as f:
                ep_data = json.load(f)

            # Find GraspObject skill in sequence
            for skill in ep_data.get("skill_sequence", []):
                if skill.get("skill") == "GraspObject":
                    xy_ref = skill.get("info", {}).get("xy_refinement", {})
                    if xy_ref:
                        data["xy_error_before"].append(xy_ref.get("xy_error_before", 0))
                        data["xy_error_after"].append(xy_ref.get("xy_error_after", 0))
                        data["steps"].append(xy_ref.get("steps", 0))
                        data["converged"].append(xy_ref.get("converged", False))
                        data["skipped"].append(xy_ref.get("skipped", False))
                        data["grasp_success"].append(skill.get("success", False))
                        data["episode_success"].append(ep_data.get("success", False))
        except Exception as e:
            print(f"Error reading {ep_file}: {e}")

    return data


def analyze_distribution(data: dict) -> dict:
    """Analyze the XY refinement distribution."""
    if not data["xy_error_before"]:
        return {"error": "No data found"}

    before = np.array(data["xy_error_before"]) * 100  # Convert to cm
    after = np.array(data["xy_error_after"]) * 100
    steps = np.array(data["steps"])
    converged = np.array(data["converged"])
    skipped = np.array(data["skipped"])
    grasp_success = np.array(data["grasp_success"])
    episode_success = np.array(data["episode_success"])

    # Cases where refinement was actually performed
    refined_mask = ~skipped

    analysis = {
        "total_episodes": len(before),
        "total_grasps": len(before),

        # Overall distribution
        "before_error_cm": {
            "mean": float(np.mean(before)),
            "std": float(np.std(before)),
            "min": float(np.min(before)),
            "max": float(np.max(before)),
            "median": float(np.median(before)),
            "p90": float(np.percentile(before, 90)),
        },
        "after_error_cm": {
            "mean": float(np.mean(after)),
            "std": float(np.std(after)),
            "min": float(np.min(after)),
            "max": float(np.max(after)),
            "median": float(np.median(after)),
            "p90": float(np.percentile(after, 90)),
        },

        # Refinement stats
        "refinement_skipped_count": int(np.sum(skipped)),
        "refinement_skipped_pct": float(np.mean(skipped) * 100),
        "refinement_performed_count": int(np.sum(refined_mask)),
        "convergence_rate": float(np.mean(converged) * 100),

        # Success correlation
        "grasp_success_rate": float(np.mean(grasp_success) * 100),
        "episode_success_rate": float(np.mean(episode_success) * 100),
    }

    # Stats for cases where refinement was performed
    if np.sum(refined_mask) > 0:
        before_refined = before[refined_mask]
        after_refined = after[refined_mask]
        steps_refined = steps[refined_mask]
        improvement = before_refined - after_refined

        analysis["refined_cases"] = {
            "count": int(np.sum(refined_mask)),
            "before_error_cm": {
                "mean": float(np.mean(before_refined)),
                "std": float(np.std(before_refined)),
                "min": float(np.min(before_refined)),
                "max": float(np.max(before_refined)),
            },
            "after_error_cm": {
                "mean": float(np.mean(after_refined)),
                "std": float(np.std(after_refined)),
                "min": float(np.min(after_refined)),
                "max": float(np.max(after_refined)),
            },
            "improvement_cm": {
                "mean": float(np.mean(improvement)),
                "std": float(np.std(improvement)),
                "min": float(np.min(improvement)),
                "max": float(np.max(improvement)),
            },
            "steps": {
                "mean": float(np.mean(steps_refined)),
                "std": float(np.std(steps_refined)),
                "min": float(np.min(steps_refined)),
                "max": float(np.max(steps_refined)),
            },
        }

        # Cases with large pre-error (>5cm)
        large_error_mask = before_refined > 5.0
        if np.sum(large_error_mask) > 0:
            analysis["large_error_cases"] = {
                "count": int(np.sum(large_error_mask)),
                "before_error_cm_mean": float(np.mean(before_refined[large_error_mask])),
                "after_error_cm_mean": float(np.mean(after_refined[large_error_mask])),
                "improvement_cm_mean": float(np.mean(improvement[large_error_mask])),
            }

    # Failure analysis
    failed_grasps = ~grasp_success
    if np.sum(failed_grasps) > 0:
        analysis["failed_grasps"] = {
            "count": int(np.sum(failed_grasps)),
            "after_error_cm_mean": float(np.mean(after[failed_grasps])),
            "converged_pct": float(np.mean(converged[failed_grasps]) * 100),
        }

    return analysis


def print_histogram(values: np.ndarray, bins: list, title: str):
    """Print ASCII histogram."""
    print(f"\n{title}")
    print("=" * 50)
    hist, edges = np.histogram(values, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    for i, count in enumerate(hist):
        bar = "█" * int(count / max_count * 30)
        label = f"{edges[i]:.1f}-{edges[i+1]:.1f}cm"
        print(f"{label:12s} | {bar} ({count})")


def main():
    # Look for logs in multiple locations
    log_dirs = [
        Path("/workspace/brain_robot/logs/xy_refine_cold_full"),
        Path("/workspace/brain_robot/logs/xy_refine_cold_test"),
        Path("/workspace/brain_robot/logs/cold_bootstrap_comprehensive"),
        Path("/workspace/brain_robot/logs/cold_bootstrap_multi_task"),
    ]

    all_data = {
        "xy_error_before": [],
        "xy_error_after": [],
        "steps": [],
        "converged": [],
        "skipped": [],
        "grasp_success": [],
        "episode_success": [],
    }

    for log_dir in log_dirs:
        if log_dir.exists():
            print(f"\nProcessing: {log_dir}")
            data = extract_xy_refinement_data(log_dir)
            for key in all_data:
                all_data[key].extend(data[key])

    if not all_data["xy_error_before"]:
        print("No XY refinement data found!")
        sys.exit(1)

    # Analyze
    analysis = analyze_distribution(all_data)

    # Print results
    print("\n" + "=" * 60)
    print("XY REFINEMENT DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Grasp Attempts: {analysis['total_grasps']}")
    print(f"Grasp Success Rate: {analysis['grasp_success_rate']:.1f}%")
    print(f"Episode Success Rate: {analysis['episode_success_rate']:.1f}%")

    print(f"\n--- Pre-Refinement XY Error (all cases) ---")
    b = analysis["before_error_cm"]
    print(f"  Mean: {b['mean']:.2f} cm")
    print(f"  Std:  {b['std']:.2f} cm")
    print(f"  Range: {b['min']:.2f} - {b['max']:.2f} cm")
    print(f"  Median: {b['median']:.2f} cm")
    print(f"  P90: {b['p90']:.2f} cm")

    print(f"\n--- Post-Refinement XY Error (all cases) ---")
    a = analysis["after_error_cm"]
    print(f"  Mean: {a['mean']:.2f} cm")
    print(f"  Std:  {a['std']:.2f} cm")
    print(f"  Range: {a['min']:.2f} - {a['max']:.2f} cm")
    print(f"  Median: {a['median']:.2f} cm")
    print(f"  P90: {a['p90']:.2f} cm")

    print(f"\n--- Refinement Stats ---")
    print(f"  Skipped (already aligned): {analysis['refinement_skipped_count']} ({analysis['refinement_skipped_pct']:.1f}%)")
    print(f"  Performed: {analysis['refinement_performed_count']}")
    print(f"  Convergence Rate: {analysis['convergence_rate']:.1f}%")

    if "refined_cases" in analysis:
        rc = analysis["refined_cases"]
        print(f"\n--- Cases Where Refinement Was Performed ({rc['count']}) ---")
        print(f"  Before: {rc['before_error_cm']['mean']:.2f} ± {rc['before_error_cm']['std']:.2f} cm")
        print(f"  After:  {rc['after_error_cm']['mean']:.2f} ± {rc['after_error_cm']['std']:.2f} cm")
        print(f"  Improvement: {rc['improvement_cm']['mean']:.2f} ± {rc['improvement_cm']['std']:.2f} cm")
        print(f"  Steps: {rc['steps']['mean']:.1f} ± {rc['steps']['std']:.1f}")

    if "large_error_cases" in analysis:
        lc = analysis["large_error_cases"]
        print(f"\n--- Large Error Cases (>5cm before) ({lc['count']}) ---")
        print(f"  Before: {lc['before_error_cm_mean']:.2f} cm")
        print(f"  After:  {lc['after_error_cm_mean']:.2f} cm")
        print(f"  Improvement: {lc['improvement_cm_mean']:.2f} cm")

    if "failed_grasps" in analysis:
        fg = analysis["failed_grasps"]
        print(f"\n--- Failed Grasps ({fg['count']}) ---")
        print(f"  Final XY Error: {fg['after_error_cm_mean']:.2f} cm")
        print(f"  Converged: {fg['converged_pct']:.1f}%")

    # Print histograms
    before = np.array(all_data["xy_error_before"]) * 100
    after = np.array(all_data["xy_error_after"]) * 100
    bins = [0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 100]

    print_histogram(before, bins, "Pre-Refinement XY Error Distribution")
    print_histogram(after, bins, "Post-Refinement XY Error Distribution")

    # Save analysis to JSON
    output_file = Path("/workspace/brain_robot/logs/xy_refinement_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    main()
