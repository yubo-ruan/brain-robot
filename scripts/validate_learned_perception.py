#!/usr/bin/env python3
"""Validate learned perception against oracle ground truth.

Computes:
1. Position error: distance between learned and oracle 3D positions
2. Spatial relation agreement: ON/INSIDE predictions match oracle
3. Per-class breakdown of errors

This is the critical validation step before trusting learned perception
in the full control loop.

Usage:
    python scripts/validate_learned_perception.py --task-suite libero_spatial --n-episodes 5
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class PerceptionComparison:
    """Single frame comparison between learned and oracle."""

    # Object position errors: {instance_id: error_meters}
    position_errors: Dict[str, float]

    # Spatial relation matches
    on_matches: int
    on_total: int
    inside_matches: int
    inside_total: int

    # Missing/extra detections
    missed_objects: List[str]  # In oracle but not learned
    extra_objects: List[str]   # In learned but not oracle

    # Metadata
    task_id: int
    episode_idx: int
    step_idx: int


def compare_perceptions(oracle_result, learned_result, task_id, episode_idx, step_idx):
    """Compare learned perception to oracle ground truth.

    Args:
        oracle_result: PerceptionResult from OraclePerception
        learned_result: PerceptionResult from LearnedPerception
        task_id, episode_idx, step_idx: Metadata

    Returns:
        PerceptionComparison with error metrics
    """
    position_errors = {}
    missed_objects = []
    extra_objects = []

    # Compare object positions
    oracle_objects = set(oracle_result.objects.keys())
    learned_objects = set(learned_result.objects.keys())

    # Objects in oracle but not learned = missed
    missed_objects = list(oracle_objects - learned_objects)

    # Objects in learned but not oracle = extra
    extra_objects = list(learned_objects - oracle_objects)

    # For objects in both, compute position error
    common_objects = oracle_objects & learned_objects
    for obj_id in common_objects:
        oracle_pos = oracle_result.objects[obj_id][:3]
        learned_pos = learned_result.objects[obj_id][:3]
        error = np.linalg.norm(oracle_pos - learned_pos)
        position_errors[obj_id] = float(error)

    # Compare spatial relations: ON
    on_matches = 0
    on_total = 0
    for obj_id, surface in oracle_result.on.items():
        on_total += 1
        if obj_id in learned_result.on:
            if learned_result.on[obj_id] == surface:
                on_matches += 1

    # Compare spatial relations: INSIDE
    inside_matches = 0
    inside_total = 0
    for obj_id, container in oracle_result.inside.items():
        inside_total += 1
        if obj_id in learned_result.inside:
            if learned_result.inside[obj_id] == container:
                inside_matches += 1

    return PerceptionComparison(
        position_errors=position_errors,
        on_matches=on_matches,
        on_total=on_total,
        inside_matches=inside_matches,
        inside_total=inside_total,
        missed_objects=missed_objects,
        extra_objects=extra_objects,
        task_id=task_id,
        episode_idx=episode_idx,
        step_idx=step_idx,
    )


def run_validation(
    task_suite: str,
    task_ids: List[int],
    n_episodes: int,
    n_steps: int,
    model_path: str,
    seed: int = 42,
):
    """Run validation comparing learned vs oracle perception.

    Args:
        task_suite: LIBERO task suite name
        task_ids: List of task IDs to evaluate
        n_episodes: Episodes per task
        n_steps: Steps per episode
        model_path: Path to YOLO model
        seed: Random seed
    """
    # Import here to avoid loading LIBERO if just checking args
    from libero.libero import benchmark, get_libero_path
    from brain_robot.perception import OraclePerception, LearnedPerception

    print("=" * 70)
    print("LEARNED PERCEPTION VALIDATION")
    print("=" * 70)
    print(f"Task suite: {task_suite}")
    print(f"Task IDs: {task_ids}")
    print(f"Episodes per task: {n_episodes}")
    print(f"Steps per episode: {n_steps}")
    print(f"Model: {model_path}")
    print("=" * 70)

    # Initialize LIBERO benchmark
    bench = benchmark.get_benchmark(task_suite)()
    n_tasks = bench.get_num_tasks()

    print(f"\nBenchmark has {n_tasks} tasks")

    # Initialize perception systems
    oracle = OraclePerception()
    learned = LearnedPerception(model_path=model_path)

    # Warm up detector
    print("Warming up YOLO detector...")
    learned.detector.warmup()

    # Collect all comparisons
    all_comparisons: List[PerceptionComparison] = []

    # Per-class error tracking
    class_errors = defaultdict(list)  # class_name -> [errors]

    for task_id in task_ids:
        if task_id >= n_tasks:
            print(f"Warning: Task {task_id} >= n_tasks {n_tasks}, skipping")
            continue

        task = bench.get_task(task_id)
        task_name = task.name
        print(f"\n{'='*70}")
        print(f"Task {task_id}: {task_name}")
        print("=" * 70)

        # Get task environment
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }

        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_args)

        for episode_idx in range(n_episodes):
            # Reset environment
            env.seed(seed + task_id * 1000 + episode_idx)
            env.reset()

            # Get initial oracle perception for bootstrap
            oracle_result = oracle.perceive(env)

            # Bootstrap learned perception with oracle knowledge
            learned.reset()
            learned.bootstrap_from_oracle(oracle_result, env)

            print(f"  Episode {episode_idx}: ", end="", flush=True)

            episode_errors = []

            for step_idx in range(n_steps):
                # Random action for variety (small random motions)
                # LIBERO action space is typically 7D: dx, dy, dz, dax, day, daz, gripper
                action = np.random.uniform(-0.1, 0.1, 7)
                action[6] = 0  # Keep gripper unchanged

                # Step environment
                obs, reward, done, info = env.step(action)

                # Get perceptions
                oracle_result = oracle.perceive(env)
                learned_result = learned.perceive(env)

                # Compare
                comparison = compare_perceptions(
                    oracle_result, learned_result,
                    task_id, episode_idx, step_idx
                )
                all_comparisons.append(comparison)

                # Track errors
                for obj_id, error in comparison.position_errors.items():
                    episode_errors.append(error)

                    # Track by class
                    from brain_robot.perception.data_collection.collector import instance_id_to_class
                    obj_class = instance_id_to_class(obj_id)
                    class_errors[obj_class].append(error)

                if done:
                    break

            # Print episode summary
            if episode_errors:
                mean_err = np.mean(episode_errors) * 100  # cm
                max_err = np.max(episode_errors) * 100
                print(f"mean={mean_err:.1f}cm, max={max_err:.1f}cm")
            else:
                print("no common objects")

        env.close()

    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_errors = []
    total_on_matches = 0
    total_on = 0
    total_inside_matches = 0
    total_inside = 0
    total_missed = 0
    total_extra = 0

    for comp in all_comparisons:
        all_errors.extend(comp.position_errors.values())
        total_on_matches += comp.on_matches
        total_on += comp.on_total
        total_inside_matches += comp.inside_matches
        total_inside += comp.inside_total
        total_missed += len(comp.missed_objects)
        total_extra += len(comp.extra_objects)

    if all_errors:
        errors_cm = np.array(all_errors) * 100  # Convert to cm
        print(f"\nPosition Error (cm):")
        print(f"  Mean:   {np.mean(errors_cm):.2f}")
        print(f"  Median: {np.median(errors_cm):.2f}")
        print(f"  95th:   {np.percentile(errors_cm, 95):.2f}")
        print(f"  Max:    {np.max(errors_cm):.2f}")

        # Histogram
        print(f"\nError Distribution:")
        bins = [0, 1, 2, 5, 10, 20, 50, 100]
        for i in range(len(bins) - 1):
            count = np.sum((errors_cm >= bins[i]) & (errors_cm < bins[i+1]))
            pct = 100 * count / len(errors_cm)
            print(f"  {bins[i]:3d}-{bins[i+1]:3d}cm: {count:4d} ({pct:5.1f}%)")

    print(f"\nSpatial Relations:")
    if total_on > 0:
        print(f"  ON accuracy:     {100*total_on_matches/total_on:.1f}% ({total_on_matches}/{total_on})")
    if total_inside > 0:
        print(f"  INSIDE accuracy: {100*total_inside_matches/total_inside:.1f}% ({total_inside_matches}/{total_inside})")

    print(f"\nDetection Coverage:")
    print(f"  Missed objects: {total_missed}")
    print(f"  Extra objects:  {total_extra}")

    # Per-class breakdown
    print(f"\nPer-Class Position Error (cm):")
    for cls in sorted(class_errors.keys()):
        errors = np.array(class_errors[cls]) * 100
        if len(errors) > 0:
            print(f"  {cls:15s}: mean={np.mean(errors):5.2f}, "
                  f"95th={np.percentile(errors, 95):5.2f}, n={len(errors)}")

    # Save results
    results = {
        "task_suite": task_suite,
        "task_ids": task_ids,
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "model_path": model_path,
        "position_error_cm": {
            "mean": float(np.mean(errors_cm)) if all_errors else 0,
            "median": float(np.median(errors_cm)) if all_errors else 0,
            "p95": float(np.percentile(errors_cm, 95)) if all_errors else 0,
            "max": float(np.max(errors_cm)) if all_errors else 0,
        },
        "spatial_relations": {
            "on_accuracy": total_on_matches / total_on if total_on > 0 else 0,
            "inside_accuracy": total_inside_matches / total_inside if total_inside > 0 else 0,
        },
        "detection": {
            "total_missed": total_missed,
            "total_extra": total_extra,
        },
        "per_class_error_cm": {
            cls: float(np.mean(np.array(errs) * 100))
            for cls, errs in class_errors.items()
        },
    }

    # Write results
    import json
    output_dir = Path("logs/perception_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"validation_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Learned Perception")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        help="LIBERO task suite")
    parser.add_argument("--task-ids", type=str, default="0,1,2",
                        help="Comma-separated task IDs")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Episodes per task")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Steps per episode")
    parser.add_argument("--model", type=str, default="models/yolo_libero.pt",
                        help="Path to YOLO model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Parse task IDs
    task_ids = [int(x) for x in args.task_ids.split(",")]

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first with train_yolo_detector.py")
        return 1

    # Run validation
    run_validation(
        task_suite=args.task_suite,
        task_ids=task_ids,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        model_path=args.model,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
