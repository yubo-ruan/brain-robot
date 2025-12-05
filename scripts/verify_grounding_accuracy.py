#!/usr/bin/env python3
"""Verify grounding accuracy against ground truth.

This script runs the Qwen semantic grounder on LIBERO spatial tasks
and compares the output against known ground-truth labels.

Usage:
    python scripts/verify_grounding_accuracy.py [--task-suite libero_spatial] [--n-episodes 5]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.grounding import QwenSemanticGrounder, GroundingMetrics, enrich_objects
from src.grounding.ground_truth import (
    get_ground_truth,
    get_all_ground_truth,
    verify_grounding,
    GroundingGroundTruth,
)
from src.perception.oracle import OraclePerception
from src.world_model.state import WorldState
from src.utils.seeds import set_global_seed, get_episode_seed


def make_libero_env(task_suite: str, task_id: int):
    """Create LIBERO environment."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(task_suite)()
    task = benchmark.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def verify_single_episode(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    grounder: QwenSemanticGrounder,
    ground_truth: GroundingGroundTruth,
    metrics: GroundingMetrics,
    task_id_str: str,
) -> Dict[str, Any]:
    """Run grounding on a single episode and verify against ground truth."""
    # Get perception
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)

    # Enrich objects
    enriched = enrich_objects(world_state)

    # Run grounder
    grounding_result = grounder.ground(
        task_description=task_description,
        objects=enriched,
        metrics=metrics,
        task_id=task_id_str,
    )

    # Verify against ground truth
    if grounding_result.valid:
        verification = verify_grounding(
            grounding_source=grounding_result.source_object,
            grounding_target=grounding_result.target_location,
            ground_truth=ground_truth,
        )
    else:
        verification = {
            "source_correct": False,
            "target_correct": False,
            "both_correct": False,
            "details": f"Grounding failed: {grounding_result.error}",
        }

    # Find the spatial context of the chosen source
    source_spatial = None
    for obj in enriched:
        if obj.id == grounding_result.source_object:
            source_spatial = obj.spatial_text
            break

    return {
        "grounding_valid": grounding_result.valid,
        "grounding_source": grounding_result.source_object,
        "grounding_target": grounding_result.target_location,
        "grounding_confidence": grounding_result.confidence,
        "grounding_ambiguous": grounding_result.ambiguous,
        "source_spatial_context": source_spatial,
        "ground_truth_source": ground_truth.true_source,
        "ground_truth_target": ground_truth.true_target,
        "source_correct": verification["source_correct"],
        "target_correct": verification["target_correct"],
        "both_correct": verification["both_correct"],
        "verification_details": verification["details"],
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Grounding Accuracy")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Episodes per task (for seed variation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/grounding_verification")
    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"verification_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GROUNDING ACCURACY VERIFICATION")
    print("=" * 70)
    print(f"Task Suite: {args.task_suite}")
    print(f"Episodes per task: {args.n_episodes}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Get all ground truth
    ground_truths = get_all_ground_truth(args.task_suite)
    if not ground_truths:
        print(f"No ground truth available for {args.task_suite}")
        return 1

    # Initialize components
    grounder = QwenSemanticGrounder(temperature=0.1, max_retries=2)
    perception = OraclePerception()
    metrics = GroundingMetrics()

    # Results storage
    all_results: List[Dict[str, Any]] = []
    task_summaries: List[Dict[str, Any]] = []

    # Process each task
    for gt in ground_truths:
        print(f"\n{'='*70}")
        print(f"Task {gt.task_id}: {gt.task_description}")
        print(f"Ground truth: source={gt.true_source}, target={gt.true_target}")
        print(f"Notes: {gt.notes}")
        print("=" * 70)

        # Create environment
        env, task_description = make_libero_env(args.task_suite, gt.task_id)
        task_id_str = f"{args.task_suite}_{gt.task_id}"

        task_results = []
        correct_count = 0

        for episode_idx in range(args.n_episodes):
            episode_seed = get_episode_seed(args.seed, episode_idx)
            set_global_seed(episode_seed, env)
            env.reset()

            world_state = WorldState()

            result = verify_single_episode(
                env=env,
                task_description=task_description,
                world_state=world_state,
                perception=perception,
                grounder=grounder,
                ground_truth=gt,
                metrics=metrics,
                task_id_str=task_id_str,
            )

            result["task_id"] = gt.task_id
            result["episode_idx"] = episode_idx
            result["seed"] = episode_seed

            task_results.append(result)
            all_results.append(result)

            status = "✓" if result["both_correct"] else "✗"
            print(f"  Episode {episode_idx + 1}: {status}")
            print(f"    Grounded: {result['grounding_source']} → {result['grounding_target']}")
            print(f"    Spatial context: {result['source_spatial_context']}")
            if not result["both_correct"]:
                print(f"    ERROR: {result['verification_details']}")

            if result["both_correct"]:
                correct_count += 1

        # Task summary
        accuracy = correct_count / args.n_episodes
        task_summary = {
            "task_id": gt.task_id,
            "task_description": gt.task_description,
            "n_episodes": args.n_episodes,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "ground_truth_source": gt.true_source,
            "ground_truth_target": gt.true_target,
        }
        task_summaries.append(task_summary)

        print(f"\n  Task {gt.task_id} Accuracy: {accuracy:.1%} ({correct_count}/{args.n_episodes})")

        env.close()

    # Overall summary
    total_episodes = len(all_results)
    total_correct = sum(1 for r in all_results if r["both_correct"])
    overall_accuracy = total_correct / total_episodes if total_episodes > 0 else 0

    source_correct = sum(1 for r in all_results if r["source_correct"])
    target_correct = sum(1 for r in all_results if r["target_correct"])

    print("\n" + "=" * 70)
    print("OVERALL GROUNDING ACCURACY")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Both correct: {total_correct} ({overall_accuracy:.1%})")
    print(f"Source correct: {source_correct} ({source_correct/total_episodes:.1%})")
    print(f"Target correct: {target_correct} ({target_correct/total_episodes:.1%})")
    print()

    print("Per-task breakdown:")
    for ts in task_summaries:
        status = "✓" if ts["accuracy"] == 1.0 else ("~" if ts["accuracy"] >= 0.8 else "✗")
        print(f"  Task {ts['task_id']}: {ts['accuracy']:.1%} {status}")

    # Save results
    summary = {
        "task_suite": args.task_suite,
        "n_episodes_per_task": args.n_episodes,
        "seed": args.seed,
        "total_episodes": total_episodes,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "source_accuracy": source_correct / total_episodes,
        "target_accuracy": target_correct / total_episodes,
        "task_summaries": task_summaries,
        "grounding_metrics": metrics.summary(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Return code based on accuracy threshold
    return 0 if overall_accuracy >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
