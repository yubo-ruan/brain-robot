#!/usr/bin/env python3
"""Stress test perception robustness with controlled noise injection.

Tests how the system degrades with increasing position noise (σ = 1cm, 2cm, 3cm, 5cm).
This tells us:
1. How close we are to the true robustness limit
2. Whether the system degrades gracefully or catastrophically

Usage:
    python scripts/run_noise_stress_test.py --task-id 0 --n-episodes 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


def run_with_noise(task_id, n_episodes, noise_cm, seed=42):
    """Run evaluation with specified noise level.

    Returns:
        dict with success_rate and episode results
    """
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from brain_robot.perception.noisy_oracle import NoisyOraclePerception
    from brain_robot.world_model.state import WorldState
    from brain_robot.skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill
    from brain_robot.config import SkillConfig
    from brain_robot.utils.seeds import set_global_seed, get_episode_seed

    # Setup environment
    benchmark = get_benchmark("libero_spatial")()
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
    task_description = task.language

    # Setup noisy perception
    noise_meters = noise_cm / 100.0
    perception = NoisyOraclePerception(
        pos_noise_std=noise_meters,
        ori_noise_std=0.0,  # Only test position noise
        seed=seed,
    )

    config = SkillConfig()

    # Parse task
    def parse_task(task_desc, object_names):
        task_lower = task_desc.lower()
        source = None
        target = None

        for obj in object_names:
            if "bowl" in obj.lower() and source is None:
                source = obj
            if "plate" in obj.lower() and "burner" not in obj.lower():
                target = obj

        if not source and object_names:
            source = object_names[0]
        if not target and len(object_names) > 1:
            target = object_names[1]

        return source, target

    successes = 0
    results = []

    for ep in range(n_episodes):
        episode_seed = get_episode_seed(seed, ep)
        set_global_seed(episode_seed, env)
        env.reset()

        world_state = WorldState()
        perc_result = perception.perceive(env)
        world_state.update_from_perception(perc_result)

        if len(perc_result.object_names) < 2:
            results.append({"episode": ep, "success": False, "reason": "no_objects"})
            continue

        source, target = parse_task(task_description, perc_result.object_names)

        skills = [
            (ApproachSkill(config=config), {"obj": source}),
            (GraspSkill(config=config), {"obj": source}),
            (MoveSkill(config=config), {"obj": source, "region": target}),
            (PlaceSkill(config=config), {"obj": source, "region": target}),
        ]

        success = True
        failed_skill = None

        for skill, args in skills:
            perc_result = perception.perceive(env)
            world_state.update_from_perception(perc_result)
            result = skill.run(env, world_state, args)

            if not result.success:
                success = False
                failed_skill = skill.name
                break

        if success:
            successes += 1

        results.append({
            "episode": ep,
            "success": success,
            "failed_skill": failed_skill if not success else None,
        })

    env.close()

    return {
        "noise_cm": noise_cm,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Noise Stress Test")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("PERCEPTION NOISE STRESS TEST")
    print("=" * 70)
    print(f"Task ID: {args.task_id}")
    print(f"Episodes per noise level: {args.n_episodes}")
    print()

    # Test noise levels: 0, 1, 2, 3, 5, 7, 10 cm
    noise_levels = [0, 1, 2, 3, 5, 7, 10]
    all_results = []

    for noise_cm in noise_levels:
        print(f"\n--- Testing σ = {noise_cm} cm ---")
        result = run_with_noise(args.task_id, args.n_episodes, noise_cm, args.seed)
        all_results.append(result)
        print(f"Success rate: {result['success_rate']:.1%} ({result['successes']}/{result['n_episodes']})")

    # Summary
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"\n{'Noise (σ)':<12} {'Success Rate':<15} {'Episodes'}")
    print("-" * 40)

    for r in all_results:
        pct = r['success_rate'] * 100
        bar = "█" * int(pct / 5)
        print(f"{r['noise_cm']:>3} cm       {pct:>5.1f}%          {r['successes']}/{r['n_episodes']}  {bar}")

    # Find degradation threshold
    baseline = all_results[0]['success_rate']
    degradation_threshold = None

    for r in all_results:
        if r['success_rate'] < baseline * 0.8:  # 20% degradation
            degradation_threshold = r['noise_cm']
            break

    print()
    if degradation_threshold:
        print(f"⚠ System degrades significantly (>20% drop) at σ = {degradation_threshold} cm")
    else:
        print("✓ System is robust up to σ = 10 cm position noise")

    # Check graceful vs catastrophic degradation
    rates = [r['success_rate'] for r in all_results]
    drops = [rates[i] - rates[i+1] for i in range(len(rates)-1)]
    max_drop = max(drops) if drops else 0

    if max_drop > 0.3:
        print(f"⚠ CATASTROPHIC degradation detected (>{max_drop*100:.0f}% drop in one step)")
    else:
        print("✓ Degradation is graceful (no sudden drops)")

    # Save results
    output_dir = Path("logs/noise_stress_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"task{args.task_id}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "task_id": args.task_id,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
