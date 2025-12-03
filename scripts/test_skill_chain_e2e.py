#!/usr/bin/env python3
"""End-to-end test of skill chaining on libero_10 tasks.

Uses oracle perception to validate the skill chain execution pipeline.
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")

from brain_robot.perception.oracle import OraclePerception
from brain_robot.world_model.state import WorldState
from brain_robot.planning.skill_chain import SkillChain
from brain_robot.config import SkillConfig


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

    return env, task


def run_skill_chain_test(task_suite: str, task_id: int, num_episodes: int = 3, verbose: bool = True):
    """Run skill chain test on a specific task."""

    print(f"\n{'='*70}")
    print(f"Testing Skill Chain on {task_suite} Task {task_id}")
    print(f"{'='*70}")

    # Create environment
    env, task = make_libero_env(task_suite, task_id)
    task_description = task.language
    print(f"Task: {task_description}")

    # Initialize components
    perception = OraclePerception()
    skill_chain = SkillChain(config=SkillConfig())

    successes = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        env.reset()

        # Initialize world state from oracle perception
        world_state = WorldState()
        perc_result = perception.perceive(env)
        world_state.update_from_perception(perc_result)

        if verbose:
            print(f"Objects detected: {list(world_state.objects.keys())}")

        # Decompose task
        steps = skill_chain.decompose_task(task_description, world_state)

        if not steps:
            print(f"[FAILED] Could not decompose task: {task_description}")
            continue

        print(f"Decomposed into {len(steps)} steps:")
        for i, step in enumerate(steps):
            print(f"  {i+1}. {step.skill_name}({step.args})")

        # Execute skill chain
        print("\nExecuting skill chain...")
        result = skill_chain.execute_chain(
            steps=steps,
            env=env,
            world_state=world_state,
            perception=perception,
            re_perceive_between_steps=True,
        )

        # Check environment success
        env_success = env.check_success()

        print(f"\nChain Result:")
        print(f"  Steps completed: {result.steps_completed}/{result.total_steps}")
        print(f"  Total env steps: {result.total_env_steps}")
        print(f"  Chain success: {result.success}")
        print(f"  Env success: {env_success}")

        if not result.success and result.failed_step:
            print(f"  Failed at: {result.failed_step.skill_name}")
            print(f"  Reason: {result.failure_reason}")

        if env_success:
            successes += 1
            print("  [SUCCESS] ✓")
        else:
            print("  [FAILED] ✗")

    env.close()

    success_rate = successes / num_episodes * 100
    print(f"\n{'='*70}")
    print(f"Summary: {successes}/{num_episodes} episodes succeeded ({success_rate:.1f}%)")
    print(f"{'='*70}")

    return successes, num_episodes


def main():
    parser = argparse.ArgumentParser(description="Test skill chaining E2E")
    parser.add_argument("--suite", default="libero_10", help="Task suite")
    parser.add_argument("--task", type=int, default=0, help="Task ID")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--all", action="store_true", help="Run all libero_10 tasks")
    args = parser.parse_args()

    if args.all:
        # Run all 10 tasks
        total_success = 0
        total_episodes = 0

        for task_id in range(10):
            try:
                s, e = run_skill_chain_test(
                    args.suite, task_id,
                    num_episodes=args.episodes,
                    verbose=args.verbose
                )
                total_success += s
                total_episodes += e
            except Exception as ex:
                print(f"\n[ERROR] Task {task_id} failed: {ex}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"OVERALL: {total_success}/{total_episodes} ({total_success/total_episodes*100:.1f}%)")
        print(f"{'='*70}")
    else:
        run_skill_chain_test(
            args.suite, args.task,
            num_episodes=args.episodes,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
