#!/usr/bin/env python3
"""Collect perception training data from LIBERO tasks.

Runs episodes with oracle perception and collects RGB images + ground truth
labels for training learned perception models.

Usage:
    python scripts/collect_perception_data.py --task-suite libero_spatial --n-episodes 10

Output:
    data/perception/
    ├── images/          # RGB images
    ├── train.json       # Training annotations
    ├── val.json         # Validation annotations
    └── metadata.json    # Dataset metadata
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from brain_robot.perception.oracle import OraclePerception
from brain_robot.perception.data_collection import (
    PerceptionDataCollector,
    KeyframeSelector,
)
from brain_robot.world_model.state import WorldState
from brain_robot.utils.seeds import set_global_seed, get_episode_seed


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
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def collect_episode(
    env,
    collector: PerceptionDataCollector,
    perception: OraclePerception,
    selector: KeyframeSelector,
    task_id: int,
    episode_idx: int,
    task_description: str,
    max_steps: int = 300,
):
    """Collect data from a single episode using random actions.

    For data collection, we use random actions to get diverse
    viewpoints and object configurations. Success doesn't matter.
    """
    import numpy as np

    collector.start_episode(task_id, episode_idx, task_description)
    selector.reset()

    obs = env.reset()
    world_state = WorldState()

    # Collect initial frame
    oracle_result = perception.perceive(env)
    world_state.update_from_perception(oracle_result)
    collector.collect_frame(env, oracle_result, step_idx=0)

    frames_collected = 1

    # Get action dimension from env
    action_dim = env.action_dim if hasattr(env, 'action_dim') else 7

    for step in range(1, max_steps):
        # Random action for diverse data
        action = np.random.uniform(-1, 1, size=action_dim)
        obs, reward, done, info = env.step(action)

        # Update perception
        oracle_result = perception.perceive(env)
        world_state.update_from_perception(oracle_result)

        # Check if should collect
        gripper_pos = None
        if oracle_result.gripper_pose is not None:
            gripper_pos = oracle_result.gripper_pose[:3]

        should_collect = selector.should_collect(
            step=step,
            holding=world_state.holding,
            gripper_width=oracle_result.gripper_width,
        )

        # Force collect at episode end
        if done or step == max_steps - 1:
            should_collect = True

        if should_collect:
            collector.collect_frame(env, oracle_result, step_idx=step)
            frames_collected += 1

        if done:
            break

    # Mark episode as complete (success=True for data collection purposes)
    collector.end_episode(success=True)

    return frames_collected


def main():
    parser = argparse.ArgumentParser(description="Collect Perception Training Data")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated task IDs")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--keyframe-interval", type=int, default=15,
                        help="Minimum steps between keyframes")
    parser.add_argument("--output-dir", type=str, default="data/perception")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cameras", type=str, default="agentview",
                        help="Comma-separated camera names")
    args = parser.parse_args()

    # Parse arguments
    task_ids = [int(x) for x in args.task_ids.split(",")]
    cameras = [x.strip() for x in args.cameras.split(",")]

    print("=" * 70)
    print("PERCEPTION DATA COLLECTION")
    print("=" * 70)
    print(f"Task Suite: {args.task_suite}")
    print(f"Task IDs: {task_ids}")
    print(f"Episodes per task: {args.n_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Keyframe interval: {args.keyframe_interval}")
    print(f"Cameras: {cameras}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Initialize collector
    collector = PerceptionDataCollector(
        output_dir=args.output_dir,
        task_suite=args.task_suite,
        cameras=cameras,
        image_size=(256, 256),
    )

    perception = OraclePerception()
    selector = KeyframeSelector(interval=args.keyframe_interval)

    total_frames = 0
    total_episodes = 0

    for task_id in task_ids:
        print(f"\n{'='*70}")
        print(f"Task {task_id}")
        print("=" * 70)

        # Create environment
        env, task_description = make_libero_env(args.task_suite, task_id)
        print(f"Task: {task_description}")

        for episode_idx in range(args.n_episodes):
            # Set seed for reproducibility
            episode_seed = get_episode_seed(args.seed, task_id * 1000 + episode_idx)
            set_global_seed(episode_seed, env)

            # Collect episode
            frames = collect_episode(
                env=env,
                collector=collector,
                perception=perception,
                selector=selector,
                task_id=task_id,
                episode_idx=episode_idx,
                task_description=task_description,
                max_steps=args.max_steps,
            )

            total_frames += frames
            total_episodes += 1
            print(f"  Episode {episode_idx + 1}/{args.n_episodes}: {frames} frames")

        env.close()

    # Save dataset
    print(f"\n{'='*70}")
    print("SAVING DATASET")
    print("=" * 70)
    collector.save_dataset(split_ratio=0.9, seed=args.seed)

    # Summary
    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Frames per task: {dict(collector.stats['frames_per_task'])}")


if __name__ == "__main__":
    main()
