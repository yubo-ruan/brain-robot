#!/usr/bin/env python3
"""Run LIBERO tasks with VLM planner and capture VLM outputs for analysis."""

import os
import sys

# Ensure patch_robosuite is imported first
sys.path.insert(0, os.path.dirname(__file__))
import patch_robosuite  # noqa: E402, F401

import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import imageio
from PIL import Image

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from brain_robot.vlm.qwen_planner import QwenVLPlanner


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
        "camera_depths": True,
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def get_camera_image(env, obs=None, camera_name="agentview"):
    """Get RGB image from camera."""
    if obs is None:
        if hasattr(env, 'env') and hasattr(env.env, '_get_observations'):
            obs = env.env._get_observations()
        else:
            return None

    img_key = f"{camera_name}_image"
    if img_key in obs:
        img = obs[img_key]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        return img
    return None


def get_gripper_state(obs) -> str:
    """Get gripper state from observation."""
    if 'robot0_gripper_qpos' in obs:
        gripper = obs['robot0_gripper_qpos']
        return "closed" if gripper[0] < 0.02 else "open"
    return "open"


def analyze_task_with_vlm(task_id: int, task_suite: str, vlm_planner: QwenVLPlanner, output_dir: Path, seed: int = 42):
    """Analyze a single task with VLM planner."""
    print(f"\n{'='*60}")
    print(f"Analyzing Task {task_id} with VLM")
    print(f"{'='*60}")

    # Create environment
    env, task_description = make_libero_env(task_suite, task_id)
    print(f"Task: {task_description}")

    # Initialize
    np.random.seed(seed)
    obs = env.reset()

    # Get image
    img = get_camera_image(env, obs)
    if img is None:
        print("  Failed to get image")
        env.close()
        return None

    # Get gripper state
    gripper_state = get_gripper_state(obs)

    # Reset VLM planner
    vlm_planner.reset()

    # Query VLM for initial plan
    print("\n  Querying VLM for initial plan...")
    vlm_output = vlm_planner.plan(
        image=img,
        task_description=task_description,
        gripper_state=gripper_state,
        steps_since_plan=0,
    )

    print("\n  VLM Output:")
    print(json.dumps(vlm_output, indent=2))

    # Save image
    output_dir.mkdir(parents=True, exist_ok=True)
    img_flipped = np.flipud(img)  # Flip for correct orientation
    img_path = output_dir / f"task_{task_id}_initial.png"
    imageio.imwrite(str(img_path), img_flipped)
    print(f"\n  Saved image: {img_path}")

    # Save VLM output
    vlm_path = output_dir / f"task_{task_id}_vlm_output.json"
    with open(vlm_path, 'w') as f:
        json.dump({
            "task_id": task_id,
            "task_description": task_description,
            "gripper_state": gripper_state,
            "vlm_output": vlm_output,
        }, f, indent=2)
    print(f"  Saved VLM output: {vlm_path}")

    # Now simulate a few phases to see VLM's progression
    # Phase 1: After "approach" - simulate gripper moved closer
    print("\n  Simulating post-approach state...")

    # Take a few steps to change the scene slightly
    for _ in range(10):
        action = np.zeros(7)
        action[0] = 0.1  # Move forward slightly
        obs, _, _, _ = env.step(action)

    img2 = get_camera_image(env, obs)
    gripper_state2 = get_gripper_state(obs)

    vlm_output2 = vlm_planner.plan(
        image=img2,
        task_description=task_description,
        gripper_state=gripper_state2,
        steps_since_plan=10,
    )

    print("\n  VLM Output (after approach movement):")
    print(json.dumps(vlm_output2, indent=2))

    # Save second state
    img2_flipped = np.flipud(img2)
    img2_path = output_dir / f"task_{task_id}_after_approach.png"
    imageio.imwrite(str(img2_path), img2_flipped)

    vlm2_path = output_dir / f"task_{task_id}_vlm_output_phase2.json"
    with open(vlm2_path, 'w') as f:
        json.dump({
            "task_id": task_id,
            "task_description": task_description,
            "phase": "after_approach",
            "gripper_state": gripper_state2,
            "vlm_output": vlm_output2,
        }, f, indent=2)

    env.close()

    return {
        "task_id": task_id,
        "task_description": task_description,
        "initial_vlm_output": vlm_output,
        "phase2_vlm_output": vlm_output2,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze LIBERO tasks with VLM")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[4, 5, 8, 9],
                        help="Task IDs to analyze")
    parser.add_argument("--task-suite", default="libero_spatial",
                        help="Task suite name")
    parser.add_argument("--output-dir", default="/workspace/brain_robot/recordings/vlm_analysis",
                        help="Output directory")
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="VLM model name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    print(f"Analyzing tasks: {args.task_ids}")
    print(f"Output directory: {output_dir}")

    # Load VLM planner
    print("\nLoading VLM planner...")
    vlm_planner = QwenVLPlanner(model_name=args.vlm_model)

    results = []
    for task_id in args.task_ids:
        try:
            result = analyze_task_with_vlm(
                task_id=task_id,
                task_suite=args.task_suite,
                vlm_planner=vlm_planner,
                output_dir=output_dir,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing task {task_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("VLM ANALYSIS SUMMARY")
    print("="*60)

    for r in results:
        print(f"\nTask {r['task_id']}: {r['task_description']}")
        print(f"  Initial phase: {r['initial_vlm_output'].get('plan', {}).get('phase', 'unknown')}")
        print(f"  Initial movements: {r['initial_vlm_output'].get('plan', {}).get('movements', [])}")
        print(f"  Reasoning: {r['initial_vlm_output'].get('reasoning', 'N/A')}")

    # Save full results
    results_path = output_dir / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
