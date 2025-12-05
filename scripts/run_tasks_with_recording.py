#!/usr/bin/env python3
"""Run specific LIBERO tasks with visual recording.

This script runs the evaluation on specified tasks and saves GIF recordings.
"""

import os
import sys

# Ensure patch_robosuite is imported first
sys.path.insert(0, os.path.dirname(__file__))
import patch_robosuite  # noqa: E402, F401

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import imageio

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from brain_robot.config import SkillConfig
from brain_robot.perception.oracle import OraclePerception
from brain_robot.world_model.state import WorldState
from brain_robot.skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill


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


def check_libero_success(env) -> bool:
    """Check LIBERO's built-in success condition."""
    if hasattr(env, 'env') and hasattr(env.env, '_check_success'):
        return bool(env.env._check_success())
    return False


def get_camera_image(env, obs=None, camera_name="agentview"):
    """Get RGB image from camera.

    Args:
        env: The LIBERO environment
        obs: Optional observation dict (if None, gets from underlying env)
        camera_name: Name of camera to capture from

    Returns:
        RGB image as uint8 numpy array, or None if not available
    """
    if obs is None:
        # Try to get observations from underlying env
        if hasattr(env, 'env') and hasattr(env.env, '_get_observations'):
            obs = env.env._get_observations()
        else:
            # Last resort - try render method
            try:
                # Try sim.render for MuJoCo
                if hasattr(env, 'env') and hasattr(env.env, 'sim'):
                    img = env.env.sim.render(width=256, height=256, camera_name=camera_name)
                    return img
            except Exception:
                pass
            return None

    img_key = f"{camera_name}_image"
    if img_key in obs:
        img = obs[img_key]
        # Convert from float [0,1] to uint8 if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        return img
    return None


def parse_task_for_grounding(task_description: str, object_names: list) -> tuple:
    """Simple task parsing to find source and target objects."""
    task_lower = task_description.lower()

    object_types = {}
    for obj_id in object_names:
        obj_lower = obj_id.lower()
        if 'bowl' in obj_lower:
            object_types.setdefault('bowl', []).append(obj_id)
        if 'plate' in obj_lower and 'burner' not in obj_lower:
            object_types.setdefault('plate', []).append(obj_id)
        if 'ramekin' in obj_lower:
            object_types.setdefault('ramekin', []).append(obj_id)
        if 'mug' in obj_lower:
            object_types.setdefault('mug', []).append(obj_id)
        if 'drawer' in obj_lower:
            object_types.setdefault('drawer', []).append(obj_id)
        if 'cabinet' in obj_lower:
            object_types.setdefault('cabinet', []).append(obj_id)

    source_obj = None
    target_obj = None

    # Pattern: "pick up the X" / "grab the X"
    source_keywords = ['pick up the', 'pick the', 'grab the', 'take the']
    for kw in source_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[1]
            rest_prefix = ' '.join(rest.split()[:5])
            for obj_type, obj_list in object_types.items():
                if obj_type in rest_prefix:
                    source_obj = obj_list[0]
                    break
            if source_obj:
                break

    # Pattern: "place it on the Y"
    target_keywords = ['place it on the', 'place it in the', 'on the', 'in the']
    for kw in target_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[-1]
            rest_prefix = ' '.join(rest.split()[:5])
            for obj_type, obj_list in object_types.items():
                if obj_type in rest_prefix:
                    target_obj = obj_list[0]
                    break
            if target_obj:
                break

    return source_obj, target_obj


def run_task_with_recording(task_id: int, task_suite: str, output_dir: Path, seed: int = 42, grasp_selector_type: str = "heuristic"):
    """Run a single task and record GIF."""
    print(f"\n{'='*60}")
    print(f"Running Task {task_id} (seed={seed}, grasp={grasp_selector_type})")
    print(f"{'='*60}")

    # Create environment
    env, task_description = make_libero_env(task_suite, task_id)
    print(f"Task: {task_description}")

    # Initialize
    np.random.seed(seed)
    obs = env.reset()

    # Setup perception (no env argument - it takes env in perceive())
    perception = OraclePerception()
    world_state = WorldState()
    skill_config = SkillConfig()

    # Get initial perception
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)

    # Find source and target objects using task parsing
    source_obj, target_obj = parse_task_for_grounding(task_description, perc_result.object_names)

    print(f"Source: {source_obj}")
    print(f"Target: {target_obj}")

    if source_obj is None or target_obj is None:
        print("  ✗ Grounding failed - cannot run skills")
        env.close()
        return {
            "task_id": task_id,
            "task_description": task_description,
            "success": False,
            "error": "grounding_failed",
            "gif_path": None,
        }

    # Collect frames for GIF
    frames = []

    # Get initial frame
    img = get_camera_image(env)
    if img is not None:
        frames.append(img)

    # Define skill sequence using proper skill API
    skills = [
        (ApproachSkill(config=skill_config), {"obj": source_obj}),
        (GraspSkill(config=skill_config, grasp_selector_type=grasp_selector_type), {"obj": source_obj}),
        (MoveSkill(config=skill_config), {"obj": source_obj, "region": target_obj}),
        (PlaceSkill(config=skill_config), {"obj": source_obj, "region": target_obj}),
    ]

    total_steps = 0
    success = False
    failed_skill = None

    for skill, args in skills:
        print(f"  Executing {skill.name}...")

        # Update perception before skill
        perc_result = perception.perceive(env)
        world_state.update_from_perception(perc_result)

        # Run skill
        result = skill.run(env, world_state, args)

        # Capture frames during/after skill (get frame after each skill)
        img = get_camera_image(env)
        if img is not None:
            frames.append(img)

        total_steps += result.info.get("steps_taken", 0)

        if not result.success:
            failed_skill = skill.name
            print(f"    ✗ {skill.name} failed: {result.info.get('error_msg', 'Unknown')}")
            break

        print(f"    ✓ {skill.name} OK ({result.info.get('steps_taken', 0)} steps)")
        # Print placement metrics if available
        if 'final_xy_error' in result.info:
            print(f"      XY error: {result.info['final_xy_error']:.3f}m (need <0.03m)")
        if 'xy_error_before' in result.info:
            print(f"      XY before centering: {result.info['xy_error_before']:.3f}m")
        if 'xy_error_after' in result.info:
            print(f"      XY after centering: {result.info['xy_error_after']:.3f}m")

        # Check success after each skill
        if check_libero_success(env):
            success = True
            print(f"  SUCCESS at step {total_steps}!")
            break

    if not success and failed_skill is None:
        # All skills completed but LIBERO doesn't consider it success
        success = check_libero_success(env)
        print(f"  Final LIBERO check: {'SUCCESS' if success else 'FAILED'}")

    # Get final frame
    img = get_camera_image(env)
    if img is not None:
        frames.append(img)

    # Save GIF
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"task_{task_id}_seed_{seed}.gif"

    if frames:
        # Flip frames vertically (OpenGL convention)
        frames = [np.flipud(f) for f in frames]
        imageio.mimsave(str(gif_path), frames, fps=10)
        print(f"  Saved recording to: {gif_path}")

    env.close()

    result = {
        "task_id": task_id,
        "task_description": task_description,
        "success": success,
        "total_steps": total_steps,
        "gif_path": str(gif_path) if frames else None,
    }

    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run LIBERO tasks with recording")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[4, 5, 8, 9],
                        help="Task IDs to run")
    parser.add_argument("--task-suite", default="libero_spatial",
                        help="Task suite name")
    parser.add_argument("--output-dir", default="/workspace/brain_robot/recordings/fresh_run",
                        help="Output directory for recordings")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task")
    parser.add_argument("--grasp-selector", default="heuristic",
                        choices=["heuristic", "contact_graspnet"],
                        help="Grasp selector type")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp

    print(f"Running tasks: {args.task_ids}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Grasp selector: {args.grasp_selector}")
    print(f"Output directory: {output_dir}")

    results = []
    for task_id in args.task_ids:
        for ep in range(args.episodes):
            seed = args.seed + ep  # Different seed per episode
            try:
                result = run_task_with_recording(
                    task_id=task_id,
                    task_suite=args.task_suite,
                    output_dir=output_dir,
                    seed=seed,
                    grasp_selector_type=args.grasp_selector,
                )
                result["episode"] = ep
                results.append(result)
            except Exception as e:
                print(f"Error running task {task_id} ep {ep}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "task_id": task_id,
                    "episode": ep,
                    "success": False,
                    "error": str(e)
                })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Overall success rate
    successes = sum(1 for r in results if r.get("success", False))
    print(f"Overall success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")

    # Per-task success rates
    from collections import defaultdict
    task_results = defaultdict(list)
    for r in results:
        task_results[r["task_id"]].append(r.get("success", False))

    print("\nPer-task results:")
    for task_id in sorted(task_results.keys()):
        task_successes = sum(task_results[task_id])
        task_total = len(task_results[task_id])
        pct = 100 * task_successes / task_total
        print(f"  Task {task_id}: {task_successes}/{task_total} ({pct:.0f}%)")

    # List all recordings
    print("\nRecordings:")
    for r in results:
        status = "✓" if r.get("success") else "✗"
        ep = r.get("episode", 0)
        print(f"  Task {r['task_id']} ep{ep}: {status}")
        if r.get("gif_path"):
            print(f"    {r['gif_path']}")

    return results


if __name__ == "__main__":
    main()
