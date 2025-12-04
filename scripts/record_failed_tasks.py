#!/usr/bin/env python3
"""Record GIF videos of failed tasks for analysis."""

import os
import sys
import numpy as np
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from brain_robot.perception.oracle import OraclePerception
from brain_robot.world_model.state import WorldState
from brain_robot.skills.approach import ApproachSkill
from brain_robot.skills.grasp import GraspSkill
from brain_robot.skills.move import MoveSkill
from brain_robot.skills.place import PlaceSkill
from brain_robot.config import SkillConfig


def add_text_to_frame(frame, text, position=(10, 30), font_scale=0.7):
    """Add text overlay to frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        # Use default font
        draw.text(position, text, fill=(255, 255, 255))
        return np.array(img)
    except ImportError:
        return frame


def run_episode_with_recording(task_id, seed=42, output_dir="recordings/failed_tasks"):
    """Run episode and record frames."""
    os.makedirs(output_dir, exist_ok=True)

    benchmark = get_benchmark('libero_spatial')()
    task = benchmark.get_task(task_id)
    task_bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    obs = env.reset()

    frames = []

    # Capture initial frame
    if 'agentview_image' in obs:
        frames.append(obs['agentview_image'].copy())

    config = SkillConfig()
    perception = OraclePerception()
    world_state = WorldState()

    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)

    # Simple grounding based on task
    source_obj = None
    target_obj = None
    for obj in perc_result.object_names:
        if 'bowl' in obj.lower() and source_obj is None:
            source_obj = obj
        elif 'plate' in obj.lower() and target_obj is None:
            target_obj = obj

    if source_obj is None:
        print(f"  Could not find source object")
        env.close()
        return None

    print(f"Task {task_id}: {task.name[:60]}...")
    print(f"  Source: {source_obj}, Target: {target_obj}")

    obj_pose = world_state.objects[source_obj].pose
    print(f"  Object position: x={obj_pose[0]:.3f}, y={obj_pose[1]:.3f}, z={obj_pose[2]:.3f}")

    # Run skills and capture frames
    skills = [
        ("Approach", ApproachSkill(config=config), {'obj': source_obj}),
        ("Grasp", GraspSkill(config=config), {'obj': source_obj}),
        ("Move", MoveSkill(config=config), {'obj': source_obj, 'target': target_obj}),
        ("Place", PlaceSkill(config=config), {'obj': source_obj, 'target': target_obj}),
    ]

    success = True
    failed_skill = None

    for skill_name, skill, args in skills:
        print(f"  Running {skill_name}...", end=" ")

        # Run skill step by step to capture frames
        result = skill.run(env, world_state, args)

        # Capture frame after skill
        obs = env.env._get_observations() if hasattr(env, 'env') else env._get_observations()
        if 'agentview_image' in obs:
            frame = obs['agentview_image'].copy()
            # Add text overlay
            status = "OK" if result.success else "FAILED"
            frame = add_text_to_frame(frame, f"{skill_name}: {status}")
            frames.append(frame)

        if result.success:
            print("OK")
            skill.update_world_state(world_state, args, result)
        else:
            print(f"FAILED - {result.info.get('error_msg', 'unknown')}")
            success = False
            failed_skill = skill_name
            break

    # Check LIBERO success
    libero_success = env.env._check_success() if hasattr(env, 'env') else env._check_success()
    print(f"  LIBERO success: {libero_success}")

    # Save GIF
    if frames:
        gif_path = os.path.join(output_dir, f"task_{task_id}_seed_{seed}.gif")
        imageio.mimsave(gif_path, frames, fps=5, loop=0)
        print(f"  Saved: {gif_path}")

    env.close()
    return {
        'task_id': task_id,
        'success': success,
        'libero_success': libero_success,
        'failed_skill': failed_skill,
        'n_frames': len(frames),
    }


def main():
    """Record GIFs for failed tasks."""
    print("=" * 60)
    print("RECORDING FAILED TASKS")
    print("=" * 60)

    # Failed tasks from ablation study
    failed_tasks = [4, 5, 8, 9]

    results = []
    for task_id in failed_tasks:
        print()
        result = run_episode_with_recording(task_id, seed=42)
        if result:
            results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"Task {r['task_id']}: {'SUCCESS' if r['success'] else 'FAILED'} "
              f"(LIBERO: {r['libero_success']}, frames: {r['n_frames']})")


if __name__ == "__main__":
    main()
