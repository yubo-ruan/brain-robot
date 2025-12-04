#!/usr/bin/env python3
"""Record detailed GIFs with frame-by-frame capture and rich annotations."""

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


def add_annotations(frame, lines, start_y=10):
    """Add multiple lines of text to frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        y = start_y
        for line in lines:
            # Background
            draw.rectangle([5, y-2, 250, y+12], fill=(0, 0, 0, 180))
            draw.text((8, y), line, fill=(255, 255, 255))
            y += 14
        
        return np.array(img)
    except Exception as e:
        return frame


def get_gripper_pose(env):
    """Get current gripper pose from env."""
    try:
        obs = env._get_observations() if hasattr(env, '_get_observations') else env.env._get_observations()
        if 'robot0_eef_pos' in obs:
            return obs['robot0_eef_pos'].copy()
    except:
        pass
    return None


def get_object_pose(env, obj_name):
    """Get object pose directly from simulator."""
    try:
        body_name = obj_name
        try:
            body_id = env.sim.model.body_name2id(body_name)
        except ValueError:
            if body_name.endswith('_main'):
                body_name = body_name[:-5]
            body_id = env.sim.model.body_name2id(body_name)
        return env.sim.data.body_xpos[body_id].copy()
    except:
        return None


def capture_frame(env, annotations):
    """Capture frame with annotations."""
    img = env.sim.render(camera_name='agentview', height=256, width=256, mode='offscreen')[::-1]
    return add_annotations(img, annotations)


class FrameCapturingSkill:
    """Wrapper to capture frames during skill execution."""
    
    def __init__(self, skill, env, bowl_name, frames_list, skill_name):
        self.skill = skill
        self.env = env
        self.bowl_name = bowl_name
        self.frames = frames_list
        self.skill_name = skill_name
        self.step_count = 0
        self.capture_every = 5  # Capture every N steps
        
    def run(self, env, world_state, args):
        # Store original _step_env
        original_step = self.skill._step_env
        
        def capturing_step(env, action):
            self.step_count += 1
            result = original_step(env, action)
            
            if self.step_count % self.capture_every == 0:
                # Get current state
                grip_pos = get_gripper_pose(self.env)
                bowl_pos = get_object_pose(self.env, self.bowl_name)
                
                annotations = [
                    f"{self.skill_name} step {self.step_count}",
                ]
                if grip_pos is not None:
                    annotations.append(f"Grip: ({grip_pos[0]:.2f}, {grip_pos[1]:.2f}, {grip_pos[2]:.2f})")
                if bowl_pos is not None:
                    annotations.append(f"Bowl: ({bowl_pos[0]:.2f}, {bowl_pos[1]:.2f}, {bowl_pos[2]:.2f})")
                if grip_pos is not None and bowl_pos is not None:
                    dist = np.linalg.norm(grip_pos - bowl_pos)
                    annotations.append(f"Distance: {dist*100:.1f}cm")
                
                self.frames.append(capture_frame(self.env, annotations))
            
            return result
        
        self.skill._step_env = capturing_step
        result = self.skill.run(env, world_state, args)
        self.skill._step_env = original_step
        
        return result


def run_detailed_recording(task_id, output_dir='recordings/annotated'):
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark = get_benchmark('libero_spatial')()
    task = benchmark.get_task(task_id)
    task_bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(42)
    obs = env.reset()
    
    frames = []
    
    # Initial frame with task description
    task_short = task.name[:60] + "..." if len(task.name) > 60 else task.name
    frames.append(capture_frame(env, [
        f"Task {task_id}",
        task_short[:40],
        task_short[40:80] if len(task_short) > 40 else "",
    ]))
    
    perception = OraclePerception()
    world_state = WorldState()
    config = SkillConfig()
    
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    
    # Find objects
    bowl = None
    plate = None
    for obj in perc_result.object_names:
        if 'bowl' in obj.lower() and bowl is None:
            bowl = obj
        if 'plate' in obj.lower() and 'burner' not in obj.lower() and plate is None:
            plate = obj
    
    if bowl is None:
        print(f"Task {task_id}: No bowl found")
        env.close()
        return
    
    bowl_pose = world_state.objects[bowl].pose[:3]
    print(f"Task {task_id}: {task.name[:50]}...")
    print(f"  Bowl: {bowl} at ({bowl_pose[0]:.2f}, {bowl_pose[1]:.2f}, {bowl_pose[2]:.2f})")
    
    # Detect context
    in_drawer = world_state.inside.get(bowl) is not None
    on_cabinet = bowl_pose[2] > 1.20
    on_elevated = bowl_pose[2] > 1.05
    
    context = "table"
    if in_drawer:
        context = f"inside {world_state.inside.get(bowl, 'drawer')}"
    elif on_cabinet:
        context = "on cabinet top"
    elif on_elevated:
        context = "on elevated surface"
    
    frames.append(capture_frame(env, [
        f"Bowl: {bowl[:30]}",
        f"Position: ({bowl_pose[0]:.2f}, {bowl_pose[1]:.2f}, {bowl_pose[2]:.2f})",
        f"Context: {context}",
    ]))
    
    # Run skills
    skills_info = [
        ('Approach', ApproachSkill(config=config), {'obj': bowl}),
        ('Grasp', GraspSkill(config=config), {'obj': bowl}),
        ('Move', MoveSkill(config=config), {'obj': bowl, 'region': plate}),
        ('Place', PlaceSkill(config=config), {'obj': bowl, 'region': plate}),
    ]
    
    final_result = "SUCCESS"
    failed_skill = None
    
    for skill_name, skill, args in skills_info:
        print(f"  {skill_name}...", end=" ", flush=True)
        
        # Re-perceive
        perc_result = perception.perceive(env)
        world_state.update_from_perception(perc_result)
        
        bowl_before = get_object_pose(env, bowl)
        grip_before = get_gripper_pose(env)
        
        # Start frame
        frames.append(capture_frame(env, [
            f"{skill_name}: STARTING",
            f"Grip: ({grip_before[0]:.2f}, {grip_before[1]:.2f}, {grip_before[2]:.2f})" if grip_before is not None else "Grip: N/A",
            f"Bowl: ({bowl_before[0]:.2f}, {bowl_before[1]:.2f}, {bowl_before[2]:.2f})" if bowl_before is not None else "Bowl: N/A",
        ]))
        
        # Run with frame capture
        wrapper = FrameCapturingSkill(skill, env, bowl, frames, skill_name)
        result = wrapper.run(env, world_state, args)
        
        bowl_after = get_object_pose(env, bowl)
        grip_after = get_gripper_pose(env)
        
        # Result frame
        status = "OK" if result.success else "FAILED"
        result_lines = [f"{skill_name}: {status}"]
        
        if bowl_before is not None and bowl_after is not None:
            bowl_delta = np.linalg.norm(bowl_after - bowl_before)
            result_lines.append(f"Bowl moved: {bowl_delta*100:.1f}cm")
        
        if not result.success:
            err = result.info.get('error_msg', 'unknown')[:35]
            result_lines.append(f"Error: {err}")
        
        frames.append(capture_frame(env, result_lines))
        
        if result.success:
            print("OK")
            skill.update_world_state(world_state, args, result)
        else:
            err = result.info.get('error_msg', 'unknown')[:40]
            print(f"FAILED - {err}")
            final_result = f"FAILED at {skill_name}"
            failed_skill = skill_name
            break
    
    # Final frame
    libero_success = env.env._check_success()
    frames.append(capture_frame(env, [
        f"RESULT: {final_result}",
        f"LIBERO Success: {libero_success}",
        f"Total frames: {len(frames)}",
    ]))
    
    # Save GIF
    gif_path = os.path.join(output_dir, f'task_{task_id}_annotated.gif')
    imageio.mimsave(gif_path, frames, duration=300, loop=0)  # 300ms per frame
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    
    env.close()
    return gif_path


def main():
    print("=" * 60)
    print("RECORDING ANNOTATED GIFs FOR FAILED TASKS")
    print("=" * 60)
    
    failed_tasks = [4, 5, 8, 9]
    gif_paths = []
    
    for task_id in failed_tasks:
        print()
        try:
            path = run_detailed_recording(task_id)
            if path:
                gif_paths.append(path)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Recorded {len(gif_paths)} GIFs:")
    for p in gif_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
