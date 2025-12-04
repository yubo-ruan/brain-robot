#!/usr/bin/env python3
"""Record rich annotated GIFs with perception, planning, and execution info."""

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


def create_info_panel(width, height, lines, title=None):
    """Create a side panel with text information."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        panel = Image.new('RGB', (width, height), color=(30, 30, 40))
        draw = ImageDraw.Draw(panel)
        
        y = 5
        if title:
            draw.rectangle([0, 0, width, 18], fill=(60, 60, 80))
            draw.text((5, 2), title, fill=(255, 255, 100))
            y = 22
        
        for line in lines:
            if line.startswith("---"):
                # Separator
                draw.line([(5, y+5), (width-5, y+5)], fill=(100, 100, 100))
                y += 12
            elif line.startswith(">>"):
                # Highlight
                draw.text((5, y), line[2:], fill=(100, 255, 100))
                y += 14
            elif line.startswith("!!"):
                # Error
                draw.text((5, y), line[2:], fill=(255, 100, 100))
                y += 14
            else:
                draw.text((5, y), line, fill=(200, 200, 200))
                y += 14
        
        return np.array(panel)
    except Exception as e:
        return np.zeros((height, width, 3), dtype=np.uint8)


def create_composite_frame(env, perception_info, planning_info, execution_info):
    """Create a composite frame with image + info panels."""
    # Main camera view (256x256)
    img = env.sim.render(camera_name='agentview', height=256, width=256, mode='offscreen')[::-1]
    
    # Info panel (200x256)
    panel_width = 200
    panel_height = 256
    
    all_lines = []
    
    # Perception section
    all_lines.append("---PERCEPTION---")
    all_lines.extend(perception_info)
    all_lines.append("---")
    
    # Planning section
    all_lines.append("---PLANNING---")
    all_lines.extend(planning_info)
    all_lines.append("---")
    
    # Execution section
    all_lines.append("---EXECUTION---")
    all_lines.extend(execution_info)
    
    panel = create_info_panel(panel_width, panel_height, all_lines)
    
    # Combine horizontally
    combined = np.concatenate([img, panel], axis=1)
    return combined


def get_gripper_pose(env):
    try:
        obs = env._get_observations() if hasattr(env, '_get_observations') else env.env._get_observations()
        if 'robot0_eef_pos' in obs:
            return obs['robot0_eef_pos'].copy()
    except:
        pass
    return None


def get_object_pose(env, obj_name):
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


class RichFrameCapture:
    """Capture frames with rich annotations during skill execution."""
    
    def __init__(self, skill, env, bowl_name, plate_name, frames_list, skill_name, world_state):
        self.skill = skill
        self.env = env
        self.bowl_name = bowl_name
        self.plate_name = plate_name
        self.frames = frames_list
        self.skill_name = skill_name
        self.world_state = world_state
        self.step_count = 0
        self.capture_every = 8
        self.bowl_start = None
        self.approach_strategy = None
        self.grasp_strategy = None
        
    def run(self, env, world_state, args):
        self.bowl_start = get_object_pose(self.env, self.bowl_name)
        self.approach_strategy = getattr(world_state, 'approach_strategy', 'top_down')
        
        original_step = self.skill._step_env
        
        def capturing_step(env, action):
            self.step_count += 1
            result = original_step(env, action)
            
            if self.step_count % self.capture_every == 0:
                self._capture_frame(action)
            
            return result
        
        self.skill._step_env = capturing_step
        result = self.skill.run(env, world_state, args)
        self.skill._step_env = original_step
        
        # Capture result info
        if hasattr(result, 'info'):
            self.grasp_strategy = result.info.get('grasp_point_info', {}).get('grasp_strategy', None)
        
        return result
    
    def _capture_frame(self, action):
        grip_pos = get_gripper_pose(self.env)
        bowl_pos = get_object_pose(self.env, self.bowl_name)
        plate_pos = get_object_pose(self.env, self.plate_name) if self.plate_name else None
        
        # Perception info
        perc_info = []
        if bowl_pos is not None:
            perc_info.append(f"Bowl: ({bowl_pos[0]:.2f},{bowl_pos[1]:.2f},{bowl_pos[2]:.2f})")
            if self.bowl_start is not None:
                delta = np.linalg.norm(bowl_pos - self.bowl_start)
                if delta > 0.01:
                    perc_info.append(f"!!Bowl moved: {delta*100:.1f}cm")
        if grip_pos is not None:
            perc_info.append(f"Grip: ({grip_pos[0]:.2f},{grip_pos[1]:.2f},{grip_pos[2]:.2f})")
        if grip_pos is not None and bowl_pos is not None:
            dist = np.linalg.norm(grip_pos - bowl_pos)
            perc_info.append(f"Distance: {dist*100:.1f}cm")
        
        # Planning info
        plan_info = [
            f"Skill: {self.skill_name}",
            f"Approach: {self.approach_strategy}",
        ]
        if self.grasp_strategy:
            plan_info.append(f"Grasp: {self.grasp_strategy}")
        
        # Execution info
        exec_info = [
            f"Step: {self.step_count}",
            f"Action XYZ: ({action[0]:.2f},{action[1]:.2f},{action[2]:.2f})",
            f"Gripper: {'CLOSE' if action[6] > 0 else 'OPEN'}",
        ]
        
        frame = create_composite_frame(self.env, perc_info, plan_info, exec_info)
        self.frames.append(frame)


def run_rich_recording(task_id, output_dir='recordings/rich'):
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark = get_benchmark('libero_spatial')()
    task = benchmark.get_task(task_id)
    task_bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(42)
    obs = env.reset()
    
    frames = []
    
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
    print(f"  Bowl: {bowl[:30]} at ({bowl_pose[0]:.2f}, {bowl_pose[1]:.2f}, {bowl_pose[2]:.2f})")
    
    # Detect context
    in_drawer = world_state.inside.get(bowl) is not None
    on_cabinet = bowl_pose[2] > 1.20
    on_elevated = bowl_pose[2] > 1.05
    
    context = "TABLE (standard)"
    if in_drawer:
        context = f"DRAWER ({world_state.inside.get(bowl, 'unknown')[:15]})"
    elif on_cabinet:
        context = "CABINET TOP (z>1.20)"
    elif on_elevated:
        context = "ELEVATED (z>1.05)"
    
    # Initial frame
    task_short = task.name.replace('_', ' ')[:50]
    frames.append(create_composite_frame(
        env,
        [f"Bowl: {bowl[:25]}", f"Pos: ({bowl_pose[0]:.2f},{bowl_pose[1]:.2f},{bowl_pose[2]:.2f})", f"Context: {context}"],
        [f">>Task: {task_short[:25]}", f"Source: bowl", f"Target: plate"],
        ["Status: INITIALIZING", f"Objects: {len(perc_result.object_names)}"]
    ))
    
    # Run skills
    skills_info = [
        ('Approach', ApproachSkill(config=config), {'obj': bowl}),
        ('Grasp', GraspSkill(config=config), {'obj': bowl}),
        ('Move', MoveSkill(config=config), {'obj': bowl, 'region': plate}),
        ('Place', PlaceSkill(config=config), {'obj': bowl, 'region': plate}),
    ]
    
    final_result = "SUCCESS"
    
    for skill_name, skill, args in skills_info:
        print(f"  {skill_name}...", end=" ", flush=True)
        
        perc_result = perception.perceive(env)
        world_state.update_from_perception(perc_result)
        
        bowl_before = get_object_pose(env, bowl)
        grip_before = get_gripper_pose(env)
        
        # Pre-skill frame
        frames.append(create_composite_frame(
            env,
            [f"Bowl: ({bowl_before[0]:.2f},{bowl_before[1]:.2f},{bowl_before[2]:.2f})" if bowl_before is not None else "Bowl: N/A",
             f"Grip: ({grip_before[0]:.2f},{grip_before[1]:.2f},{grip_before[2]:.2f})" if grip_before is not None else "Grip: N/A"],
            [f">>Starting: {skill_name}", f"Strategy: {getattr(world_state, 'approach_strategy', 'N/A')}"],
            ["Status: EXECUTING"]
        ))
        
        # Run with capture
        wrapper = RichFrameCapture(skill, env, bowl, plate, frames, skill_name, world_state)
        result = wrapper.run(env, world_state, args)
        
        bowl_after = get_object_pose(env, bowl)
        grip_after = get_gripper_pose(env)
        
        # Result frame
        if result.success:
            print("OK")
            skill.update_world_state(world_state, args, result)
            
            result_lines = [f">>SKILL OK"]
            if bowl_before is not None and bowl_after is not None:
                delta = np.linalg.norm(bowl_after - bowl_before)
                result_lines.append(f"Bowl moved: {delta*100:.1f}cm")
            
            frames.append(create_composite_frame(
                env,
                [f"Bowl: ({bowl_after[0]:.2f},{bowl_after[1]:.2f},{bowl_after[2]:.2f})" if bowl_after is not None else "N/A"],
                [f">>{skill_name} COMPLETE", f"Steps: {result.info.get('steps_taken', 'N/A')}"],
                result_lines
            ))
        else:
            err = result.info.get('error_msg', 'unknown')[:30]
            print(f"FAILED - {err}")
            final_result = f"FAILED at {skill_name}"
            
            # Detailed failure info
            fail_info = [f"!!FAILED: {skill_name}"]
            fail_info.append(f"!!{err}")
            
            # Add specific failure details
            if 'xy_error' in result.info:
                fail_info.append(f"XY error: {result.info['xy_error']*100:.1f}cm")
            if 'gripper_width' in result.info:
                fail_info.append(f"Grip width: {result.info['gripper_width']*1000:.1f}mm")
            if 'descent_info' in result.info:
                d = result.info['descent_info']
                fail_info.append(f"Descent: {d.get('start_z', 0):.2f}->{d.get('end_z', 0):.2f}")
            
            frames.append(create_composite_frame(
                env,
                [f"Bowl: ({bowl_after[0]:.2f},{bowl_after[1]:.2f},{bowl_after[2]:.2f})" if bowl_after is not None else "N/A"],
                [f"!!{skill_name} FAILED"],
                fail_info
            ))
            break
    
    # Final frame
    libero_success = env.env._check_success()
    frames.append(create_composite_frame(
        env,
        [f"Final bowl pos:", f"  {get_object_pose(env, bowl)}"],
        [f"Result: {final_result}"],
        [f">>LIBERO: {'SUCCESS' if libero_success else 'FAILED'}", f"Total frames: {len(frames)}"]
    ))
    
    # Save GIF
    gif_path = os.path.join(output_dir, f'task_{task_id}_rich.gif')
    imageio.mimsave(gif_path, frames, duration=400, loop=0)
    print(f"  Saved: {gif_path} ({len(frames)} frames)")
    
    env.close()
    return gif_path


def main():
    print("=" * 60)
    print("RECORDING RICH ANNOTATED GIFs")
    print("=" * 60)
    
    failed_tasks = [4, 5, 8, 9]
    gif_paths = []
    
    for task_id in failed_tasks:
        print()
        try:
            path = run_rich_recording(task_id)
            if path:
                gif_paths.append(path)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("RECORDED GIFs:")
    for p in gif_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
