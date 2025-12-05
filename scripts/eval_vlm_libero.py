#!/usr/bin/env python3
"""
Evaluate VLM-conditioned policy on LIBERO tasks with visualization.

This script:
1. Loads a trained VLM-conditioned policy
2. Runs evaluation episodes on LIBERO tasks
3. Generates GIFs with VLM output overlay showing:
   - Current phase and planned direction
   - Gripper state and confidence
   - Primitive weights (for brain-inspired policy)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm

# LIBERO imports
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T


def create_visualization_frame(
    image: np.ndarray,
    vlm_plan: Dict,
    step: int,
    action: np.ndarray,
    policy_info: Optional[Dict] = None,
    task_description: str = "",
) -> np.ndarray:
    """
    Create visualization frame with VLM output overlay.

    Args:
        image: Robot camera image (H, W, 3)
        vlm_plan: VLM plan dictionary
        step: Current step number
        action: Current action
        policy_info: Optional policy info (primitive weights, etc.)
        task_description: Task description string

    Returns:
        Annotated image as numpy array
    """
    # Flip image vertically (LIBERO camera has flipped Y-axis)
    image = np.flipud(image).copy()

    # Resize image if needed
    if image.shape[0] != 256:
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((256, 256), Image.BILINEAR)
        image = np.array(pil_img)

    # Create larger canvas for annotations
    canvas_height = 400
    canvas_width = 350
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 40  # Dark gray background

    # Place image at top
    canvas[10:266, 10:266] = image

    # Create PIL image for text
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font

    # Colors
    white = (255, 255, 255)
    yellow = (255, 255, 0)
    green = (0, 255, 0)
    cyan = (0, 255, 255)
    orange = (255, 165, 0)

    # Draw step counter
    draw.text((275, 15), f"Step: {step}", fill=white, font=font)

    # Extract VLM plan info
    if vlm_plan:
        plan = vlm_plan.get('plan', {})
        obs = vlm_plan.get('observation', {})
        reasoning = vlm_plan.get('reasoning', '')

        phase = plan.get('phase', 'unknown')
        movements = plan.get('movements', [])
        gripper = plan.get('gripper', 'maintain')
        confidence = plan.get('confidence', 0.0)

        # Direction string
        if movements:
            directions = [f"{m['direction']}({m.get('speed', 'medium')[0]})" for m in movements[:2]]
            direction_str = ", ".join(directions)
        else:
            direction_str = "none"
    else:
        phase = "no plan"
        direction_str = "N/A"
        gripper = "N/A"
        confidence = 0.0
        reasoning = ""

    # VLM Info panel (right side)
    y_offset = 50
    draw.text((275, y_offset), "VLM Plan:", fill=yellow, font=font_title)
    y_offset += 18

    # Phase with color coding
    phase_colors = {
        'approach': green,
        'align': cyan,
        'descend': orange,
        'grasp': (255, 0, 0),
        'lift': (0, 200, 255),
        'move': (200, 100, 255),
        'place': (255, 200, 0),
        'release': (100, 255, 100),
    }
    phase_color = phase_colors.get(phase, white)
    draw.text((275, y_offset), f"Phase: {phase}", fill=phase_color, font=font)
    y_offset += 16

    draw.text((275, y_offset), f"Dir: {direction_str}", fill=white, font=font_small)
    y_offset += 14

    draw.text((275, y_offset), f"Grip: {gripper}", fill=white, font=font_small)
    y_offset += 14

    # Confidence bar
    draw.text((275, y_offset), f"Conf: ", fill=white, font=font_small)
    bar_x = 310
    bar_width = 30
    bar_height = 8
    draw.rectangle([bar_x, y_offset, bar_x + bar_width, y_offset + bar_height], outline=white)
    fill_width = int(bar_width * confidence)
    if fill_width > 0:
        conf_color = green if confidence > 0.7 else (yellow if confidence > 0.4 else (255, 100, 100))
        draw.rectangle([bar_x, y_offset, bar_x + fill_width, y_offset + bar_height], fill=conf_color)
    y_offset += 20

    # Action info panel (bottom)
    y_offset = 280
    draw.text((10, y_offset), "Action:", fill=yellow, font=font_title)
    y_offset += 16

    # Format action nicely
    if action is not None:
        action_str = f"xyz: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
        draw.text((10, y_offset), action_str, fill=white, font=font_small)
        y_offset += 14

        if len(action) >= 7:
            action_str2 = f"rot: [{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}]"
            draw.text((10, y_offset), action_str2, fill=white, font=font_small)
            y_offset += 14

            grip_val = action[6] if len(action) > 6 else 0
            grip_str = "CLOSE" if grip_val < 0 else "OPEN"
            grip_color = (255, 100, 100) if grip_val < 0 else green
            draw.text((10, y_offset), f"gripper: {grip_str} ({grip_val:.2f})", fill=grip_color, font=font_small)
            y_offset += 14

    # Policy info (if available, for brain-inspired policy)
    if policy_info and 'primitive_weights' in policy_info:
        y_offset += 10
        draw.text((10, y_offset), "Primitives:", fill=cyan, font=font_title)
        y_offset += 16

        weights = policy_info['primitive_weights']
        prim_names = ['reach', 'grasp', 'lift', 'move', 'place', 'release']

        for i, (name, w) in enumerate(zip(prim_names, weights)):
            # Draw mini bar
            bar_x = 10
            bar_width = 60
            bar_height = 10

            draw.text((bar_x, y_offset), f"{name[:4]}:", fill=white, font=font_small)
            bar_x = 50
            draw.rectangle([bar_x, y_offset, bar_x + bar_width, y_offset + bar_height], outline=white)
            fill_width = int(bar_width * min(w, 1.0))
            if fill_width > 0:
                draw.rectangle([bar_x, y_offset, bar_x + fill_width, y_offset + bar_height], fill=cyan)
            y_offset += 13

    # Reasoning (truncated)
    if reasoning:
        y_offset = 370
        reason_short = reasoning[:50] + "..." if len(reasoning) > 50 else reasoning
        draw.text((10, y_offset), reason_short, fill=(180, 180, 180), font=font_small)

    # Task description at very bottom
    task_short = task_description[:60] if task_description else ""
    draw.text((10, 385), task_short, fill=(150, 150, 150), font=font_small)

    return np.array(pil_canvas)


class VLMEvaluator:
    """Evaluator for VLM-conditioned policies on LIBERO."""

    def __init__(
        self,
        policy,
        vlm_planner=None,
        task_suite: str = "libero_spatial",
        task_id: int = 0,
        device: str = "cuda",
        use_real_vlm: bool = False,
    ):
        self.policy = policy
        self.vlm_planner = vlm_planner
        self.device = device
        self.use_real_vlm = use_real_vlm
        self.task_id = task_id

        # Load LIBERO task
        self.benchmark = get_benchmark(task_suite)()
        self.task = self.benchmark.get_task(task_id)
        self.task_name = self.task.name
        self.task_description = self.task.language

        print(f"Task: {self.task_name}")
        print(f"Description: {self.task_description}")

        # Create environment
        self.env = self._create_env()

        # Load init states manually to control weights_only
        self.init_states = self._load_init_states()

    def _get_init_states_folder(self):
        """Get path to init states folder."""
        from libero.libero import get_libero_path
        try:
            return get_libero_path("init_states")
        except:
            import libero.libero
            return os.path.join(os.path.dirname(libero.libero.__file__), "init_states")

    def _load_init_states(self):
        """Load init states with proper torch.load settings."""
        try:
            init_states_path = os.path.join(
                self._get_init_states_folder(),
                self.task.problem_folder,
                self.task.init_states_file,
            )
            return torch.load(init_states_path, weights_only=False)
        except Exception as e:
            print(f"[Warning] Could not load init states: {e}")
            return None

    def _create_env(self):
        """Create LIBERO environment."""
        # Get proper BDDL file path from benchmark
        bddl_file = self.benchmark.get_task_bddl_file_path(self.task_id)
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        return OffScreenRenderEnv(**env_args)

    def _get_state(self, obs: Dict) -> np.ndarray:
        """Extract state from observation."""
        ee_pos = obs['robot0_eef_pos']
        ee_quat = obs['robot0_eef_quat']
        gripper = obs['robot0_gripper_qpos']
        joints = obs['robot0_joint_pos']

        # Convert quaternion to axis-angle for consistency
        ee_ori = T.quat2axisangle(ee_quat)

        return np.concatenate([ee_pos, ee_ori, gripper, joints])

    def _get_gripper_state(self, obs: Dict) -> str:
        """Get gripper state string."""
        gripper = obs['robot0_gripper_qpos']
        return "closed" if gripper[0] < 0.02 else "open"

    def run_episode(
        self,
        episode_idx: int = 0,
        max_steps: int = 300,
        plan_interval: int = 20,
        record_video: bool = True,
    ) -> Dict:
        """
        Run a single evaluation episode.

        Args:
            episode_idx: Episode index (for init state selection)
            max_steps: Maximum steps per episode
            plan_interval: Steps between VLM queries
            record_video: Whether to record video frames

        Returns:
            Dictionary with episode results
        """
        # Reset environment
        self.env.reset()
        if self.init_states is not None:
            init_state = self.init_states[episode_idx % len(self.init_states)]
            obs = self.env.set_init_state(init_state)
        else:
            obs = self.env.reset()

        frames = []
        rewards = []
        actions_taken = []
        vlm_plans = []

        current_plan = None
        steps_since_plan = 0

        for step in tqdm(range(max_steps), desc=f"Episode {episode_idx}"):
            # Get image and state
            image = obs['agentview_image']
            state = self._get_state(obs)
            gripper_state = self._get_gripper_state(obs)

            # Query VLM if needed
            if step == 0 or steps_since_plan >= plan_interval:
                if self.use_real_vlm and self.vlm_planner is not None:
                    try:
                        current_plan = self.vlm_planner.plan(
                            image=image,
                            task_description=self.task_description,
                            gripper_state=gripper_state,
                            steps_since_plan=steps_since_plan,
                        )
                    except Exception as e:
                        print(f"VLM error at step {step}: {e}")
                        # Keep previous plan
                else:
                    # Use heuristic plan
                    current_plan = self._create_heuristic_plan(step, max_steps, gripper_state)

                steps_since_plan = 0

            # Get action from policy
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            policy_info = None
            with torch.no_grad():
                if hasattr(self.policy, 'forward'):
                    # Simple policy
                    action = self.policy([current_plan], state_tensor)
                    action = action.cpu().numpy().squeeze()
                else:
                    # VLMConditionedPolicy wrapper
                    action, info = self.policy.get_action(
                        image=image,
                        proprio=state,
                        task_description=self.task_description,
                        gripper_state=gripper_state,
                    )
                    current_plan = info.get('vlm_plan', current_plan)
                    policy_info = info.get('policy_info')

            # Execute action
            obs, reward, done, info = self.env.step(action)

            rewards.append(reward)
            actions_taken.append(action.copy())
            vlm_plans.append(current_plan)
            steps_since_plan += 1

            # Record frame
            if record_video:
                frame = create_visualization_frame(
                    image=image,
                    vlm_plan=current_plan,
                    step=step,
                    action=action,
                    policy_info=policy_info,
                    task_description=self.task_description,
                )
                frames.append(frame)

            if done:
                print(f"Task completed at step {step}!")
                break

        # Get final success
        success = self.env.check_success()

        return {
            'success': success,
            'total_reward': sum(rewards),
            'n_steps': len(rewards),
            'frames': frames,
            'actions': actions_taken,
            'vlm_plans': vlm_plans,
        }

    def _create_heuristic_plan(
        self,
        step: int,
        total_steps: int,
        gripper_state: str,
    ) -> Dict:
        """Create heuristic plan for testing."""
        progress = step / total_steps

        if progress < 0.2:
            phase = "approach"
            movements = [{"direction": "forward", "speed": "fast", "steps": 2}]
            gripper = "open"
        elif progress < 0.3:
            phase = "align"
            movements = [{"direction": "forward", "speed": "slow", "steps": 1}]
            gripper = "open"
        elif progress < 0.4:
            phase = "descend"
            movements = [{"direction": "down", "speed": "slow", "steps": 2}]
            gripper = "open"
        elif progress < 0.5:
            phase = "grasp"
            movements = []
            gripper = "close"
        elif progress < 0.6:
            phase = "lift"
            movements = [{"direction": "up", "speed": "medium", "steps": 2}]
            gripper = "maintain"
        elif progress < 0.8:
            phase = "move"
            movements = [{"direction": "right", "speed": "fast", "steps": 2}]
            gripper = "maintain"
        elif progress < 0.9:
            phase = "place"
            movements = [{"direction": "down", "speed": "slow", "steps": 2}]
            gripper = "maintain"
        else:
            phase = "release"
            movements = []
            gripper = "open"

        return {
            "observation": {
                "target_object": "target object",
                "distance_to_target": "medium",
            },
            "plan": {
                "phase": phase,
                "movements": movements,
                "gripper": gripper,
                "confidence": 0.8,
            },
            "reasoning": f"Heuristic plan for {phase} phase",
        }

    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps: int = 300,
        plan_interval: int = 20,
        output_dir: str = "eval_results",
        save_gifs: bool = True,
    ) -> Dict:
        """
        Run full evaluation.

        Args:
            n_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            plan_interval: Steps between VLM queries
            output_dir: Directory to save results
            save_gifs: Whether to save GIF visualizations

        Returns:
            Dictionary with evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'task_name': self.task_name,
            'task_description': self.task_description,
            'n_episodes': n_episodes,
            'successes': [],
            'rewards': [],
            'steps': [],
        }

        for ep in range(n_episodes):
            print(f"\n=== Episode {ep + 1}/{n_episodes} ===")

            ep_result = self.run_episode(
                episode_idx=ep,
                max_steps=max_steps,
                plan_interval=plan_interval,
                record_video=save_gifs,
            )

            results['successes'].append(ep_result['success'])
            results['rewards'].append(ep_result['total_reward'])
            results['steps'].append(ep_result['n_steps'])

            print(f"Success: {ep_result['success']}, Reward: {ep_result['total_reward']:.2f}, Steps: {ep_result['n_steps']}")

            # Save GIF
            if save_gifs and ep_result['frames']:
                gif_path = os.path.join(output_dir, f"episode_{ep:02d}.gif")
                imageio.mimsave(gif_path, ep_result['frames'], fps=10)
                print(f"Saved GIF: {gif_path}")

        # Compute summary statistics
        results['success_rate'] = np.mean(results['successes'])
        results['avg_reward'] = np.mean(results['rewards'])
        results['avg_steps'] = np.mean(results['steps'])

        print(f"\n=== Evaluation Summary ===")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Steps: {results['avg_steps']:.1f}")

        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['frames']}, f, indent=2, default=float)
        print(f"Saved results: {results_path}")

        return results

    def close(self):
        """Clean up environment."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained policy checkpoint")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--plan_interval", type=int, default=20)
    parser.add_argument("--use_real_vlm", action="store_true",
                        help="Use real VLM (requires GPU)")
    parser.add_argument("--vlm_model", type=str,
                        default="/workspace/src/models/qwen2.5-vl-7b")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/src/eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Create output directory with task info
    output_dir = os.path.join(
        args.output_dir,
        f"{args.task_suite}_task{args.task_id}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load policy
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)

        # Determine policy type
        policy_type = checkpoint.get('policy_type', 'simple')
        state_dim = checkpoint.get('state_dim', 15)
        action_dim = checkpoint.get('action_dim', 7)

        if policy_type == 'simple':
            from src.policy.vlm_policy import VLMConditionedPolicySimple
            policy = VLMConditionedPolicySimple(
                plan_dim=128,
                proprio_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
            )
        else:
            from src.action_generator.brain_model import BrainInspiredActionGenerator
            policy = BrainInspiredActionGenerator(
                plan_dim=128,
                proprio_dim=state_dim,
                action_dim=action_dim,
                chunk_size=10,
            )

        policy.load_state_dict(checkpoint['model_state_dict'])
        policy = policy.to(args.device)
        policy.eval()
    else:
        print("No checkpoint provided, using untrained policy")
        from src.policy.vlm_policy import VLMConditionedPolicySimple
        policy = VLMConditionedPolicySimple(
            plan_dim=128,
            proprio_dim=15,
            action_dim=7,
            hidden_dim=256,
        ).to(args.device)
        policy.eval()

    # Load VLM if using real VLM
    vlm_planner = None
    if args.use_real_vlm:
        print("Loading VLM planner...")
        from src.vlm.qwen_planner import QwenVLPlanner
        vlm_planner = QwenVLPlanner(model_name=args.vlm_model)

    # Create evaluator
    evaluator = VLMEvaluator(
        policy=policy,
        vlm_planner=vlm_planner,
        task_suite=args.task_suite,
        task_id=args.task_id,
        device=args.device,
        use_real_vlm=args.use_real_vlm,
    )

    # Run evaluation
    try:
        results = evaluator.evaluate(
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            plan_interval=args.plan_interval,
            output_dir=output_dir,
            save_gifs=True,
        )
    finally:
        evaluator.close()

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
