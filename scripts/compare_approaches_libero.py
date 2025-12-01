#!/usr/bin/env python3
"""
Compare Simple MLP vs Brain-Inspired approaches on LIBERO tasks.

This script:
1. Tests both approaches on LIBERO benchmark tasks
2. Generates GIF recordings with VLM output visualization
3. Reports comparative metrics (success rate, FPS, etc.)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set rendering backend
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import torch.nn as nn
import time
import imageio
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat):
    """Convert quaternion (w, x, y, z) to euler angles."""
    # scipy uses (x, y, z, w) format
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return r.as_euler('xyz')


# ============================================================================
# Policy Architectures
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP policy (proven to work well)."""

    def __init__(self, state_dim: int = 32, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class BrainInspiredPolicy(nn.Module):
    """
    Brain-inspired architecture with:
    - VLM-based plan encoding
    - Primitive selection
    - Motor execution with modulation
    """

    def __init__(
        self,
        state_dim: int = 32,
        plan_dim: int = 64,
        action_dim: int = 7,
        hidden_dim: int = 256,
        n_primitives: int = 4,
    ):
        super().__init__()

        # Plan encoder (simulates VLM output)
        self.plan_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plan_dim),
        )

        # Primitive selector
        self.primitive_selector = nn.Sequential(
            nn.Linear(plan_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_primitives),
            nn.Softmax(dim=-1),
        )

        # Motion primitives
        self.primitives = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh(),
            )
            for _ in range(n_primitives)
        ])

        # Modulator
        self.modulator = nn.Sequential(
            nn.Linear(plan_dim + state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        self.n_primitives = n_primitives

    def forward(
        self,
        state: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor:
        # Encode plan
        plan = self.plan_encoder(state)

        # Select primitive
        selector_input = torch.cat([plan, state], dim=-1)
        primitive_weights = self.primitive_selector(selector_input)

        # Execute primitives
        primitive_outputs = torch.stack([
            prim(state) for prim in self.primitives
        ], dim=1)  # [B, n_primitives, action_dim]

        # Weighted combination
        base_action = torch.sum(
            primitive_outputs * primitive_weights.unsqueeze(-1),
            dim=1
        )

        # Modulate
        modulation = self.modulator(selector_input)
        action = base_action + 0.1 * modulation
        action = torch.clamp(action, -1, 1)

        if return_info:
            info = {
                'plan': plan,
                'primitive_weights': primitive_weights,
                'base_action': base_action,
                'modulation': modulation,
            }
            return action, info

        return action


# ============================================================================
# VLM Interface (Mock for visualization)
# ============================================================================

class MockVLM:
    """Mock VLM for generating plan descriptions."""

    def __init__(self):
        self.phase_names = [
            "approach",
            "grasp",
            "lift",
            "transport",
            "place",
            "release",
        ]
        self.current_phase = 0
        self.last_update = 0

    def get_plan(self, image: np.ndarray, task_description: str, step: int) -> Dict:
        """Generate mock VLM plan output."""
        # Update phase based on step
        if step - self.last_update > 30:
            self.current_phase = min(self.current_phase + 1, len(self.phase_names) - 1)
            self.last_update = step

        phase = self.phase_names[self.current_phase]

        return {
            'phase': phase,
            'instruction': f"Current phase: {phase}. {task_description}",
            'confidence': 0.85 + 0.1 * np.random.random(),
            'primitive_suggestion': self.current_phase % 4,
        }

    def reset(self):
        self.current_phase = 0
        self.last_update = 0


# ============================================================================
# Visualization Utilities
# ============================================================================

def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 12,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    """Draw text on image with optional background."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)

    # Draw background
    if bg_color is not None:
        padding = 2
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=bg_color
        )

    # Draw text
    draw.text(position, text, fill=color, font=font)

    return np.array(pil_image)


def create_comparison_frame(
    env_image: np.ndarray,
    vlm_output: Dict,
    policy_info: Dict,
    approach_name: str,
    step: int,
    reward: float,
    success: bool,
) -> np.ndarray:
    """Create visualization frame with VLM output and policy info."""
    h, w = env_image.shape[:2]

    # Create larger canvas for info panel
    canvas_h = h + 100
    canvas_w = w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place environment image
    canvas[:h, :w] = env_image

    # Add border based on success
    if success:
        canvas[:5, :] = [0, 255, 0]
        canvas[-5:, :] = [0, 255, 0]
        canvas[:, :5] = [0, 255, 0]
        canvas[:, -5:] = [0, 255, 0]

    # Info panel background
    canvas[h:, :] = [40, 40, 40]

    # Draw info text
    canvas = draw_text_on_image(
        canvas, f"{approach_name} | Step: {step}", (5, h + 5), font_size=10
    )
    canvas = draw_text_on_image(
        canvas, f"VLM Phase: {vlm_output.get('phase', 'N/A')}", (5, h + 20), font_size=10
    )
    canvas = draw_text_on_image(
        canvas, f"Conf: {vlm_output.get('confidence', 0):.2f}", (5, h + 35), font_size=10
    )

    # Draw primitive weights if available
    if 'primitive_weights' in policy_info:
        weights = policy_info['primitive_weights']
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights_str = "Prims: " + " ".join([f"{w:.2f}" for w in weights.flatten()[:4]])
        canvas = draw_text_on_image(canvas, weights_str, (5, h + 50), font_size=10)

    # Draw phase bar
    phase_colors = {
        'approach': (255, 255, 0),
        'grasp': (255, 128, 0),
        'lift': (0, 255, 255),
        'transport': (0, 128, 255),
        'place': (128, 255, 0),
        'release': (0, 255, 0),
    }
    phase = vlm_output.get('phase', 'approach')
    phase_color = phase_colors.get(phase, (128, 128, 128))
    canvas[h + 80:h + 90, 5:w-5] = phase_color

    return canvas


def create_side_by_side_frame(
    simple_frame: np.ndarray,
    brain_frame: np.ndarray,
    step: int,
) -> np.ndarray:
    """Create side-by-side comparison frame."""
    h1, w1 = simple_frame.shape[:2]
    h2, w2 = brain_frame.shape[:2]

    h = max(h1, h2)
    w = w1 + w2 + 10  # 10 pixel separator

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = simple_frame
    canvas[:, w1:w1+10] = [128, 128, 128]  # Separator
    canvas[:h2, w1+10:] = brain_frame

    return canvas


# ============================================================================
# Training and Evaluation
# ============================================================================

def load_expert_demonstrations(
    task_suite: str,
    task_name: str,
    n_demos: int = 50,
    data_dir: str = "/workspace/data/libero",
) -> List[Dict]:
    """Load expert demonstrations from LIBERO HDF5 files."""
    import h5py

    # Construct demo file path
    demo_path = os.path.join(data_dir, task_suite, f"{task_name}_demo.hdf5")

    if not os.path.exists(demo_path):
        print(f"[Warning] Demo file not found: {demo_path}")
        return []

    demos = []
    with h5py.File(demo_path, 'r') as f:
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')],
                          key=lambda x: int(x.split('_')[1]))

        for i, dk in enumerate(demo_keys[:n_demos]):
            demo_data = f['data'][dk]

            # Extract state: ee_pos (3) + ee_ori (3) + gripper_states (2) + joint_states (7) = 15
            ee_pos = demo_data['obs/ee_pos'][:]
            ee_ori = demo_data['obs/ee_ori'][:]
            gripper = demo_data['obs/gripper_states'][:]
            joints = demo_data['obs/joint_states'][:]

            states = np.concatenate([ee_pos, ee_ori, gripper, joints], axis=1)
            actions = demo_data['actions'][:]

            demo = {
                'states': list(states),
                'actions': list(actions),
                'rewards': list(demo_data['rewards'][:]),
            }
            demos.append(demo)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{min(n_demos, len(demo_keys))} demos")

    print(f"  Total demos: {len(demos)}, Total transitions: {sum(len(d['states']) for d in demos)}")
    return demos


def collect_demonstrations(env, n_demos: int = 50, max_steps: int = 200) -> List[Dict]:
    """Collect demonstrations using random exploration (fallback)."""
    demos = []

    for i in range(n_demos):
        obs, info = env.reset()
        demo = {'states': [], 'actions': [], 'rewards': []}

        for step in range(max_steps):
            # Simple heuristic policy for demo collection
            action = env.action_space.sample() * 0.5

            # Get state
            state = obs['proprio']
            demo['states'].append(state)
            demo['actions'].append(action)

            obs, reward, done, truncated, info = env.step(action)
            demo['rewards'].append(reward)

            if done or truncated:
                break

        demos.append(demo)
        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{n_demos} demos")

    return demos


def train_bc(policy, demos, n_epochs: int = 100, lr: float = 1e-3):
    """Train policy using behavioral cloning."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Prepare data
    states = []
    actions = []
    for demo in demos:
        states.extend(demo['states'])
        actions.extend(demo['actions'])

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32)

    # Pad states if needed
    if states.shape[1] < 32:
        padding = torch.zeros(states.shape[0], 32 - states.shape[1])
        states = torch.cat([states, padding], dim=1)

    dataset = torch.utils.data.TensorDataset(states, actions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            pred_actions = policy(batch_states)
            loss = nn.MSELoss()(pred_actions, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate_policy(
    env,
    policy,
    vlm: MockVLM,
    approach_name: str,
    n_episodes: int = 10,
    max_steps: int = 200,
    record: bool = True,
) -> Tuple[Dict, List[np.ndarray]]:
    """Evaluate policy and optionally record frames."""

    policy.eval()
    vlm.reset()

    results = {
        'success_rate': 0,
        'avg_reward': 0,
        'avg_steps': 0,
        'fps': 0,
    }

    all_frames = []
    successes = 0
    total_reward = 0
    total_steps = 0

    start_time = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset()
        vlm.reset()
        episode_reward = 0
        episode_frames = []

        for step in range(max_steps):
            # Get state - extract in exact same format as training demos
            # Demo format: ee_pos (3) + ee_ori (3 euler) + gripper (2) + joints (7) = 15
            # Access the inner robosuite env's observations
            try:
                raw_obs = env._env.env._get_observations()
            except:
                raw_obs = None

            if raw_obs is not None:
                ee_pos = raw_obs.get('robot0_eef_pos', np.zeros(3))
                ee_quat = raw_obs.get('robot0_eef_quat', np.array([1, 0, 0, 0]))
                ee_ori = quat_to_euler(ee_quat)  # Convert quat to euler
                gripper = raw_obs.get('robot0_gripper_qpos', np.zeros(2))
                joints = raw_obs.get('robot0_joint_pos', np.zeros(7))
                state = np.concatenate([ee_pos, ee_ori, gripper, joints]).astype(np.float32)
            else:
                state = obs['proprio']

            # Pad to 32 dims for model compatibility
            if len(state) < 32:
                state = np.pad(state, (0, 32 - len(state)))
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get action
            with torch.no_grad():
                if hasattr(policy, 'forward') and 'return_info' in policy.forward.__code__.co_varnames:
                    action, policy_info = policy(state_tensor, return_info=True)
                    policy_info = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                                  for k, v in policy_info.items()}
                else:
                    action = policy(state_tensor)
                    policy_info = {}
                action = action.squeeze(0).numpy()

            # Get VLM output
            image = obs.get('image', np.zeros((128, 128, 3), dtype=np.uint8))
            vlm_output = vlm.get_plan(image, info.get('task_description', ''), step)

            # Record frame
            if record and ep == 0:  # Only record first episode
                frame = create_comparison_frame(
                    image, vlm_output, policy_info, approach_name,
                    step, episode_reward, False
                )
                episode_frames.append(frame)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if done:
                successes += 1
                # Add success frames
                if record and ep == 0:
                    for _ in range(10):
                        frame = create_comparison_frame(
                            image, vlm_output, policy_info, approach_name,
                            step, episode_reward, True
                        )
                        episode_frames.append(frame)
                break

            if truncated:
                break

        total_reward += episode_reward
        total_steps += step + 1

        if ep == 0:
            all_frames = episode_frames

    elapsed = time.time() - start_time

    results['success_rate'] = successes / n_episodes
    results['avg_reward'] = total_reward / n_episodes
    results['avg_steps'] = total_steps / n_episodes
    results['fps'] = total_steps / elapsed

    return results, all_frames


# ============================================================================
# Main Comparison
# ============================================================================

def run_comparison(
    task_suite: str = "libero_spatial",
    task_id: int = 0,
    n_demos: int = 30,
    n_eval_episodes: int = 5,
    output_dir: str = "/workspace/brain_robot/recordings/libero_comparison",
):
    """Run full comparison between approaches."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LIBERO Approach Comparison: Simple MLP vs Brain-Inspired")
    print("=" * 70)

    # Create environment
    print(f"\nCreating LIBERO environment: {task_suite}, task {task_id}")
    from brain_robot.env.libero_wrapper import make_libero_env

    env = make_libero_env(
        task_suite=task_suite,
        task_id=task_id,
        max_episode_steps=200,
    )

    print(f"Task: {env.task_description}")

    # Initialize VLM
    vlm = MockVLM()

    # Load expert demonstrations from LIBERO dataset
    print("\n" + "-" * 50)
    print("Loading expert demonstrations...")
    task_name = env.task.name
    demos = load_expert_demonstrations(task_suite, task_name, n_demos=n_demos)

    if not demos:
        print("[Warning] No expert demos found, falling back to random collection")
        demos = collect_demonstrations(env, n_demos=n_demos)

    # Create policies
    simple_policy = SimpleMLP(state_dim=32, action_dim=7, hidden_dim=256)
    brain_policy = BrainInspiredPolicy(state_dim=32, action_dim=7, hidden_dim=256)

    # Train Simple MLP
    print("\n" + "-" * 50)
    print("Training Simple MLP (Behavioral Cloning)...")
    train_bc(simple_policy, demos, n_epochs=200, lr=3e-4)

    # Train Brain-Inspired
    print("\n" + "-" * 50)
    print("Training Brain-Inspired Policy (Behavioral Cloning)...")
    train_bc(brain_policy, demos, n_epochs=200, lr=3e-4)

    # Evaluate Simple MLP
    print("\n" + "-" * 50)
    print("Evaluating Simple MLP...")
    simple_results, simple_frames = evaluate_policy(
        env, simple_policy, vlm, "Simple MLP",
        n_episodes=n_eval_episodes, record=True
    )
    print(f"  Success Rate: {simple_results['success_rate']*100:.1f}%")
    print(f"  Avg Reward: {simple_results['avg_reward']:.2f}")
    print(f"  FPS: {simple_results['fps']:.1f}")

    # Evaluate Brain-Inspired
    print("\n" + "-" * 50)
    print("Evaluating Brain-Inspired Policy...")
    brain_results, brain_frames = evaluate_policy(
        env, brain_policy, vlm, "Brain-Inspired",
        n_episodes=n_eval_episodes, record=True
    )
    print(f"  Success Rate: {brain_results['success_rate']*100:.1f}%")
    print(f"  Avg Reward: {brain_results['avg_reward']:.2f}")
    print(f"  FPS: {brain_results['fps']:.1f}")

    # Create side-by-side comparison GIF
    print("\n" + "-" * 50)
    print("Creating comparison GIF...")

    # Save individual GIFs
    if simple_frames:
        simple_path = os.path.join(output_dir, "simple_mlp.gif")
        imageio.mimsave(simple_path, simple_frames, fps=15, loop=0)
        print(f"  Saved: {simple_path}")

    if brain_frames:
        brain_path = os.path.join(output_dir, "brain_inspired.gif")
        imageio.mimsave(brain_path, brain_frames, fps=15, loop=0)
        print(f"  Saved: {brain_path}")

    # Create side-by-side comparison
    if simple_frames and brain_frames:
        n_frames = min(len(simple_frames), len(brain_frames))
        comparison_frames = []
        for i in range(n_frames):
            combined = create_side_by_side_frame(simple_frames[i], brain_frames[i], i)
            comparison_frames.append(combined)

        comparison_path = os.path.join(output_dir, "comparison.gif")
        imageio.mimsave(comparison_path, comparison_frames, fps=15, loop=0)
        print(f"  Saved: {comparison_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Simple MLP':<15} {'Brain-Inspired':<15}")
    print("-" * 50)
    print(f"{'Success Rate':<20} {simple_results['success_rate']*100:>13.1f}% {brain_results['success_rate']*100:>13.1f}%")
    print(f"{'Avg Reward':<20} {simple_results['avg_reward']:>14.2f} {brain_results['avg_reward']:>14.2f}")
    print(f"{'Avg Steps':<20} {simple_results['avg_steps']:>14.1f} {brain_results['avg_steps']:>14.1f}")
    print(f"{'FPS':<20} {simple_results['fps']:>14.1f} {brain_results['fps']:>14.1f}")

    # Determine winner
    print("\n" + "-" * 50)
    if simple_results['success_rate'] > brain_results['success_rate']:
        print("Winner: Simple MLP (higher success rate)")
    elif brain_results['success_rate'] > simple_results['success_rate']:
        print("Winner: Brain-Inspired (higher success rate)")
    else:
        if simple_results['fps'] > brain_results['fps']:
            print("Winner: Simple MLP (same success rate, faster)")
        else:
            print("Tie: Both approaches perform similarly")

    env.close()

    return {
        'simple': simple_results,
        'brain': brain_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_demos", type=int, default=30)
    parser.add_argument("--n_eval", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/brain_robot/recordings/libero_comparison")
    args = parser.parse_args()

    run_comparison(
        task_suite=args.task_suite,
        task_id=args.task_id,
        n_demos=args.n_demos,
        n_eval_episodes=args.n_eval,
        output_dir=args.output_dir,
    )
