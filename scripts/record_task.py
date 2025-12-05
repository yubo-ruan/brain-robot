#!/usr/bin/env python3
"""
Record task execution as GIF.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
import imageio

from src.env.mock_env import make_mock_env


def get_expert_action(env):
    """Scripted expert policy."""
    ee_pos = env.robot_pos
    obj_pos = env.object_pos
    target_pos = env.target_pos
    dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_to_target = np.linalg.norm(obj_pos[:2] - target_pos[:2])

    if not env.object_grasped:
        if dist_to_obj > 0.08:
            direction = obj_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = -1.0
            phase = "approach"
        else:
            action = np.zeros(7)
            action[6] = 1.0
            phase = "grasp"
    else:
        if dist_to_target > 0.08:
            direction = target_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = 1.0
            phase = "transport"
        else:
            action = np.zeros(7)
            action[6] = -1.0
            phase = "release"

    return action, phase


def render_enhanced(env, phase, step, success=False):
    """Render enhanced visualization with labels."""
    # Get base image
    image = env.render().copy()

    # Scale up for better visibility
    scale = 2
    h, w = image.shape[:2]
    image_large = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            image_large[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = image[i, j]

    # Add text info using simple drawing
    # Draw phase indicator at top
    phase_colors = {
        "approach": [255, 255, 0],   # Yellow
        "grasp": [255, 128, 0],      # Orange
        "transport": [0, 255, 255],  # Cyan
        "release": [0, 255, 0],      # Green
    }

    # Draw colored bar at top based on phase
    color = phase_colors.get(phase, [255, 255, 255])
    image_large[:20, :] = color

    # Draw step counter as simple bar
    bar_width = int((step / 100) * w * scale)
    image_large[h*scale-10:, :bar_width] = [100, 100, 100]

    if success:
        # Green border for success
        image_large[:5, :] = [0, 255, 0]
        image_large[-5:, :] = [0, 255, 0]
        image_large[:, :5] = [0, 255, 0]
        image_large[:, -5:] = [0, 255, 0]

    return image_large


def record_episode(output_path, use_learned=False, checkpoint_path=None):
    """Record a single episode."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    # Load learned policy if specified
    policy = None
    if use_learned and checkpoint_path and os.path.exists(checkpoint_path):
        from scripts.train_plannet_distill import DistilledPolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = DistilledPolicy(plan_dim=32, hidden_dim=128).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint['policy'])
        policy.eval()
        print(f"Loaded policy from {checkpoint_path}")

    frames = []
    obs, info = env.reset()

    for step in range(100):
        if policy is not None:
            # Use learned policy
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)

            state_t = torch.tensor(state, device=device).unsqueeze(0)
            proprio_t = torch.tensor(proprio, device=device).unsqueeze(0)

            with torch.no_grad():
                action = policy(state_t, proprio_t)
            action = action.squeeze(0).cpu().numpy()
            _, phase = get_expert_action(env)  # Just for visualization
        else:
            # Use expert policy
            action, phase = get_expert_action(env)

        # Render frame
        frame = render_enhanced(env, phase, step)
        frames.append(frame)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        if done:
            # Add a few success frames
            for _ in range(10):
                frame = render_enhanced(env, "success", step, success=True)
                frames.append(frame)
            break

        if truncated:
            break

    env.close()

    # Save as GIF
    print(f"Saving {len(frames)} frames to {output_path}")
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print(f"Saved GIF to {output_path}")

    return info.get('success', False)


def record_multiple_episodes(output_dir, num_episodes=3):
    """Record multiple episodes."""
    os.makedirs(output_dir, exist_ok=True)

    successes = 0
    for i in range(num_episodes):
        output_path = os.path.join(output_dir, f"episode_{i+1}.gif")
        success = record_episode(output_path)
        if success:
            successes += 1
        print(f"Episode {i+1}: {'Success' if success else 'Failed'}")

    print(f"\nTotal: {successes}/{num_episodes} successful")


def create_comparison_gif(output_path):
    """Create side-by-side comparison of expert vs random."""
    env_expert = make_mock_env(max_episode_steps=100, action_scale=1.0)
    env_random = make_mock_env(max_episode_steps=100, action_scale=1.0)

    # Use same seed for fair comparison
    np.random.seed(42)
    obs_expert, _ = env_expert.reset(seed=42)
    obs_random, _ = env_random.reset(seed=42)

    frames = []

    for step in range(100):
        # Expert action
        action_expert, phase = get_expert_action(env_expert)
        frame_expert = render_enhanced(env_expert, phase, step)

        # Random action
        action_random = env_random.action_space.sample()
        frame_random = render_enhanced(env_random, "random", step)

        # Combine side by side
        combined = np.concatenate([frame_expert, frame_random], axis=1)

        # Add separator line
        combined[:, frame_expert.shape[1]-2:frame_expert.shape[1]+2] = [255, 255, 255]

        frames.append(combined)

        # Step both
        _, _, done_e, trunc_e, info_e = env_expert.step(action_expert)
        _, _, done_r, trunc_r, info_r = env_random.step(action_random)

        if done_e or trunc_e:
            # Add success frames for expert
            for _ in range(10):
                frame_expert = render_enhanced(env_expert, "success", step, success=info_e.get('success', False))
                frame_random = render_enhanced(env_random, "random", step)
                combined = np.concatenate([frame_expert, frame_random], axis=1)
                combined[:, frame_expert.shape[1]-2:frame_expert.shape[1]+2] = [255, 255, 255]
                frames.append(combined)
            break

    env_expert.close()
    env_random.close()

    print(f"Saving comparison GIF with {len(frames)} frames")
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    output_dir = "/workspace/src/recordings"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Recording Task Executions")
    print("=" * 60)

    # Record expert policy
    print("\n1. Recording expert policy episode...")
    record_episode(os.path.join(output_dir, "expert_episode.gif"))

    # Record comparison
    print("\n2. Recording expert vs random comparison...")
    create_comparison_gif(os.path.join(output_dir, "expert_vs_random.gif"))

    print(f"\nRecordings saved to {output_dir}/")
