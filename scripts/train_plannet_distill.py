#!/usr/bin/env python3
"""
Phase C: Distill VLM planning into fast PlanNet.

Architecture:
1. PlanNet: Image → Plan embedding (replaces VLM)
2. ActionNet: Plan embedding + proprio → action (BC trained)

This removes the VLM from the inner loop entirely.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

from brain_robot.env.mock_env import make_mock_env


class PlanNet(nn.Module):
    """
    Fast plan predictor from image.
    Replaces VLM - predicts plan embedding directly.

    For mock env, we use positions instead of images since
    the mock env doesn't have realistic images.
    """
    def __init__(self, input_dim: int = 9, plan_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        # Input: robot_pos (3) + object_pos (3) + target_pos (3)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plan_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActionNet(nn.Module):
    """
    Action predictor from plan + proprio.
    Trained with BC from expert demonstrations.
    """
    def __init__(self, plan_dim: int = 32, proprio_dim: int = 2, action_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        # proprio_dim = 2: gripper_open (1) + object_grasped (1)
        self.net = nn.Sequential(
            nn.Linear(plan_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, plan_embed, proprio):
        x = torch.cat([plan_embed, proprio], dim=-1)
        return self.net(x)


class DistilledPolicy(nn.Module):
    """Combined PlanNet + ActionNet."""
    def __init__(self, plan_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.plan_net = PlanNet(input_dim=9, plan_dim=plan_dim, hidden_dim=hidden_dim)
        self.action_net = ActionNet(plan_dim=plan_dim, proprio_dim=2, action_dim=7, hidden_dim=hidden_dim)

    def forward(self, state_input, proprio):
        """
        state_input: robot_pos + object_pos + target_pos (9,)
        proprio: gripper_open + object_grasped (2,)
        """
        plan_embed = self.plan_net(state_input)
        action = self.action_net(plan_embed, proprio)
        return action


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
            action[6] = -1.0  # Open
        else:
            action = np.zeros(7)
            action[6] = 1.0  # Close
    else:
        if dist_to_target > 0.08:
            direction = target_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = 1.0  # Keep closed
        else:
            action = np.zeros(7)
            action[6] = -1.0  # Open

    return action


def collect_expert_demos(num_demos=1000):
    """Collect demonstrations."""
    print("Collecting expert demonstrations...")
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    state_data = []  # robot_pos + object_pos + target_pos
    proprio_data = []  # gripper_open + object_grasped
    action_data = []
    successes = 0

    for ep in range(num_demos):
        obs, info = env.reset()
        for step in range(100):
            # State input for PlanNet
            state_input = np.concatenate([
                env.robot_pos,
                env.object_pos,
                env.target_pos,
            ]).astype(np.float32)

            # Proprio for ActionNet
            proprio = np.array([
                1.0 if env.gripper_open else 0.0,
                1.0 if env.object_grasped else 0.0,
            ], dtype=np.float32)

            action = get_expert_action(env)

            state_data.append(state_input)
            proprio_data.append(proprio)
            action_data.append(action)

            next_obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    print(f"Expert success rate: {successes/num_demos*100:.1f}%")
    print(f"Collected {len(state_data)} transitions")

    return np.array(state_data), np.array(proprio_data), np.array(action_data)


def train_distilled_policy(state_data, proprio_data, action_data, device, epochs=200):
    """Train the distilled policy with BC."""
    print("\nTraining distilled policy...")

    policy = DistilledPolicy(plan_dim=32, hidden_dim=128).to(device)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch_size = 256

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        total_loss = 0
        num_batches = 0

        for start in range(0, len(state_data), batch_size):
            end = min(start + batch_size, len(state_data))
            mb_idx = indices[start:end]

            mb_state = state_t[mb_idx]
            mb_proprio = proprio_t[mb_idx]
            mb_actions = action_t[mb_idx]

            pred_actions = policy(mb_state, mb_proprio)
            loss = F.mse_loss(pred_actions, mb_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/num_batches:.4f}")

    return policy


def evaluate_policy(policy, device, num_episodes=100):
    """Evaluate the distilled policy."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    successes = 0
    total_reward = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0

        for step in range(100):
            state_input = np.concatenate([
                env.robot_pos,
                env.object_pos,
                env.target_pos,
            ]).astype(np.float32)

            proprio = np.array([
                1.0 if env.gripper_open else 0.0,
                1.0 if env.object_grasped else 0.0,
            ], dtype=np.float32)

            state_t = torch.tensor(state_input, device=device).unsqueeze(0)
            proprio_t = torch.tensor(proprio, device=device).unsqueeze(0)

            with torch.no_grad():
                action = policy(state_t, proprio_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(action_np)
            ep_reward += reward

            if done or truncated:
                break

        total_reward += ep_reward
        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100, total_reward / num_episodes


def benchmark_inference_speed(policy, device, num_iterations=1000):
    """Benchmark inference speed."""
    state_input = torch.randn(1, 9, device=device)
    proprio = torch.randn(1, 2, device=device)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            policy(state_input, proprio)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            policy(state_input, proprio)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    return num_iterations / elapsed  # Inferences per second


def train():
    print("=" * 60)
    print("Phase C: Distilled PlanNet + ActionNet")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Collect expert demos
    state_data, proprio_data, action_data = collect_expert_demos(num_demos=500)

    # Step 2: Train distilled policy
    policy = train_distilled_policy(state_data, proprio_data, action_data, device, epochs=200)

    # Step 3: Evaluate
    success_rate, avg_reward = evaluate_policy(policy, device, num_episodes=100)
    print(f"\nDistilled policy: Success={success_rate:.1f}%, Avg reward={avg_reward:.1f}")

    # Step 4: Benchmark speed
    fps = benchmark_inference_speed(policy, device)
    print(f"Inference speed: {fps:.0f} FPS")

    # Compare to VLM speed (approximate)
    print(f"\nComparison:")
    print(f"  Distilled policy: ~{fps:.0f} FPS")
    print(f"  VLM (Qwen 7B): ~0.5-1 FPS")
    print(f"  Speedup: ~{fps/1:.0f}x")

    # Save checkpoint
    checkpoint_path = "/workspace/brain_robot/checkpoints/distilled_policy.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'policy': policy.state_dict(),
        'success_rate': success_rate,
    }, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")

    return success_rate


if __name__ == "__main__":
    train()
