#!/usr/bin/env python3
"""
Debug what the policy is actually doing in pick-and-place.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.env.mock_env import make_mock_env


def debug_random_policy():
    """See what happens with random actions."""
    print("=" * 60)
    print("Debugging with Random Policy")
    print("=" * 60)

    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    successes = 0
    for ep in range(100):
        obs, info = env.reset()
        ep_reward = 0
        trajectory = []

        for step in range(100):
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)

            trajectory.append({
                'step': step,
                'ee_pos': env.robot_pos.copy(),
                'obj_pos': env.object_pos.copy(),
                'grasped': env.object_grasped,
                'gripper_open': env.gripper_open,
                'reward': reward,
            })

            ep_reward += reward
            obs = next_obs

            if done or truncated:
                break

        if info.get('success', False):
            successes += 1
            print(f"\nSuccess on ep {ep}!")
            print(f"  Final obj pos: {env.object_pos}")
            print(f"  Target pos: {env.target_pos}")

    print(f"\nRandom policy success rate: {successes}%")


def debug_scripted_policy():
    """Test a hand-crafted policy."""
    print("\n" + "=" * 60)
    print("Debugging with Scripted Policy")
    print("=" * 60)

    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    successes = 0
    for ep in range(100):
        obs, info = env.reset()
        ep_reward = 0

        for step in range(100):
            ee_pos = env.robot_pos
            obj_pos = env.object_pos
            target_pos = env.target_pos

            # Simple state machine
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
            dist_to_target = np.linalg.norm(obj_pos[:2] - target_pos[:2])

            if not env.object_grasped:
                # Phase 1: Move to object
                if dist_to_obj > 0.08:
                    direction = obj_pos - ee_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    # Scale to max action
                    action = np.zeros(7)
                    action[:3] = direction * 1.0  # Full speed
                    action[6] = -1.0  # Open gripper
                else:
                    # Phase 2: Close gripper
                    action = np.zeros(7)
                    action[6] = 1.0  # Close gripper
            else:
                # Phase 3: Move to target
                if dist_to_target > 0.08:
                    direction = target_pos - ee_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    action = np.zeros(7)
                    action[:3] = direction * 1.0
                    action[6] = 1.0  # Keep closed
                else:
                    # Phase 4: Release
                    action = np.zeros(7)
                    action[6] = -1.0  # Open

            next_obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    print(f"Scripted policy success rate: {successes}%")


def debug_env_mechanics():
    """Debug the environment mechanics."""
    print("\n" + "=" * 60)
    print("Debugging Environment Mechanics")
    print("=" * 60)

    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    obs, info = env.reset()

    print(f"Initial state:")
    print(f"  Robot pos: {env.robot_pos}")
    print(f"  Object pos: {env.object_pos}")
    print(f"  Target pos: {env.target_pos}")
    print(f"  Distance obj->target: {np.linalg.norm(env.object_pos[:2] - env.target_pos[:2]):.3f}")

    # Move toward object
    print("\nMoving toward object...")
    for _ in range(20):
        direction = env.object_pos - env.robot_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        action = np.zeros(7)
        action[:3] = direction
        action[6] = -1.0  # Open
        env.step(action)

    print(f"  Robot pos: {env.robot_pos}")
    print(f"  Distance to object: {np.linalg.norm(env.robot_pos - env.object_pos):.3f}")
    print(f"  Object grasped: {env.object_grasped}")

    # Close gripper
    print("\nClosing gripper...")
    for _ in range(5):
        action = np.zeros(7)
        action[6] = 1.0
        env.step(action)

    print(f"  Gripper open: {env.gripper_open}")
    print(f"  Object grasped: {env.object_grasped}")
    print(f"  Distance to object: {np.linalg.norm(env.robot_pos - env.object_pos):.3f}")

    # Move to target
    print("\nMoving toward target...")
    for _ in range(30):
        direction = env.target_pos - env.robot_pos
        direction[:2] = direction[:2] / (np.linalg.norm(direction[:2]) + 1e-8)
        direction[2] = 0  # Keep height
        action = np.zeros(7)
        action[:3] = direction
        action[6] = 1.0  # Keep closed
        env.step(action)

    print(f"  Robot pos: {env.robot_pos}")
    print(f"  Object pos: {env.object_pos}")
    print(f"  Distance obj->target: {np.linalg.norm(env.object_pos[:2] - env.target_pos[:2]):.3f}")

    # Release
    print("\nReleasing...")
    for _ in range(5):
        action = np.zeros(7)
        action[6] = -1.0
        next_obs, reward, done, truncated, info = env.step(action)

    print(f"  Gripper open: {env.gripper_open}")
    print(f"  Object grasped: {env.object_grasped}")
    print(f"  Final object pos: {env.object_pos}")
    print(f"  Success: {info.get('success', False)}")


if __name__ == "__main__":
    debug_env_mechanics()
    debug_scripted_policy()
    debug_random_policy()
