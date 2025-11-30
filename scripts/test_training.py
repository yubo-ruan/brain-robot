#!/usr/bin/env python3
"""
Test training script using mock environment.
Tests the full training pipeline without LIBERO.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm

from brain_robot.vlm.qwen_planner import QwenVLPlanner
from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator
from brain_robot.env.mock_env import make_mock_env
from brain_robot.training.rewards import RewardShaper


def test_training_loop(num_episodes: int = 5, verbose: bool = True):
    """
    Test the training loop with mock environment.

    Args:
        num_episodes: Number of episodes to run
        verbose: Whether to print detailed output
    """
    print("="*60)
    print("Testing Training Loop with Mock Environment")
    print("="*60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create mock environment
    print("\nCreating mock environment...")
    env = make_mock_env(max_episode_steps=50, action_scale=1.0)
    print(f"Task: {env.task_description}")

    # Create VLM planner
    print("\nLoading VLM planner...")
    planner = QwenVLPlanner(
        model_name="/workspace/brain_robot/models/qwen2.5-vl-7b",
        device=device,
        max_new_tokens=512,
        temperature=0.1,
    )
    replan_every = 10

    # Create action generator
    print("\nCreating action generator...")
    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)

    # Create critic
    critic = nn.Sequential(
        nn.Linear(128 + 15, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)

    # Create reward shaper
    reward_shaper = RewardShaper(
        task_success_reward=100.0,
        direction_reward_scale=2.0,
        speed_reward_scale=0.5,
        gripper_reward_scale=1.0,
        forward_model_bonus=0.5,
        time_penalty=0.01,
    )

    # Optimizers
    actor_optimizer = torch.optim.Adam(action_generator.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # Statistics
    episode_rewards = deque(maxlen=100)
    success_rate = deque(maxlen=100)

    print("\n" + "="*60)
    print("Starting Training Loop")
    print("="*60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        planner.reset()

        episode_data = []
        current_plan = None
        steps_since_plan = 0
        total_reward = 0
        action_chunk = None
        action_idx = 0

        for step in range(env.max_episode_steps):
            # Replan if needed
            if current_plan is None or steps_since_plan >= replan_every:
                gripper_state = "open" if obs['gripper_state'] == 0 else "closed"

                if verbose:
                    print(f"\n[Episode {episode+1}, Step {step}] Generating plan...")

                current_plan = planner.plan(
                    image=obs['image'],
                    task_description=env.task_description,
                    gripper_state=gripper_state,
                    steps_since_plan=steps_since_plan,
                )

                if verbose:
                    print(f"  Phase: {current_plan.get('plan', {}).get('phase', 'N/A')}")
                    print(f"  Movements: {current_plan.get('plan', {}).get('movements', [])}")

                steps_since_plan = 0
                action_idx = 0

                # Generate action chunk
                proprio = torch.tensor(
                    obs['proprio'], dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action_chunk, components = action_generator(
                        [current_plan], proprio, return_components=True
                    )
                    action_chunk = action_chunk.squeeze(0).cpu().numpy()

            # Get action from chunk
            if action_chunk is not None and action_idx < len(action_chunk):
                action = action_chunk[action_idx]
                # Handle NaN values
                if np.isnan(action).any():
                    action = np.zeros(7)
            else:
                action = np.zeros(7)

            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)

            # Execute action
            next_obs, env_reward, done, truncated, info = env.step(action)

            # Compute shaped reward
            reward = reward_shaper.compute_reward(
                obs=obs,
                action=action,
                next_obs=next_obs,
                plan=current_plan,
                info=info,
            )

            # Store transition
            episode_data.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done or truncated,
                'plan': current_plan,
                'proprio': obs['proprio'],
            })

            total_reward += reward
            obs = next_obs
            steps_since_plan += 1
            action_idx += 1

            if done or truncated:
                break

        # Record statistics
        success = info.get('success', False)
        episode_rewards.append(total_reward)
        success_rate.append(float(success))

        print(f"\nEpisode {episode+1}/{num_episodes}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step + 1}")
        print(f"  Success: {success}")
        print(f"  Object grasped: {info.get('object_grasped', False)}")

        # Simple training update
        if len(episode_data) > 0:
            # Collect data
            all_plans = [d['plan'] for d in episode_data]
            all_proprios = np.array([d['proprio'] for d in episode_data])
            all_actions = np.array([d['action'] for d in episode_data])
            all_rewards = np.array([d['reward'] for d in episode_data])

            # Convert to tensors
            proprios = torch.tensor(all_proprios, dtype=torch.float32, device=device)
            actions = torch.tensor(all_actions, dtype=torch.float32, device=device)
            rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)

            # --- Policy Update ---
            pred_actions = action_generator(all_plans, proprios)[:, 0, :]

            # Compute advantages with fresh value computation (no gradient)
            with torch.no_grad():
                plan_embeds_for_value = action_generator.plan_encoder(all_plans).to(device)
                values_baseline = critic(torch.cat([plan_embeds_for_value, proprios], dim=-1)).squeeze(-1)
                advantages = rewards - values_baseline
                # Normalize advantages (handle single sample case)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

            # Policy loss
            action_diff = (pred_actions - actions).pow(2).mean(dim=-1)
            policy_loss = (action_diff * advantages).mean()

            # Update actor
            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(action_generator.parameters(), 1.0)
            actor_optimizer.step()

            # --- Value Update (separate forward pass) ---
            plan_embeds_for_critic = action_generator.plan_encoder(all_plans).detach().to(device)
            values = critic(torch.cat([plan_embeds_for_critic, proprios], dim=-1)).squeeze(-1)
            value_loss = F.mse_loss(values, rewards)

            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            if verbose:
                print(f"  Policy Loss: {policy_loss.item():.4f}")
                print(f"  Value Loss: {value_loss.item():.4f}")

    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(list(episode_rewards)):.2f}")
    print(f"Success Rate: {np.mean(list(success_rate)) * 100:.1f}%")

    env.close()

    return True


def main():
    try:
        test_training_loop(num_episodes=3, verbose=True)
        print("\n✓ Training loop test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
