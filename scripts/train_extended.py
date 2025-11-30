#!/usr/bin/env python3
"""
Extended training script with detailed logging.
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

from brain_robot.vlm.qwen_planner import QwenVLPlanner
from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator
from brain_robot.env.mock_env import make_mock_env
from brain_robot.training.rewards import RewardShaper


def train(num_episodes=10, verbose=True):
    """Run extended training."""
    print("="*60)
    print("Extended Training Session")
    print("="*60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create environment
    print("\nCreating environment...")
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
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

    # Load pre-trained selector if available
    pretrained_path = "/workspace/brain_robot/checkpoints/pretrained_selector.pt"
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained selector from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        action_generator.load_state_dict(checkpoint['action_generator'])
        print("Pre-trained selector loaded successfully!")

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
    fm_optimizer = torch.optim.Adam(action_generator.cerebellum.parameters(), lr=1e-3)

    # Statistics
    episode_rewards = deque(maxlen=100)
    success_rate = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    policy_losses = deque(maxlen=100)
    value_losses = deque(maxlen=100)

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    total_steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        episode_start = time.time()
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

                current_plan = planner.plan(
                    image=obs['image'],
                    task_description=env.task_description,
                    gripper_state=gripper_state,
                    steps_since_plan=steps_since_plan,
                )

                if verbose and step == 0:
                    print(f"\n[Ep {episode+1}] Initial plan: {current_plan.get('plan', {}).get('phase')} - "
                          f"{current_plan.get('plan', {}).get('movements', [])}")

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
                if np.isnan(action).any():
                    action = np.zeros(7)
            else:
                action = np.zeros(7)

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
            total_steps += 1

            if done or truncated:
                break

        # Record statistics
        success = info.get('success', False)
        episode_rewards.append(total_reward)
        success_rate.append(float(success))
        episode_lengths.append(len(episode_data))

        # Training update
        if len(episode_data) > 1:
            all_plans = [d['plan'] for d in episode_data]
            all_proprios = np.array([d['proprio'] for d in episode_data])
            all_actions = np.array([d['action'] for d in episode_data])
            all_rewards = np.array([d['reward'] for d in episode_data])

            proprios = torch.tensor(all_proprios, dtype=torch.float32, device=device)
            actions = torch.tensor(all_actions, dtype=torch.float32, device=device)
            rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)

            # Policy update
            pred_actions = action_generator(all_plans, proprios)[:, 0, :]

            with torch.no_grad():
                plan_embeds = action_generator.plan_encoder(all_plans).to(device)
                values_baseline = critic(torch.cat([plan_embeds, proprios], dim=-1)).squeeze(-1)
                advantages = rewards - values_baseline
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            action_diff = (pred_actions - actions).pow(2).mean(dim=-1)
            policy_loss = (action_diff * advantages).mean()

            actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(action_generator.parameters(), 1.0)
            actor_optimizer.step()

            # Value update
            plan_embeds = action_generator.plan_encoder(all_plans).detach().to(device)
            values = critic(torch.cat([plan_embeds, proprios], dim=-1)).squeeze(-1)
            value_loss = F.mse_loss(values, rewards)

            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            # Forward model update
            if len(episode_data) > 2:
                proprio_seq = proprios[:-1]
                action_seq = actions[:-1]
                next_proprio = proprios[1:]

                pred_next = action_generator.cerebellum(proprio_seq, action_seq)
                fm_loss = F.mse_loss(pred_next, next_proprio)

                fm_optimizer.zero_grad()
                fm_loss.backward()
                fm_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        episode_time = time.time() - episode_start

        # Log progress
        print(f"\nEpisode {episode+1}/{num_episodes}:")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Steps: {step + 1}")
        print(f"  Success: {success}")
        print(f"  Object grasped: {info.get('object_grasped', False)}")
        print(f"  Dist to object: {info.get('dist_to_object', -1):.3f}")
        print(f"  Time: {episode_time:.1f}s")

        if len(policy_losses) > 0:
            print(f"  Policy Loss: {policy_losses[-1]:.6f}")
            print(f"  Value Loss: {value_losses[-1]:.4f}")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Steps/second: {total_steps/total_time:.1f}")
    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Avg Reward: {np.mean(list(episode_rewards)):.2f} ± {np.std(list(episode_rewards)):.2f}")
    print(f"  Success Rate: {np.mean(list(success_rate)) * 100:.1f}%")
    print(f"  Avg Episode Length: {np.mean(list(episode_lengths)):.1f}")
    print(f"  Avg Policy Loss: {np.mean(list(policy_losses)):.6f}")
    print(f"  Avg Value Loss: {np.mean(list(value_losses)):.4f}")

    # Save checkpoint
    checkpoint_path = "/workspace/brain_robot/checkpoints/checkpoint_latest.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'episode': num_episodes,
        'action_generator': action_generator.state_dict(),
        'critic': critic.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'critic_optimizer': critic_optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    env.close()
    return {
        'avg_reward': np.mean(list(episode_rewards)),
        'success_rate': np.mean(list(success_rate)),
        'policy_loss': np.mean(list(policy_losses)) if policy_losses else 0,
        'value_loss': np.mean(list(value_losses)) if value_losses else 0,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    results = train(num_episodes=args.episodes, verbose=args.verbose)
    print(f"\nFinal results: {results}")
