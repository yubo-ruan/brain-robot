#!/usr/bin/env python3
"""
Proper PPO implementation following the critique:
- Gaussian policy with log-probs
- Likelihood ratios and clipping
- GAE for advantage estimation
- Proper handling of episode boundaries

Phase A: Fix RL foundations
Phase B: Test without VLM (oracle plans)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import time

from brain_robot.env.mock_env import make_mock_env


class GaussianPolicy(nn.Module):
    """
    Simple Gaussian policy: π(a|s) = N(μ(s), σ)
    No VLM, no primitives - just prove RL works.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Initialize mean head to output small actions
        nn.init.uniform_(self.mean_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor):
        """Return mean and std for action distribution."""
        features = self.net(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and compute log-prob."""
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Clip to action space
        action = torch.tanh(action)

        return action, log_prob, mean, std

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Compute log-prob of given actions under current policy."""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        # Inverse tanh to get pre-squash actions
        # actions are in [-1, 1], need to map back
        eps = 1e-6
        actions_clamped = actions.clamp(-1 + eps, 1 - eps)
        pre_tanh = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)

        # Correction for tanh squashing
        log_prob -= torch.log(1 - actions_clamped.pow(2) + eps).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Simple value function V(s)."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.
    Properly handles episode boundaries.
    """
    advantages = []
    gae = 0

    # Iterate backwards through trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Bootstrap from 0 at end
        else:
            next_value = values[t + 1]

        # If episode ended, don't bootstrap
        if dones[t]:
            next_value = 0
            gae = 0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


def ppo_update(
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    action_batch: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_clip: float = 0.2,
    entropy_coef: float = 0.01,
    num_epochs: int = 4,
    mini_batch_size: int = 64,
):
    """
    Proper PPO update with clipped objective.
    """
    device = obs_batch.device
    batch_size = obs_batch.shape[0]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_losses = []
    value_losses = []
    entropy_losses = []

    for _ in range(num_epochs):
        # Shuffle indices
        indices = torch.randperm(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb_indices = indices[start:end]

            mb_obs = obs_batch[mb_indices]
            mb_actions = action_batch[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # Get current policy log-probs
            new_log_probs, entropy = policy.evaluate_actions(mb_obs, mb_actions)

            # Compute ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total policy loss
            total_policy_loss = policy_loss + entropy_coef * entropy_loss

            # Update policy
            policy_optimizer.zero_grad()
            total_policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optimizer.step()

            # Value loss
            values = value_net(mb_obs)
            value_loss = F.mse_loss(values, mb_returns)

            # Update value network
            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(-entropy_loss.item())

    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropy_losses),
    }


def train_ppo(
    num_episodes: int = 500,
    steps_per_update: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    lr: float = 3e-4,
    num_epochs: int = 10,
    mini_batch_size: int = 64,
):
    """
    Train with proper PPO.
    Phase B: No VLM, just proprio → action.
    """
    print("="*60)
    print("Proper PPO Training (No VLM)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    print(f"Task: {env.task_description}")

    # Need to include object and target positions for the policy to know where to go
    # proprio (15) + object_pos (3) + target_pos (3) = 21
    obs_dim = 21
    action_dim = 7

    # Create networks
    policy = GaussianPolicy(obs_dim, action_dim, hidden_dim=256).to(device)
    value_net = ValueNetwork(obs_dim, hidden_dim=256).to(device)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    # Statistics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    success_rates = deque(maxlen=100)

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    total_steps = 0
    episode_count = 0
    start_time = time.time()

    while episode_count < num_episodes:
        # Collect rollout
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        log_prob_buffer = []
        value_buffer = []

        steps_collected = 0

        while steps_collected < steps_per_update:
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # Get full observation: proprio + object_pos + target_pos
                full_obs = np.concatenate([
                    obs['proprio'],
                    env.object_pos,
                    env.target_pos,
                ])
                obs_tensor = torch.tensor(full_obs, dtype=torch.float32, device=device).unsqueeze(0)

                # Get action from policy
                with torch.no_grad():
                    action, log_prob, _, _ = policy.get_action(obs_tensor)
                    value = value_net(obs_tensor)

                action_np = action.squeeze(0).cpu().numpy()

                # Execute action
                next_obs, reward, done, truncated, info = env.step(action_np)

                # Store transition
                obs_buffer.append(full_obs)
                action_buffer.append(action_np)
                reward_buffer.append(reward)
                done_buffer.append(done or truncated)
                log_prob_buffer.append(log_prob.item())
                value_buffer.append(value.item())

                episode_reward += reward
                episode_length += 1
                steps_collected += 1
                total_steps += 1

                obs = next_obs

                if done or truncated:
                    break

            # Record episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_rates.append(float(info.get('success', False)))
            episode_count += 1

            if episode_count % 10 == 0:
                avg_reward = np.mean(list(episode_rewards))
                avg_length = np.mean(list(episode_lengths))
                success_rate = np.mean(list(success_rates)) * 100
                elapsed = time.time() - start_time

                print(f"Episode {episode_count}: "
                      f"Reward={avg_reward:.1f}, "
                      f"Length={avg_length:.0f}, "
                      f"Success={success_rate:.1f}%, "
                      f"Steps={total_steps}, "
                      f"Time={elapsed:.0f}s")

            if episode_count >= num_episodes:
                break

        # Convert to tensors
        obs_batch = torch.tensor(np.array(obs_buffer), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.array(action_buffer), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(np.array(log_prob_buffer), dtype=torch.float32, device=device)
        values = torch.tensor(np.array(value_buffer), dtype=torch.float32, device=device)
        rewards = np.array(reward_buffer)
        dones = np.array(done_buffer)

        # Compute advantages with GAE
        advantages = compute_gae(rewards, values.cpu().numpy(), dones, gamma, gae_lambda)
        advantages = advantages.to(device)

        # Compute returns
        returns = advantages + values

        # PPO update
        update_info = ppo_update(
            policy, value_net,
            policy_optimizer, value_optimizer,
            obs_batch, action_batch, old_log_probs,
            advantages, returns,
            clip_epsilon=clip_epsilon,
            num_epochs=num_epochs,
            mini_batch_size=mini_batch_size,
        )

        if episode_count % 50 == 0:
            print(f"  PPO Update: policy_loss={update_info['policy_loss']:.4f}, "
                  f"value_loss={update_info['value_loss']:.4f}, "
                  f"entropy={update_info['entropy']:.4f}")

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {total_steps}")
    print(f"Final avg reward: {np.mean(list(episode_rewards)):.1f}")
    print(f"Final success rate: {np.mean(list(success_rates)) * 100:.1f}%")

    # Save checkpoint
    checkpoint_path = "/workspace/brain_robot/checkpoints/ppo_proper.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'policy': policy.state_dict(),
        'value_net': value_net.state_dict(),
        'episode': episode_count,
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    env.close()
    return {
        'avg_reward': np.mean(list(episode_rewards)),
        'success_rate': np.mean(list(success_rates)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps_per_update", type=int, default=2048)
    args = parser.parse_args()

    results = train_ppo(
        num_episodes=args.episodes,
        steps_per_update=args.steps_per_update,
    )
    print(f"\nFinal results: {results}")
