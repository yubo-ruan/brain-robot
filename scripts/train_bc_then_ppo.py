#!/usr/bin/env python3
"""
Behavioral Cloning + PPO for pick-and-place.

1. Collect expert demos from scripted policy
2. Pre-train with BC
3. Fine-tune with PPO
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

from src.env.mock_env import make_mock_env


class PolicyNetwork(nn.Module):
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
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        features = self.net(obs)
        mean = torch.tanh(self.mean_head(features))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean, None, mean, std
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return torch.clamp(action, -1, 1), log_prob, mean, std

    def evaluate_actions(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


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


def get_obs(env):
    """Build observation."""
    return np.concatenate([
        env.robot_pos,
        env.object_pos,
        env.target_pos,
        [1.0 if env.gripper_open else 0.0],
        [1.0 if env.object_grasped else 0.0],
    ]).astype(np.float32)


def collect_expert_demos(num_demos=1000):
    """Collect demonstrations from expert."""
    print("Collecting expert demonstrations...")
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    obs_data = []
    action_data = []
    successes = 0

    for ep in range(num_demos):
        obs, info = env.reset()
        for step in range(100):
            full_obs = get_obs(env)
            action = get_expert_action(env)

            obs_data.append(full_obs)
            action_data.append(action)

            next_obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    print(f"Expert success rate: {successes/num_demos*100:.1f}%")
    print(f"Collected {len(obs_data)} transitions")

    return np.array(obs_data), np.array(action_data)


def behavioral_cloning(policy, obs_data, action_data, device, epochs=100):
    """Pre-train with BC."""
    print("\nBehavioral Cloning...")

    obs_t = torch.tensor(obs_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch_size = 256

    for epoch in range(epochs):
        indices = torch.randperm(len(obs_data))
        total_loss = 0
        num_batches = 0

        for start in range(0, len(obs_data), batch_size):
            end = min(start + batch_size, len(obs_data))
            mb_idx = indices[start:end]

            mb_obs = obs_t[mb_idx]
            mb_actions = action_t[mb_idx]

            # MSE loss on mean
            mean, std = policy.forward(mb_obs)
            loss = F.mse_loss(mean, mb_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/num_batches:.4f}")

    return policy


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        if dones[t]:
            next_value = 0
            gae = 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)


def ppo_finetune(policy, value_net, device, num_episodes=500):
    """Fine-tune with PPO."""
    print("\nPPO Fine-tuning...")

    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-5)  # Lower LR for fine-tuning
    value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    steps_per_update = 2048
    num_epochs = 10
    mini_batch_size = 64
    clip_eps = 0.2
    gamma = 0.99
    gae_lam = 0.95

    episode_rewards = deque(maxlen=100)
    success_rates = deque(maxlen=100)

    total_steps = 0
    episode_count = 0
    start = time.time()

    while episode_count < num_episodes:
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
        steps = 0

        while steps < steps_per_update:
            obs, info = env.reset()
            ep_reward = 0

            while True:
                full_obs = get_obs(env)
                obs_t = torch.tensor(full_obs, device=device).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, _, _ = policy.get_action(obs_t)
                    value = value_net(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, done, truncated, info = env.step(action_np)

                obs_buf.append(full_obs)
                act_buf.append(action_np)
                rew_buf.append(reward)
                done_buf.append(done or truncated)
                logp_buf.append(log_prob.item())
                val_buf.append(value.item())

                ep_reward += reward
                steps += 1
                total_steps += 1

                if done or truncated:
                    break

            episode_rewards.append(ep_reward)
            success_rates.append(float(info.get('success', False)))
            episode_count += 1

            if episode_count % 50 == 0:
                avg_rew = np.mean(list(episode_rewards))
                avg_succ = np.mean(list(success_rates)) * 100
                print(f"Ep {episode_count}: Reward={avg_rew:.1f}, Success={avg_succ:.1f}%, "
                      f"Time={time.time()-start:.0f}s")

            if episode_count >= num_episodes:
                break

        # PPO update
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_logp = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        values = torch.tensor(val_buf, dtype=torch.float32, device=device)

        advantages = compute_gae(rew_buf, val_buf, done_buf, gamma, gae_lam).to(device)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(obs_buf)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size)
            for start_idx in range(0, batch_size, mini_batch_size):
                end = min(start_idx + mini_batch_size, batch_size)
                mb_idx = indices[start_idx:end]

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                new_logp, entropy = policy.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_opt.step()

                value_loss = F.mse_loss(value_net(mb_obs), mb_ret)
                value_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                value_opt.step()

    env.close()
    return np.mean(list(success_rates))


def evaluate_policy(policy, device, num_episodes=100):
    """Evaluate policy."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        for step in range(100):
            full_obs = get_obs(env)
            obs_t = torch.tensor(full_obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.get_action(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(action_np)
            if done or truncated:
                break
        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100


def train():
    print("=" * 60)
    print("BC + PPO for Pick-and-Place")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs_dim = 11
    action_dim = 7

    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=256).to(device)
    value_net = ValueNetwork(obs_dim, hidden_dim=256).to(device)

    # Step 1: Collect expert demos
    obs_data, action_data = collect_expert_demos(num_demos=500)

    # Step 2: Behavioral cloning
    policy = behavioral_cloning(policy, obs_data, action_data, device, epochs=100)

    # Evaluate after BC
    bc_success = evaluate_policy(policy, device)
    print(f"\nAfter BC: Success={bc_success:.1f}%")

    # Step 3: PPO fine-tuning
    final_success = ppo_finetune(policy, value_net, device, num_episodes=500)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"After BC: {bc_success:.1f}%")
    print(f"After PPO: {final_success*100:.1f}%")

    # Final evaluation
    final_eval = evaluate_policy(policy, device)
    print(f"Final evaluation: {final_eval:.1f}%")


if __name__ == "__main__":
    train()
