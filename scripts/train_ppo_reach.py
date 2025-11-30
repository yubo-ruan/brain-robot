#!/usr/bin/env python3
"""
Simplified reaching task to prove PPO works.
Just reach the object - no grasping required.
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
import gymnasium as gym


class SimpleReachEnv(gym.Env):
    """
    Simplified reaching environment.
    Goal: Move end-effector to object position.
    Success: Distance < 0.05
    """
    def __init__(self, max_steps: int = 50):
        super().__init__()
        self.max_steps = max_steps

        # 3D position control only
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        # State: ee_pos (3) + object_pos (3)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Random starting position
        self.ee_pos = np.array([0.0, 0.0, 0.3])

        # Random object position
        self.object_pos = np.array([
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.1, 0.4),
            0.1,
        ])

        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.ee_pos, self.object_pos]).astype(np.float32)

    def step(self, action):
        # Scale action
        action = np.clip(action, -1, 1) * 0.05

        # Move end-effector
        self.ee_pos += action
        self.ee_pos = np.clip(self.ee_pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 0.5])

        self.step_count += 1

        # Compute distance
        dist = np.linalg.norm(self.ee_pos - self.object_pos)

        # Dense reward: negative distance
        reward = -dist

        # Success bonus
        success = dist < 0.05
        if success:
            reward += 10.0

        done = success
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, done, truncated, {'success': success, 'distance': dist}


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        features = self.net(obs)
        mean = torch.tanh(self.mean_head(features))  # Bound mean
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
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


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


def train():
    print("="*60)
    print("PPO Reaching Task")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = SimpleReachEnv(max_steps=50)

    policy = GaussianPolicy(6, 3, 128).to(device)
    value_net = ValueNetwork(6, 128).to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Training params
    num_episodes = 1000
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
        # Collect rollout
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []

        steps = 0
        while steps < steps_per_update:
            obs, _ = env.reset()
            ep_reward = 0

            while True:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, _, _ = policy.get_action(obs_t)
                    value = value_net(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, done, truncated, info = env.step(action_np)

                obs_buf.append(obs)
                act_buf.append(action_np)
                rew_buf.append(reward)
                done_buf.append(done or truncated)
                logp_buf.append(log_prob.item())
                val_buf.append(value.item())

                ep_reward += reward
                steps += 1
                total_steps += 1
                obs = next_obs

                if done or truncated:
                    break

            episode_rewards.append(ep_reward)
            success_rates.append(float(info.get('success', False)))
            episode_count += 1

            if episode_count % 50 == 0:
                avg_rew = np.mean(list(episode_rewards))
                avg_succ = np.mean(list(success_rates)) * 100
                print(f"Ep {episode_count}: Reward={avg_rew:.1f}, Success={avg_succ:.1f}%, "
                      f"Steps={total_steps}, Time={time.time()-start:.0f}s")

            if episode_count >= num_episodes:
                break

        # Convert to tensors
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_logp = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        values = torch.tensor(val_buf, dtype=torch.float32, device=device)

        # GAE
        advantages = compute_gae(rew_buf, val_buf, done_buf, gamma, gae_lam).to(device)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        batch_size = len(obs_buf)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_idx = indices[start:end]

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

    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Final success rate: {np.mean(list(success_rates))*100:.1f}%")
    print(f"Final avg reward: {np.mean(list(episode_rewards)):.1f}")

    return np.mean(list(success_rates))


if __name__ == "__main__":
    train()
