#!/usr/bin/env python3
"""
Phase D: Reintroduce plan-based shaping carefully.

Test whether explicit phase conditioning helps:
1. Phase-conditioned action heads (separate networks per phase)
2. Phase prediction as auxiliary task
3. Plan embedding as intermediate representation
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


# ============================================================
# Model 1: Baseline (no phase conditioning) - from Phase C
# ============================================================

class BaselinePolicy(nn.Module):
    """Simple MLP policy (no phase conditioning)."""
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, proprio):
        x = torch.cat([state, proprio], dim=-1)
        return self.net(x)


# ============================================================
# Model 2: Phase-Conditioned Policy
# ============================================================

class PhaseConditionedPolicy(nn.Module):
    """
    Policy with explicit phase conditioning.
    Phase is provided as input (oracle) during training.
    """
    def __init__(self, state_dim=9, proprio_dim=2, phase_dim=4, action_dim=7, hidden_dim=128):
        super().__init__()
        # Phase embedding
        self.phase_embed = nn.Embedding(4, phase_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + proprio_dim + phase_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, proprio, phase_idx):
        phase_emb = self.phase_embed(phase_idx)
        x = torch.cat([state, proprio, phase_emb], dim=-1)
        return self.net(x)


# ============================================================
# Model 3: Multi-Head Policy (separate head per phase)
# ============================================================

class MultiHeadPolicy(nn.Module):
    """
    Separate action heads for each phase.
    Shared encoder, phase-specific decoders.
    """
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128, num_phases=4):
        super().__init__()
        self.num_phases = num_phases

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Phase-specific action heads
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh(),
            )
            for _ in range(num_phases)
        ])

    def forward(self, state, proprio, phase_idx):
        x = torch.cat([state, proprio], dim=-1)
        features = self.encoder(x)

        # Get action from appropriate head
        actions = torch.zeros(state.shape[0], 7, device=state.device)
        for i in range(self.num_phases):
            mask = (phase_idx == i)
            if mask.any():
                actions[mask] = self.action_heads[i](features[mask])

        return actions


# ============================================================
# Model 4: Phase Predictor + Action Policy
# ============================================================

class PhasePredictorPolicy(nn.Module):
    """
    Learns to predict phase, then uses it for action.
    End-to-end differentiable via soft phase selection.
    """
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128, num_phases=4):
        super().__init__()
        self.num_phases = num_phases

        # Phase predictor
        self.phase_predictor = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases),
        )

        # Phase-conditioned action network
        self.action_net = nn.Sequential(
            nn.Linear(state_dim + proprio_dim + num_phases, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, proprio, return_phase=False):
        x = torch.cat([state, proprio], dim=-1)

        # Predict phase (soft)
        phase_logits = self.phase_predictor(x)
        phase_probs = F.softmax(phase_logits, dim=-1)

        # Concatenate with phase probs
        x_with_phase = torch.cat([state, proprio, phase_probs], dim=-1)
        action = self.action_net(x_with_phase)

        if return_phase:
            return action, phase_logits
        return action


# ============================================================
# Training and Evaluation
# ============================================================

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
            phase = 0  # approach
        else:
            action = np.zeros(7)
            action[6] = 1.0
            phase = 1  # grasp
    else:
        if dist_to_target > 0.08:
            direction = target_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = 1.0
            phase = 2  # transport
        else:
            action = np.zeros(7)
            action[6] = -1.0
            phase = 3  # release

    return action, phase


def collect_demos(num_demos=500):
    """Collect expert demonstrations with phase labels."""
    print("Collecting demonstrations...")
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    state_data, proprio_data, action_data, phase_data = [], [], [], []
    successes = 0

    for ep in range(num_demos):
        obs, info = env.reset()
        for step in range(100):
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)
            action, phase = get_expert_action(env)

            state_data.append(state)
            proprio_data.append(proprio)
            action_data.append(action)
            phase_data.append(phase)

            next_obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    print(f"Expert success: {successes/num_demos*100:.1f}%, {len(state_data)} transitions")

    return (np.array(state_data), np.array(proprio_data),
            np.array(action_data), np.array(phase_data))


def train_baseline(state_data, proprio_data, action_data, device, epochs=200):
    """Train baseline policy (no phase conditioning)."""
    policy = BaselinePolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            pred = policy(state_t[idx], proprio_t[idx])
            loss = F.mse_loss(pred, action_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def train_phase_conditioned(state_data, proprio_data, action_data, phase_data, device, epochs=200):
    """Train phase-conditioned policy."""
    policy = PhaseConditionedPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    phase_t = torch.tensor(phase_data, dtype=torch.long, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            pred = policy(state_t[idx], proprio_t[idx], phase_t[idx])
            loss = F.mse_loss(pred, action_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def train_multihead(state_data, proprio_data, action_data, phase_data, device, epochs=200):
    """Train multi-head policy."""
    policy = MultiHeadPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    phase_t = torch.tensor(phase_data, dtype=torch.long, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            pred = policy(state_t[idx], proprio_t[idx], phase_t[idx])
            loss = F.mse_loss(pred, action_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def train_phase_predictor(state_data, proprio_data, action_data, phase_data, device, epochs=200):
    """Train phase predictor policy."""
    policy = PhasePredictorPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    phase_t = torch.tensor(phase_data, dtype=torch.long, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            pred_action, phase_logits = policy(state_t[idx], proprio_t[idx], return_phase=True)

            action_loss = F.mse_loss(pred_action, action_t[idx])
            phase_loss = F.cross_entropy(phase_logits, phase_t[idx])
            loss = action_loss + 0.1 * phase_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def evaluate_baseline(policy, device, num_episodes=100):
    """Evaluate baseline policy."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        for step in range(100):
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)

            state_t = torch.tensor(state, device=device).unsqueeze(0)
            proprio_t = torch.tensor(proprio, device=device).unsqueeze(0)

            with torch.no_grad():
                action = policy(state_t, proprio_t)

            next_obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100


def evaluate_phase_conditioned(policy, device, num_episodes=100, oracle_phase=True):
    """Evaluate phase-conditioned policy."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    successes = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        for step in range(100):
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)

            # Get oracle phase
            _, phase = get_expert_action(env)

            state_t = torch.tensor(state, device=device).unsqueeze(0)
            proprio_t = torch.tensor(proprio, device=device).unsqueeze(0)
            phase_t = torch.tensor([phase], dtype=torch.long, device=device)

            with torch.no_grad():
                action = policy(state_t, proprio_t, phase_t)

            next_obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100


def evaluate_phase_predictor(policy, device, num_episodes=100):
    """Evaluate phase predictor policy (no oracle phase)."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    successes = 0
    phase_accuracies = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        for step in range(100):
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)

            state_t = torch.tensor(state, device=device).unsqueeze(0)
            proprio_t = torch.tensor(proprio, device=device).unsqueeze(0)

            with torch.no_grad():
                action, phase_logits = policy(state_t, proprio_t, return_phase=True)
                pred_phase = phase_logits.argmax(dim=-1).item()

            # Check phase accuracy
            _, true_phase = get_expert_action(env)
            phase_accuracies.append(pred_phase == true_phase)

            next_obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100, np.mean(phase_accuracies) * 100


def main():
    print("=" * 60)
    print("Phase D: Plan-Based Shaping Comparison")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect demos
    state_data, proprio_data, action_data, phase_data = collect_demos(500)

    results = {}

    # 1. Baseline (no phase)
    print("\n--- Training Baseline (no phase) ---")
    baseline = train_baseline(state_data, proprio_data, action_data, device)
    results['baseline'] = evaluate_baseline(baseline, device)
    print(f"Baseline: {results['baseline']:.1f}%")

    # 2. Phase-conditioned (oracle phase at test time)
    print("\n--- Training Phase-Conditioned (oracle phase) ---")
    phase_cond = train_phase_conditioned(state_data, proprio_data, action_data, phase_data, device)
    results['phase_conditioned_oracle'] = evaluate_phase_conditioned(phase_cond, device, oracle_phase=True)
    print(f"Phase-Conditioned (oracle): {results['phase_conditioned_oracle']:.1f}%")

    # 3. Multi-head (oracle phase at test time)
    print("\n--- Training Multi-Head (oracle phase) ---")
    multihead = train_multihead(state_data, proprio_data, action_data, phase_data, device)
    results['multihead_oracle'] = evaluate_phase_conditioned(multihead, device, oracle_phase=True)
    print(f"Multi-Head (oracle): {results['multihead_oracle']:.1f}%")

    # 4. Phase predictor (no oracle at test time)
    print("\n--- Training Phase Predictor (learned phase) ---")
    phase_pred = train_phase_predictor(state_data, proprio_data, action_data, phase_data, device)
    results['phase_predictor'], phase_acc = evaluate_phase_predictor(phase_pred, device)
    print(f"Phase Predictor: {results['phase_predictor']:.1f}% (phase accuracy: {phase_acc:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Phase-Based Shaping Results")
    print("=" * 60)
    print(f"  Baseline (no phase):              {results['baseline']:.1f}%")
    print(f"  Phase-Conditioned (oracle):       {results['phase_conditioned_oracle']:.1f}%")
    print(f"  Multi-Head (oracle):              {results['multihead_oracle']:.1f}%")
    print(f"  Phase Predictor (learned):        {results['phase_predictor']:.1f}%")

    return results


if __name__ == "__main__":
    main()
