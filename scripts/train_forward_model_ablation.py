#!/usr/bin/env python3
"""
Phase E: Forward Model Ablation Study.

Test whether a forward model helps or hurts:
1. No forward model (baseline)
2. Forward model for state prediction (auxiliary loss)
3. Forward model for action correction
4. Model-based planning with forward model
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
# Model 1: Baseline (no forward model)
# ============================================================

class BaselinePolicy(nn.Module):
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
# Model 2: Policy + Forward Model (auxiliary loss)
# ============================================================

class PolicyWithForwardModel(nn.Module):
    """
    Policy with forward model as auxiliary task.
    Forward model predicts next state given current state + action.
    """
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128):
        super().__init__()

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Forward model: (state, action) -> next_state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),  # Predict next state
        )

    def forward(self, state, proprio):
        return self.policy(torch.cat([state, proprio], dim=-1))

    def predict_next_state(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.forward_model(x)


# ============================================================
# Model 3: Policy with Action Correction
# ============================================================

class PolicyWithCorrection(nn.Module):
    """
    Policy that uses forward model to correct actions.
    Similar to original brain_model's cerebellum.
    """
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128):
        super().__init__()

        # Base policy
        self.base_policy = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Correction network: takes prediction error, outputs action correction
        self.corrector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        self.correction_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, proprio, target_state=None):
        x = torch.cat([state, proprio], dim=-1)
        base_action = self.base_policy(x)

        if target_state is not None:
            # Predict next state
            pred_next = self.forward_model(torch.cat([state, base_action], dim=-1))
            # Compute error
            error = target_state - pred_next
            # Compute correction
            correction = self.corrector(error) * torch.sigmoid(self.correction_scale)
            return base_action + correction

        return base_action


# ============================================================
# Model 4: Model-Based Planning (MPC-style)
# ============================================================

class ModelBasedPlanner(nn.Module):
    """
    Uses forward model for short-horizon planning.
    At test time, evaluates multiple action sequences.
    """
    def __init__(self, state_dim=9, proprio_dim=2, action_dim=7, hidden_dim=128):
        super().__init__()

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Action proposal network (for warm starting)
        self.action_proposal = nn.Sequential(
            nn.Linear(state_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, proprio, target_pos=None, num_samples=10, horizon=3):
        """
        Model-based planning with shooting.
        """
        batch_size = state.shape[0]

        # Get base action proposal
        base_action = self.action_proposal(torch.cat([state, proprio], dim=-1))

        if target_pos is None:
            return base_action

        # Sample action perturbations
        best_actions = base_action.clone()
        best_costs = torch.full((batch_size,), float('inf'), device=state.device)

        for _ in range(num_samples):
            # Perturb base action
            noise = torch.randn_like(base_action) * 0.2
            candidate_action = torch.clamp(base_action + noise, -1, 1)

            # Simulate forward
            sim_state = state.clone()
            total_cost = torch.zeros(batch_size, device=state.device)

            for h in range(horizon):
                # Predict next state
                sim_state = self.forward_model(torch.cat([sim_state, candidate_action], dim=-1))
                # Compute cost (distance to target)
                cost = ((sim_state[:, :3] - target_pos) ** 2).sum(dim=-1)
                total_cost += cost

            # Update best
            better = total_cost < best_costs
            best_costs[better] = total_cost[better]
            best_actions[better] = candidate_action[better]

        return best_actions


# ============================================================
# Training
# ============================================================

def get_expert_action(env):
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
        else:
            action = np.zeros(7)
            action[6] = 1.0
    else:
        if dist_to_target > 0.08:
            direction = target_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = 1.0
        else:
            action = np.zeros(7)
            action[6] = -1.0

    return action


def collect_demos_with_transitions(num_demos=500):
    """Collect demos with (s, a, s') transitions."""
    print("Collecting demonstrations with transitions...")
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)

    state_data, proprio_data, action_data, next_state_data, target_data = [], [], [], [], []
    successes = 0

    for ep in range(num_demos):
        obs, info = env.reset()
        for step in range(100):
            state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)
            proprio = np.array([1.0 if env.gripper_open else 0.0, 1.0 if env.object_grasped else 0.0], dtype=np.float32)
            target = env.object_pos.copy() if not env.object_grasped else env.target_pos.copy()

            action = get_expert_action(env)
            next_obs, reward, done, truncated, info = env.step(action)

            next_state = np.concatenate([env.robot_pos, env.object_pos, env.target_pos]).astype(np.float32)

            state_data.append(state)
            proprio_data.append(proprio)
            action_data.append(action)
            next_state_data.append(next_state)
            target_data.append(target.astype(np.float32))

            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    print(f"Expert success: {successes/num_demos*100:.1f}%, {len(state_data)} transitions")

    return (np.array(state_data), np.array(proprio_data), np.array(action_data),
            np.array(next_state_data), np.array(target_data))


def train_baseline(data, device, epochs=200):
    state_data, proprio_data, action_data, _, _ = data
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


def train_with_forward_model(data, device, epochs=200, fm_weight=0.1):
    """Train policy with forward model as auxiliary task."""
    state_data, proprio_data, action_data, next_state_data, _ = data
    policy = PolicyWithForwardModel().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    next_state_t = torch.tensor(next_state_data, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            # Policy loss
            pred_action = policy(state_t[idx], proprio_t[idx])
            policy_loss = F.mse_loss(pred_action, action_t[idx])

            # Forward model loss
            pred_next = policy.predict_next_state(state_t[idx], action_t[idx])
            fm_loss = F.mse_loss(pred_next, next_state_t[idx])

            loss = policy_loss + fm_weight * fm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def train_with_correction(data, device, epochs=200):
    """Train policy with action correction."""
    state_data, proprio_data, action_data, next_state_data, target_data = data
    policy = PolicyWithCorrection().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    next_state_t = torch.tensor(next_state_data, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            # Train with correction toward actual next state
            pred_action = policy(state_t[idx], proprio_t[idx], target_state=next_state_t[idx])
            loss = F.mse_loss(pred_action, action_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy


def train_model_based(data, device, epochs=200):
    """Train model-based planner."""
    state_data, proprio_data, action_data, next_state_data, _ = data
    planner = ModelBasedPlanner().to(device)
    optimizer = torch.optim.Adam(planner.parameters(), lr=1e-3)

    state_t = torch.tensor(state_data, dtype=torch.float32, device=device)
    proprio_t = torch.tensor(proprio_data, dtype=torch.float32, device=device)
    action_t = torch.tensor(action_data, dtype=torch.float32, device=device)
    next_state_t = torch.tensor(next_state_data, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        indices = torch.randperm(len(state_data))
        for start in range(0, len(state_data), 256):
            end = min(start + 256, len(state_data))
            idx = indices[start:end]

            # Action proposal loss
            pred_action = planner.action_proposal(torch.cat([state_t[idx], proprio_t[idx]], dim=-1))
            action_loss = F.mse_loss(pred_action, action_t[idx])

            # Forward model loss
            pred_next = planner.forward_model(torch.cat([state_t[idx], action_t[idx]], dim=-1))
            fm_loss = F.mse_loss(pred_next, next_state_t[idx])

            loss = action_loss + 0.1 * fm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return planner


def evaluate(policy, device, num_episodes=100, use_planning=False):
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
                if use_planning:
                    target = env.object_pos if not env.object_grasped else env.target_pos
                    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy(state_t, proprio_t, target_pos=target_t)
                else:
                    action = policy(state_t, proprio_t)

            next_obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            if done or truncated:
                break

        if info.get('success', False):
            successes += 1

    env.close()
    return successes / num_episodes * 100


def benchmark_speed(policy, device, num_iters=1000, use_planning=False):
    state = torch.randn(1, 9, device=device)
    proprio = torch.randn(1, 2, device=device)
    target = torch.randn(1, 3, device=device)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            if use_planning:
                policy(state, proprio, target_pos=target)
            else:
                policy(state, proprio)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(num_iters):
        with torch.no_grad():
            if use_planning:
                policy(state, proprio, target_pos=target)
            else:
                policy(state, proprio)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    return num_iters / elapsed


def main():
    print("=" * 60)
    print("Phase E: Forward Model Ablation Study")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect demos
    data = collect_demos_with_transitions(500)

    results = {}

    # 1. Baseline
    print("\n--- Baseline (no forward model) ---")
    baseline = train_baseline(data, device)
    results['baseline'] = evaluate(baseline, device)
    results['baseline_fps'] = benchmark_speed(baseline, device)
    print(f"Baseline: {results['baseline']:.1f}%, {results['baseline_fps']:.0f} FPS")

    # 2. Forward model as auxiliary task
    print("\n--- Forward Model (auxiliary task) ---")
    fm_aux = train_with_forward_model(data, device)
    results['fm_auxiliary'] = evaluate(fm_aux, device)
    results['fm_auxiliary_fps'] = benchmark_speed(fm_aux, device)
    print(f"FM Auxiliary: {results['fm_auxiliary']:.1f}%, {results['fm_auxiliary_fps']:.0f} FPS")

    # 3. Forward model with correction
    print("\n--- Forward Model (action correction) ---")
    fm_corr = train_with_correction(data, device)
    results['fm_correction'] = evaluate(fm_corr, device)
    results['fm_correction_fps'] = benchmark_speed(fm_corr, device)
    print(f"FM Correction: {results['fm_correction']:.1f}%, {results['fm_correction_fps']:.0f} FPS")

    # 4. Model-based planning
    print("\n--- Model-Based Planning ---")
    mb_planner = train_model_based(data, device)
    results['model_based'] = evaluate(mb_planner, device, use_planning=True)
    results['model_based_fps'] = benchmark_speed(mb_planner, device, use_planning=True)
    print(f"Model-Based: {results['model_based']:.1f}%, {results['model_based_fps']:.0f} FPS")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Forward Model Ablation Results")
    print("=" * 60)
    print(f"  Baseline (no FM):         {results['baseline']:.1f}% @ {results['baseline_fps']:.0f} FPS")
    print(f"  FM (auxiliary task):      {results['fm_auxiliary']:.1f}% @ {results['fm_auxiliary_fps']:.0f} FPS")
    print(f"  FM (action correction):   {results['fm_correction']:.1f}% @ {results['fm_correction_fps']:.0f} FPS")
    print(f"  Model-Based Planning:     {results['model_based']:.1f}% @ {results['model_based_fps']:.0f} FPS")

    return results


if __name__ == "__main__":
    main()
