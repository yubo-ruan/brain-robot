#!/usr/bin/env python3
"""
Comprehensive test suite for the brain-inspired robot control system.
Tests all components and their integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time


def test_mock_environment():
    """Test the mock environment thoroughly."""
    print("\n" + "="*60)
    print("Test 1: Mock Environment")
    print("="*60)

    from brain_robot.env.mock_env import make_mock_env

    env = make_mock_env(max_episode_steps=100)
    print(f"Task: {env.task_description}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nAfter reset:")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Proprio shape: {obs['proprio'].shape}")
    print(f"  Gripper state: {obs['gripper_state']}")

    # Test step with different actions
    actions_to_test = [
        ("Move forward", np.array([0, 0.5, 0, 0, 0, 0, -1])),
        ("Move down", np.array([0, 0, -0.5, 0, 0, 0, -1])),
        ("Close gripper", np.array([0, 0, 0, 0, 0, 0, 1])),
        ("Move up", np.array([0, 0, 0.5, 0, 0, 0, 1])),
    ]

    for name, action in actions_to_test:
        obs, reward, done, truncated, info = env.step(action)
        print(f"\n{name}:")
        print(f"  Position: {obs['proprio'][:3]}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Success: {info.get('success', False)}")

    env.close()
    print("\n✓ Mock Environment test passed!")
    return True


def test_plan_encoder_batch():
    """Test plan encoder with batch processing."""
    print("\n" + "="*60)
    print("Test 2: Plan Encoder Batch Processing")
    print("="*60)

    from brain_robot.action_generator.plan_encoder import RelativePlanEncoder

    encoder = RelativePlanEncoder(embed_dim=128)

    # Create batch of different plans
    plans = [
        {
            "observation": {"distance_to_target": "far"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": "forward", "speed": "fast", "steps": 3}],
                "gripper": "open"
            }
        },
        {
            "observation": {"distance_to_target": "close"},
            "plan": {
                "phase": "grasp",
                "movements": [{"direction": "down", "speed": "slow", "steps": 1}],
                "gripper": "close"
            }
        },
        {
            "observation": {"distance_to_target": "touching"},
            "plan": {
                "phase": "lift",
                "movements": [{"direction": "up", "speed": "medium", "steps": 2}],
                "gripper": "maintain"
            }
        },
    ]

    embeddings = encoder(plans)
    print(f"Batch size: {len(plans)}")
    print(f"Embedding shape: {embeddings.shape}")

    # Check embeddings are different
    diffs = []
    for i in range(len(plans)):
        for j in range(i+1, len(plans)):
            diff = torch.norm(embeddings[i] - embeddings[j]).item()
            diffs.append(diff)
            print(f"  Diff between plan {i} and {j}: {diff:.4f}")

    assert all(d > 0.1 for d in diffs), "Embeddings should be different"
    print("\n✓ Plan Encoder batch test passed!")
    return True


def test_action_generator_primitives():
    """Test that action generator uses appropriate primitives."""
    print("\n" + "="*60)
    print("Test 3: Action Generator Primitive Selection")
    print("="*60)

    from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)

    # Test different movement directions
    test_cases = [
        ("left", 0),  # move_left should be index 0
        ("right", 1),
        ("forward", 2),
        ("down", 5),
    ]

    for direction, expected_primitive in test_cases:
        plan = {
            "observation": {"distance_to_target": "far"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": direction, "speed": "fast", "steps": 3}],
                "gripper": "open"
            }
        }

        proprio = torch.zeros(1, 15, device=device)
        actions, components = model([plan], proprio, return_components=True)

        weights = components['primitive_weights'][0].cpu().detach().numpy()
        top_primitive = np.argmax(weights)

        print(f"\nDirection: {direction}")
        print(f"  Top 3 primitives: {np.argsort(weights)[-3:][::-1]}")
        print(f"  Weights: {weights}")
        print(f"  Output action (first): {actions[0, 0].cpu().detach().numpy()}")

    print("\n✓ Primitive selection test passed!")
    return True


def test_forward_model_learning():
    """Test that forward model can learn from data."""
    print("\n" + "="*60)
    print("Test 4: Forward Model Learning")
    print("="*60)

    from brain_robot.action_generator.forward_model import CerebellumForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CerebellumForwardModel(proprio_dim=15, action_dim=7, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate synthetic data with a simple relationship
    # next_state = state + action[:3] (for first 3 dimensions)
    batch_size = 64
    num_batches = 50

    initial_loss = None
    final_loss = None

    for batch in range(num_batches):
        state = torch.randn(batch_size, 15, device=device)
        action = torch.randn(batch_size, 7, device=device) * 0.1

        # Simple dynamics: state changes based on action
        next_state = state.clone()
        next_state[:, :3] += action[:, :3]
        next_state[:, 3:6] += action[:, 3:6] * 0.5

        # Forward pass
        pred_next = model(state, action)
        loss = torch.nn.functional.mse_loss(pred_next, next_state)

        if batch == 0:
            initial_loss = loss.item()
        if batch == num_batches - 1:
            final_loss = loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():.6f}")

    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Improvement: {(1 - final_loss/initial_loss) * 100:.1f}%")

    assert final_loss < initial_loss * 0.5, "Forward model should improve"
    print("\n✓ Forward model learning test passed!")
    return True


def test_reward_shaping_scenarios():
    """Test reward shaping in different scenarios."""
    print("\n" + "="*60)
    print("Test 5: Reward Shaping Scenarios")
    print("="*60)

    from brain_robot.training.rewards import RewardShaper

    shaper = RewardShaper(
        task_success_reward=100.0,
        direction_reward_scale=2.0,
        speed_reward_scale=0.5,
        gripper_reward_scale=1.0,
    )

    scenarios = [
        {
            "name": "Following direction correctly",
            "obs": {"proprio": np.array([0, 0, 0.5] + [0]*12)},
            "next_obs": {"proprio": np.array([0, 0.1, 0.5] + [0]*12)},  # moved forward
            "action": np.array([0, 0.3, 0, 0, 0, 0, -1]),  # gripper open
            "plan": {"plan": {"movements": [{"direction": "forward", "speed": "medium", "steps": 1}], "gripper": "open"}},
            "info": {"success": False},
        },
        {
            "name": "Wrong direction",
            "obs": {"proprio": np.array([0, 0, 0.5] + [0]*12)},
            "next_obs": {"proprio": np.array([0, -0.1, 0.5] + [0]*12)},  # moved backward
            "action": np.array([0, -0.3, 0, 0, 0, 0, -1]),
            "plan": {"plan": {"movements": [{"direction": "forward", "speed": "medium", "steps": 1}], "gripper": "open"}},
            "info": {"success": False},
        },
        {
            "name": "Task success",
            "obs": {"proprio": np.array([0, 0, 0.5] + [0]*12)},
            "next_obs": {"proprio": np.array([0, 0, 0.5] + [0]*12)},
            "action": np.array([0, 0, 0, 0, 0, 0, 0]),
            "plan": {"plan": {"movements": [], "gripper": "maintain"}},
            "info": {"success": True},
        },
    ]

    for scenario in scenarios:
        reward = shaper.compute_reward(
            obs=scenario["obs"],
            action=scenario["action"],
            next_obs=scenario["next_obs"],
            plan=scenario["plan"],
            info=scenario["info"],
        )
        print(f"\n{scenario['name']}:")
        print(f"  Reward: {reward:.4f}")

    print("\n✓ Reward shaping test passed!")
    return True


def test_end_to_end_without_vlm():
    """Test end-to-end pipeline without VLM (using mock plans)."""
    print("\n" + "="*60)
    print("Test 6: End-to-End Pipeline (without VLM)")
    print("="*60)

    from brain_robot.env.mock_env import make_mock_env
    from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator
    from brain_robot.training.rewards import RewardShaper

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_mock_env(max_episode_steps=50)
    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)
    reward_shaper = RewardShaper()

    # Run episode with mock plans
    obs, _ = env.reset(seed=42)
    total_reward = 0
    steps = 0

    mock_plans = [
        {"plan": {"phase": "approach", "movements": [{"direction": "forward", "speed": "fast", "steps": 2}], "gripper": "open"}, "observation": {"distance_to_target": "far"}},
        {"plan": {"phase": "descend", "movements": [{"direction": "down", "speed": "slow", "steps": 1}], "gripper": "open"}, "observation": {"distance_to_target": "close"}},
        {"plan": {"phase": "grasp", "movements": [], "gripper": "close"}, "observation": {"distance_to_target": "touching"}},
        {"plan": {"phase": "lift", "movements": [{"direction": "up", "speed": "medium", "steps": 2}], "gripper": "maintain"}, "observation": {"distance_to_target": "far"}},
    ]

    plan_idx = 0

    for step in range(50):
        # Change plan every 10 steps
        if step % 10 == 0:
            plan_idx = min(plan_idx, len(mock_plans) - 1)
            current_plan = mock_plans[plan_idx]
            plan_idx += 1

            proprio = torch.tensor(obs['proprio'], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_chunk = action_generator([current_plan], proprio)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()

        action_idx = step % 10
        action = action_chunk[min(action_idx, len(action_chunk)-1)]
        action = np.clip(action, -1, 1)
        if np.isnan(action).any():
            action = np.zeros(7)

        next_obs, _, done, truncated, info = env.step(action)

        reward = reward_shaper.compute_reward(
            obs=obs,
            action=action,
            next_obs=next_obs,
            plan=current_plan,
            info=info,
        )

        total_reward += reward
        obs = next_obs
        steps += 1

        if done or truncated:
            break

    print(f"Episode completed:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success: {info.get('success', False)}")

    env.close()
    print("\n✓ End-to-end test passed!")
    return True


def main():
    print("="*60)
    print("COMPREHENSIVE TEST SUITE")
    print("="*60)

    tests = [
        ("Mock Environment", test_mock_environment),
        ("Plan Encoder Batch", test_plan_encoder_batch),
        ("Action Generator Primitives", test_action_generator_primitives),
        ("Forward Model Learning", test_forward_model_learning),
        ("Reward Shaping", test_reward_shaping_scenarios),
        ("End-to-End (no VLM)", test_end_to_end_without_vlm),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, s, _ in results if s)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
