#!/usr/bin/env python3
"""
Test script to verify brain-robot components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_plan_encoder():
    """Test the plan encoder."""
    print("\n" + "="*50)
    print("Testing Plan Encoder")
    print("="*50)

    from brain_robot.action_generator.plan_encoder import RelativePlanEncoder

    encoder = RelativePlanEncoder(embed_dim=128, max_movements=5, hidden_dim=64)

    # Test plan
    plan = {
        "observation": {
            "target_object": "red bowl",
            "gripper_position": "above",
            "distance_to_target": "medium",
            "obstacles": []
        },
        "plan": {
            "phase": "approach",
            "movements": [
                {"direction": "left", "speed": "fast", "steps": 3},
                {"direction": "down", "speed": "slow", "steps": 1}
            ],
            "gripper": "open",
            "confidence": 0.9
        },
        "reasoning": "Moving to approach the bowl"
    }

    # Encode
    embeddings = encoder([plan])
    print(f"Input plan: {plan['plan']['phase']}, {len(plan['plan']['movements'])} movements")
    print(f"Output embedding shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings).item():.4f}")
    print("✓ Plan Encoder works!")


def test_forward_model():
    """Test the forward model (cerebellum)."""
    print("\n" + "="*50)
    print("Testing Forward Model (Cerebellum)")
    print("="*50)

    from brain_robot.action_generator.forward_model import CerebellumForwardModel

    model = CerebellumForwardModel(proprio_dim=15, action_dim=7, hidden_dim=128)

    # Test inputs
    proprio = torch.randn(4, 15)  # batch of 4
    action = torch.randn(4, 7)

    # Forward pass
    next_proprio = model(proprio, action)
    print(f"Input proprio shape: {proprio.shape}")
    print(f"Input action shape: {action.shape}")
    print(f"Output next_proprio shape: {next_proprio.shape}")

    # Test loss computation
    proprio_seq = torch.randn(2, 10, 15)  # batch=2, timesteps=10
    action_seq = torch.randn(2, 10, 7)
    loss = model.compute_loss(proprio_seq, action_seq)
    print(f"Prediction loss: {loss.item():.4f}")
    print("✓ Forward Model works!")


def test_brain_model():
    """Test the full brain-inspired action generator."""
    print("\n" + "="*50)
    print("Testing Brain-Inspired Action Generator")
    print("="*50)

    from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator

    model = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    )

    # Test plan
    plan = {
        "observation": {
            "target_object": "bowl",
            "gripper_position": "above",
            "distance_to_target": "far",
            "obstacles": []
        },
        "plan": {
            "phase": "approach",
            "movements": [
                {"direction": "forward", "speed": "fast", "steps": 3}
            ],
            "gripper": "open",
            "confidence": 0.8
        },
        "reasoning": "Moving forward"
    }

    proprio = torch.randn(1, 15)

    # Forward pass
    actions, components = model([plan], proprio, return_components=True)

    print(f"Input proprio shape: {proprio.shape}")
    print(f"Output actions shape: {actions.shape}")
    print(f"Primitive weights shape: {components['primitive_weights'].shape}")
    print(f"Primitive weights: {components['primitive_weights'][0].detach().numpy()}")

    # Show primitive usage
    print("\nPrimitive Usage:")
    model.get_primitive_usage([plan])

    print("\n✓ Brain-Inspired Action Generator works!")


def test_reward_shaper():
    """Test the reward shaper."""
    print("\n" + "="*50)
    print("Testing Reward Shaper")
    print("="*50)

    from brain_robot.training.rewards import RewardShaper

    shaper = RewardShaper()

    # Test data
    obs = {'proprio': np.array([0.0, 0.0, 0.5] + [0.0]*12)}
    next_obs = {'proprio': np.array([0.0, 0.1, 0.5] + [0.0]*12)}  # Moved forward
    action = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, -1.0])  # Moving forward, gripper open
    plan = {
        "plan": {
            "movements": [{"direction": "forward", "speed": "medium", "steps": 2}],
            "gripper": "open"
        }
    }
    info = {"success": False}

    reward = shaper.compute_reward(obs, action, next_obs, plan, info)
    print(f"Computed reward: {reward:.4f}")
    print("✓ Reward Shaper works!")


def main():
    print("="*60)
    print("Brain-Robot Component Tests")
    print("="*60)

    try:
        test_plan_encoder()
        test_forward_model()
        test_brain_model()
        test_reward_shaper()

        print("\n" + "="*60)
        print("All component tests passed!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
