#!/usr/bin/env python3
"""
Quick test to verify pre-trained selector works with different directions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.action_generator.brain_model import BrainInspiredActionGenerator


def test_pretrained():
    """Test the pre-trained model generates different actions for different directions."""
    print("="*60)
    print("Testing Pre-trained Action Generator")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create action generator
    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)

    # Load pre-trained weights
    checkpoint_path = "/workspace/src/checkpoints/pretrained_selector.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        action_generator.load_state_dict(checkpoint['action_generator'])
        print("Loaded successfully!")
    else:
        print("WARNING: No pre-trained checkpoint found!")
        return False

    # Test different directions
    test_plans = [
        {"name": "Move LEFT", "direction": "left", "expected_dim": 0, "expected_sign": -1},
        {"name": "Move RIGHT", "direction": "right", "expected_dim": 0, "expected_sign": 1},
        {"name": "Move FORWARD", "direction": "forward", "expected_dim": 1, "expected_sign": 1},
        {"name": "Move BACKWARD", "direction": "backward", "expected_dim": 1, "expected_sign": -1},
        {"name": "Move UP", "direction": "up", "expected_dim": 2, "expected_sign": 1},
        {"name": "Move DOWN", "direction": "down", "expected_dim": 2, "expected_sign": -1},
    ]

    proprio = torch.zeros(1, 15, device=device)

    print("\n" + "-"*60)
    print("Testing direction-to-action mapping:")
    print("-"*60)

    results = []

    for test in test_plans:
        plan = {
            "observation": {"distance_to_target": "far"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": test["direction"], "speed": "fast", "steps": 3}],
                "gripper": "open"
            }
        }

        with torch.no_grad():
            action_chunk, components = action_generator(
                [plan], proprio, return_components=True
            )

        action_chunk = action_chunk.squeeze(0).cpu().numpy()
        primitive_weights = components['primitive_weights'][0].cpu().numpy()

        # Get average action
        avg_action = action_chunk.mean(axis=0)

        # Check which dimension has the most movement
        pos_action = avg_action[:3]
        max_dim = np.argmax(np.abs(pos_action))
        sign = np.sign(pos_action[test["expected_dim"]])

        is_correct = (max_dim == test["expected_dim"] and sign == test["expected_sign"])
        results.append(is_correct)

        print(f"\n{test['name']}:")
        print(f"  Top primitive: {np.argmax(primitive_weights)} (weights: {primitive_weights.round(2)})")
        print(f"  Avg position action: [{avg_action[0]:.3f}, {avg_action[1]:.3f}, {avg_action[2]:.3f}]")
        print(f"  Expected: dim={test['expected_dim']}, sign={test['expected_sign']}")
        print(f"  Actual: dim={max_dim}, sign={int(sign)}")
        print(f"  {'PASS' if is_correct else 'FAIL'}")

    # Summary
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} passed")
    print("="*60)

    return sum(results) >= len(results) // 2


if __name__ == "__main__":
    success = test_pretrained()
    sys.exit(0 if success else 1)
