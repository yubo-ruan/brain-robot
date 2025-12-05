#!/usr/bin/env python3
"""
Pre-train the primitive selector to map directions to correct primitives.
This gives the model a good starting point before RL training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.action_generator.brain_model import BrainInspiredActionGenerator


def generate_training_data(batch_size=32):
    """Generate supervised training data for primitive selection."""
    # Direction to primitive index mapping
    direction_to_primitive = {
        'left': 0,
        'right': 1,
        'forward': 2,
        'backward': 3,
        'up': 4,
        'down': 5,
    }

    gripper_to_primitive = {
        'open': 6,
        'close': 7,
        'maintain': None,  # No specific primitive needed
    }

    directions = list(direction_to_primitive.keys())
    speeds = ['very_slow', 'slow', 'medium', 'fast']
    phases = ['approach', 'align', 'descend', 'grasp', 'lift', 'move', 'place', 'release']
    distances = ['far', 'medium', 'close', 'touching']
    grippers = ['open', 'close', 'maintain']

    plans = []
    targets = []

    for _ in range(batch_size):
        # Random direction
        direction = np.random.choice(directions)
        speed = np.random.choice(speeds)
        phase = np.random.choice(phases)
        distance = np.random.choice(distances)
        gripper = np.random.choice(grippers)

        plan = {
            "observation": {"distance_to_target": distance},
            "plan": {
                "phase": phase,
                "movements": [{"direction": direction, "speed": speed, "steps": np.random.randint(1, 4)}],
                "gripper": gripper
            }
        }

        # Target: one-hot for direction primitive + gripper primitive (if applicable)
        target = torch.zeros(8)

        # Main movement primitive
        target[direction_to_primitive[direction]] = 1.0

        # Gripper primitive (lower weight)
        if gripper in ['open', 'close']:
            target[gripper_to_primitive[gripper]] = 0.3

        # Normalize
        target = target / target.sum()

        plans.append(plan)
        targets.append(target)

    return plans, torch.stack(targets)


def pretrain_selector(
    action_generator,
    device,
    num_steps=1000,
    batch_size=64,
    lr=1e-3,
):
    """Pre-train the primitive selector with supervised learning."""
    print("="*60)
    print("Pre-training Primitive Selector")
    print("="*60)

    # Only train the selector
    optimizer = torch.optim.Adam(action_generator.selector.parameters(), lr=lr)

    losses = []

    for step in range(num_steps):
        # Generate batch
        plans, targets = generate_training_data(batch_size)
        targets = targets.to(device)

        # Get plan embeddings
        plan_embed = action_generator.plan_encoder(plans).to(device)

        # Get predicted primitive weights
        pred_weights = action_generator.selector(plan_embed)

        # Cross-entropy loss (KL divergence)
        loss = F.kl_div(pred_weights.log(), targets, reduction='batchmean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Average loss (last 100): {np.mean(losses[-100:]):.6f}")

    return action_generator


def verify_selector(action_generator, device):
    """Verify the selector works correctly after pre-training."""
    print("\n" + "="*60)
    print("Verifying Primitive Selection")
    print("="*60)

    test_plans = [
        {"direction": "left", "expected_primitive": 0},
        {"direction": "right", "expected_primitive": 1},
        {"direction": "forward", "expected_primitive": 2},
        {"direction": "backward", "expected_primitive": 3},
        {"direction": "up", "expected_primitive": 4},
        {"direction": "down", "expected_primitive": 5},
    ]

    correct = 0
    total = len(test_plans)

    for test in test_plans:
        plan = {
            "observation": {"distance_to_target": "medium"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": test["direction"], "speed": "medium", "steps": 2}],
                "gripper": "open"
            }
        }

        plan_embed = action_generator.plan_encoder([plan]).to(device)
        weights = action_generator.selector(plan_embed)
        top_primitive = weights.argmax(dim=-1).item()

        is_correct = top_primitive == test["expected_primitive"]
        correct += int(is_correct)

        print(f"  {test['direction']:10} -> primitive {top_primitive} "
              f"(expected {test['expected_primitive']}) {'✓' if is_correct else '✗'}")
        print(f"    weights: {weights[0].detach().cpu().numpy().round(3)}")

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    return accuracy >= 80


def main():
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

    # Pre-train
    action_generator = pretrain_selector(
        action_generator,
        device,
        num_steps=2000,
        batch_size=64,
        lr=1e-3,
    )

    # Verify
    success = verify_selector(action_generator, device)

    if success:
        # Save checkpoint
        checkpoint_path = "/workspace/src/checkpoints/pretrained_selector.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'action_generator': action_generator.state_dict(),
        }, checkpoint_path)
        print(f"\nCheckpoint saved to: {checkpoint_path}")
    else:
        print("\nPre-training did not achieve sufficient accuracy. Training longer may help.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
