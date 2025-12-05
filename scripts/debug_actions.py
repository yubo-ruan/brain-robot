#!/usr/bin/env python3
"""
Debug script to understand how plans translate to actions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.action_generator.brain_model import BrainInspiredActionGenerator
from src.action_generator.plan_encoder import RelativePlanEncoder


def debug_action_generation():
    """Debug how plans translate to actions."""
    print("="*60)
    print("Debugging Action Generation")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create action generator
    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)

    # Test different plans
    test_plans = [
        {
            "name": "Move left",
            "plan": {
                "observation": {"distance_to_target": "far"},
                "plan": {
                    "phase": "approach",
                    "movements": [{"direction": "left", "speed": "fast", "steps": 3}],
                    "gripper": "open"
                }
            }
        },
        {
            "name": "Move right",
            "plan": {
                "observation": {"distance_to_target": "far"},
                "plan": {
                    "phase": "approach",
                    "movements": [{"direction": "right", "speed": "fast", "steps": 3}],
                    "gripper": "open"
                }
            }
        },
        {
            "name": "Move forward",
            "plan": {
                "observation": {"distance_to_target": "far"},
                "plan": {
                    "phase": "approach",
                    "movements": [{"direction": "forward", "speed": "fast", "steps": 3}],
                    "gripper": "open"
                }
            }
        },
        {
            "name": "Move down",
            "plan": {
                "observation": {"distance_to_target": "close"},
                "plan": {
                    "phase": "descend",
                    "movements": [{"direction": "down", "speed": "slow", "steps": 1}],
                    "gripper": "open"
                }
            }
        },
        {
            "name": "Grasp (close gripper)",
            "plan": {
                "observation": {"distance_to_target": "touching"},
                "plan": {
                    "phase": "grasp",
                    "movements": [],
                    "gripper": "close"
                }
            }
        },
    ]

    proprio = torch.zeros(1, 15, device=device)

    for test in test_plans:
        print(f"\n{'-'*50}")
        print(f"Plan: {test['name']}")
        print(f"{'-'*50}")

        with torch.no_grad():
            action_chunk, components = action_generator(
                [test['plan']], proprio, return_components=True
            )

        action_chunk = action_chunk.squeeze(0).cpu().numpy()
        primitive_weights = components['primitive_weights'][0].cpu().numpy()

        print(f"Primitive weights: {primitive_weights}")
        print(f"Top primitive: {np.argmax(primitive_weights)}")
        print(f"\nAction chunk shape: {action_chunk.shape}")
        print(f"First 3 actions:")
        for i in range(min(3, len(action_chunk))):
            a = action_chunk[i]
            print(f"  Action {i}: pos=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}], "
                  f"rot=[{a[3]:.3f}, {a[4]:.3f}, {a[5]:.3f}], gripper={a[6]:.3f}")

        # Check if action matches expected direction
        avg_action = action_chunk.mean(axis=0)
        print(f"\nAvg action: pos=[{avg_action[0]:.3f}, {avg_action[1]:.3f}, {avg_action[2]:.3f}]")


def debug_plan_encoder():
    """Debug plan encoder embeddings."""
    print("\n" + "="*60)
    print("Debugging Plan Encoder")
    print("="*60)

    encoder = RelativePlanEncoder(embed_dim=128)

    test_plans = [
        {"phase": "approach", "direction": "left", "speed": "fast", "gripper": "open"},
        {"phase": "approach", "direction": "right", "speed": "fast", "gripper": "open"},
        {"phase": "descend", "direction": "down", "speed": "slow", "gripper": "open"},
        {"phase": "grasp", "direction": None, "speed": None, "gripper": "close"},
    ]

    for test in test_plans:
        plan = {
            "observation": {"distance_to_target": "medium"},
            "plan": {
                "phase": test["phase"],
                "movements": [{"direction": test["direction"], "speed": test["speed"], "steps": 2}] if test["direction"] else [],
                "gripper": test["gripper"]
            }
        }

        embedding = encoder([plan])
        print(f"\n{test['phase']}, {test['direction']}, {test['gripper']}:")
        print(f"  Embedding mean: {embedding.mean().item():.4f}, std: {embedding.std().item():.4f}")
        print(f"  Embedding[:10]: {embedding[0, :10].tolist()}")


def debug_primitives():
    """Debug the motion primitives directly."""
    print("\n" + "="*60)
    print("Debugging Motion Primitives (CPGs)")
    print("="*60)

    from src.action_generator.brain_model import BrainInspiredActionGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to(device)

    # Look at the primitive patterns directly
    cpg = action_generator.cpg

    print(f"\nNumber of primitives: {cpg.num_primitives}")
    print(f"Chunk size: {cpg.chunk_size}")
    print(f"Action dim: {cpg.action_dim}")

    # Get all primitive outputs
    primitive_names = ["move_left", "move_right", "move_forward", "move_backward",
                       "move_up", "move_down", "gripper_open", "gripper_close"]

    for i, name in enumerate(primitive_names):
        one_hot = torch.zeros(1, cpg.num_primitives, device=device)
        one_hot[0, i] = 1.0

        with torch.no_grad():
            pattern = cpg(one_hot)  # [1, chunk_size, action_dim]

        pattern = pattern.squeeze(0).cpu().numpy()
        avg = pattern.mean(axis=0)
        print(f"\n{name} (primitive {i}):")
        print(f"  Avg action: pos=[{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}], gripper={avg[6]:.3f}")


if __name__ == "__main__":
    debug_action_generation()
    debug_plan_encoder()
    debug_primitives()
