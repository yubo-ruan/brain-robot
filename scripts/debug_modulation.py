#!/usr/bin/env python3
"""
Debug the modulation chain to see where actions get corrupted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.action_generator.brain_model import BrainInspiredActionGenerator


def debug_modulation():
    """Debug the complete modulation chain."""
    print("="*60)
    print("Debugging Modulation Chain")
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

    # Load pre-trained weights
    checkpoint_path = "/workspace/src/checkpoints/pretrained_selector.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        action_generator.load_state_dict(checkpoint['action_generator'])
        print("Loaded pre-trained checkpoint")

    # Test just the CPG primitives directly
    print("\n" + "-"*60)
    print("Raw primitive outputs (CPGs):")
    print("-"*60)

    for i in range(6):
        prim = action_generator.primitives.primitives[i]
        avg = prim.mean(dim=0).detach().cpu().numpy()
        name = action_generator.primitives.primitive_names[i]
        print(f"  {name}: [{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}]")

    # Test blending with one-hot weights
    print("\n" + "-"*60)
    print("Blended primitive outputs:")
    print("-"*60)

    for i in range(6):
        one_hot = torch.zeros(1, 8, device=device)
        one_hot[0, i] = 1.0

        blended = action_generator.primitives(one_hot)
        avg = blended[0].mean(dim=0).detach().cpu().numpy()
        name = action_generator.primitives.primitive_names[i]
        print(f"  {name}: [{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}]")

    # Test with plan encoding
    print("\n" + "-"*60)
    print("Plan encoded + blended:")
    print("-"*60)

    test_cases = ["left", "right", "forward", "backward", "up", "down"]
    proprio = torch.zeros(1, 15, device=device)

    for direction in test_cases:
        plan = {
            "observation": {"distance_to_target": "far"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": direction, "speed": "fast", "steps": 3}],
                "gripper": "open"
            }
        }

        with torch.no_grad():
            plan_embed = action_generator.plan_encoder([plan]).to(device)
            primitive_weights = action_generator.selector(plan_embed)
            blended = action_generator.primitives(primitive_weights)

            print(f"\n  {direction}:")
            print(f"    weights: {primitive_weights[0].cpu().numpy().round(2)}")
            avg = blended[0].mean(dim=0).cpu().numpy()
            print(f"    blended: [{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}]")

            # Now check modulation
            amplitude, speed, offset = action_generator.modulator(plan_embed, proprio)
            print(f"    amplitude: {amplitude[0].cpu().numpy().round(2)}")
            print(f"    offset: {offset[0].cpu().numpy().round(2)}")

            # Apply modulation
            modulated = blended * amplitude.unsqueeze(1)
            modulated = modulated + offset.unsqueeze(1)
            avg = modulated[0].mean(dim=0).cpu().numpy()
            print(f"    after mod: [{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}]")


def debug_full_chain():
    """Debug the complete forward pass."""
    print("\n" + "="*60)
    print("Full Forward Pass Debug")
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

    # Load pre-trained weights
    checkpoint_path = "/workspace/src/checkpoints/pretrained_selector.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        action_generator.load_state_dict(checkpoint['action_generator'])

    proprio = torch.zeros(1, 15, device=device)

    for direction in ["left", "right", "forward"]:
        plan = {
            "observation": {"distance_to_target": "far"},
            "plan": {
                "phase": "approach",
                "movements": [{"direction": direction, "speed": "fast", "steps": 3}],
                "gripper": "open"
            }
        }

        print(f"\n{direction.upper()}:")

        with torch.no_grad():
            # Step by step
            plan_embed = action_generator.plan_encoder([plan]).to(device)
            primitive_weights = action_generator.selector(plan_embed)
            blended = action_generator.primitives(primitive_weights)
            amplitude, speed, offset = action_generator.modulator(plan_embed, proprio)

            modulated = blended * amplitude.unsqueeze(1)
            modulated = modulated + offset.unsqueeze(1)

            # Forward model + error correction
            first_action = modulated[:, 0, :]
            predicted_next_proprio = action_generator.cerebellum(proprio, first_action)
            prediction_error = predicted_next_proprio - proprio
            correction = action_generator.error_corrector(prediction_error)
            correction_gain = torch.sigmoid(action_generator.correction_gain)
            correction = correction * correction_gain

            corrected = modulated + correction.unsqueeze(1)

            # Temporal smoothing
            smoothed = corrected.permute(0, 2, 1)
            smoothed = action_generator.smoother(smoothed)
            smoothed = smoothed.permute(0, 2, 1)

            # Final output
            actions = torch.tanh(smoothed)

            # Print at each stage
            avg_mod = modulated[0].mean(dim=0).cpu().numpy()
            avg_corr = corrected[0].mean(dim=0).cpu().numpy()
            avg_smooth = smoothed[0].mean(dim=0).cpu().numpy()
            avg_final = actions[0].mean(dim=0).cpu().numpy()

            print(f"  After modulation: [{avg_mod[0]:.3f}, {avg_mod[1]:.3f}, {avg_mod[2]:.3f}]")
            print(f"  Correction: {correction[0].cpu().numpy()[:3].round(3)}")
            print(f"  After correction: [{avg_corr[0]:.3f}, {avg_corr[1]:.3f}, {avg_corr[2]:.3f}]")
            print(f"  After smoothing: [{avg_smooth[0]:.3f}, {avg_smooth[1]:.3f}, {avg_smooth[2]:.3f}]")
            print(f"  Final (tanh): [{avg_final[0]:.3f}, {avg_final[1]:.3f}, {avg_final[2]:.3f}]")


if __name__ == "__main__":
    debug_modulation()
    debug_full_chain()
