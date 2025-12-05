#!/usr/bin/env python3
"""
Test the VLM planner with the downloaded Qwen2.5-VL model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image


def test_vlm_planner():
    """Test the VLM planner."""
    print("\n" + "="*60)
    print("Testing VLM Planner (Qwen2.5-VL-7B)")
    print("="*60)

    from src.vlm.qwen_planner import QwenVLPlanner

    # Create a simple test image (random colored image simulating a robot scene)
    print("\nCreating test image...")
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add some colored rectangles to simulate objects
    test_image[50:100, 50:100] = [255, 0, 0]  # Red object
    test_image[150:200, 150:200] = [0, 255, 0]  # Green object
    test_image[100:150, 100:150] = [128, 128, 128]  # Gray (gripper)

    # Initialize planner
    print("\nLoading Qwen2.5-VL planner...")
    planner = QwenVLPlanner(
        model_name="/workspace/src/models/qwen2.5-vl-7b",
        device="cuda:0",
        max_new_tokens=512,
        temperature=0.1,
    )

    # Test planning
    print("\nGenerating plan for task: 'Pick up the red object'")
    plan = planner.plan(
        image=test_image,
        task_description="Pick up the red object on the left",
        gripper_state="open",
        steps_since_plan=0,
    )

    print("\nGenerated Plan:")
    print(f"  Observation: {plan.get('observation', {})}")
    print(f"  Plan: {plan.get('plan', {})}")
    print(f"  Reasoning: {plan.get('reasoning', 'N/A')}")

    # Verify plan structure
    assert 'plan' in plan, "Plan should have 'plan' key"
    assert 'movements' in plan['plan'], "Plan should have movements"
    assert 'gripper' in plan['plan'], "Plan should have gripper action"

    print("\n✓ VLM Planner works!")
    return True


def test_full_pipeline():
    """Test the full VLM → Action Generator pipeline."""
    print("\n" + "="*60)
    print("Testing Full Pipeline: VLM → Action Generator")
    print("="*60)

    from src.vlm.qwen_planner import QwenVLPlanner
    from src.action_generator.brain_model import BrainInspiredActionGenerator

    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Initialize components
    print("\nInitializing components...")
    planner = QwenVLPlanner(
        model_name="/workspace/src/models/qwen2.5-vl-7b",
        device="cuda:0",
        max_new_tokens=512,
        temperature=0.1,
    )

    action_generator = BrainInspiredActionGenerator(
        plan_dim=128,
        proprio_dim=15,
        action_dim=7,
        chunk_size=10,
        num_primitives=8,
        hidden_dim=128,
    ).to("cuda:0")

    # Generate plan
    print("\nGenerating plan...")
    plan = planner.plan(
        image=test_image,
        task_description="Move the gripper to the center of the scene",
        gripper_state="open",
        steps_since_plan=0,
    )

    # Generate actions
    print("\nGenerating actions from plan...")
    proprio = torch.randn(1, 15, device="cuda:0")
    actions, components = action_generator([plan], proprio, return_components=True)

    print(f"\nGenerated actions shape: {actions.shape}")
    print(f"First action: {actions[0, 0].cpu().detach().numpy()}")

    # Show primitive weights
    weights = components['primitive_weights'][0].cpu().detach().numpy()
    print(f"\nPrimitive weights: {weights}")

    print("\n✓ Full Pipeline works!")
    return True


def main():
    print("="*60)
    print("VLM Integration Tests")
    print("="*60)

    try:
        test_vlm_planner()
        test_full_pipeline()

        print("\n" + "="*60)
        print("All VLM tests passed!")
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
