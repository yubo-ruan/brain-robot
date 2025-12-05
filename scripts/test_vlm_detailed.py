#!/usr/bin/env python3
"""
Detailed VLM planner tests with different scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import time


def create_test_scene(object_pos="left", gripper_pos="center", target_pos="right"):
    """Create a test scene image."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Background (table)
    image[100:256, :] = [139, 90, 43]  # Brown table

    # Object positions
    positions = {
        "left": (50, 150),
        "center": (128, 150),
        "right": (200, 150),
        "far_left": (30, 150),
        "far_right": (220, 150),
    }

    # Draw object (red cube)
    obj_x, obj_y = positions.get(object_pos, (50, 150))
    image[obj_y-20:obj_y+20, obj_x-20:obj_x+20] = [255, 0, 0]

    # Draw target (green circle outline)
    target_x, target_y = positions.get(target_pos, (200, 150))
    for angle in range(360):
        x = int(target_x + 25 * np.cos(np.radians(angle)))
        y = int(target_y + 25 * np.sin(np.radians(angle)))
        if 0 <= x < 256 and 0 <= y < 256:
            image[y, x] = [0, 255, 0]

    # Draw gripper (gray)
    grip_x, grip_y = positions.get(gripper_pos, (128, 80))
    grip_y = 80  # Gripper is above
    image[grip_y-10:grip_y+10, grip_x-5:grip_x+5] = [128, 128, 128]
    image[grip_y+5:grip_y+20, grip_x-15:grip_x-5] = [128, 128, 128]
    image[grip_y+5:grip_y+20, grip_x+5:grip_x+15] = [128, 128, 128]

    return image


def test_vlm_scenarios():
    """Test VLM with different scenarios."""
    print("="*60)
    print("VLM Detailed Scenario Tests")
    print("="*60)

    from src.vlm.qwen_planner import QwenVLPlanner

    print("\nLoading VLM...")
    planner = QwenVLPlanner(
        model_name="/workspace/src/models/qwen2.5-vl-7b",
        device="cuda:0",
        max_new_tokens=512,
        temperature=0.1,
    )

    scenarios = [
        {
            "name": "Object on left, gripper at center",
            "image_params": {"object_pos": "left", "gripper_pos": "center"},
            "task": "Pick up the red object",
            "gripper_state": "open",
            "expected_direction": "left",
        },
        {
            "name": "Object on right, gripper at center",
            "image_params": {"object_pos": "right", "gripper_pos": "center"},
            "task": "Pick up the red object",
            "gripper_state": "open",
            "expected_direction": "right",
        },
        {
            "name": "Object below gripper (descend)",
            "image_params": {"object_pos": "center", "gripper_pos": "center"},
            "task": "Pick up the red object",
            "gripper_state": "open",
            "expected_phase": "descend",
        },
        {
            "name": "Object at target, release",
            "image_params": {"object_pos": "right", "gripper_pos": "right", "target_pos": "right"},
            "task": "Place the red object on the green target",
            "gripper_state": "closed",
            "expected_gripper": "open",
        },
    ]

    results = []
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*50}")

        # Create image
        image = create_test_scene(**scenario["image_params"])

        # Generate plan
        planner.reset()
        start_time = time.time()
        plan = planner.plan(
            image=image,
            task_description=scenario["task"],
            gripper_state=scenario["gripper_state"],
            steps_since_plan=0,
        )
        elapsed = time.time() - start_time

        print(f"\nTask: {scenario['task']}")
        print(f"Gripper state: {scenario['gripper_state']}")
        print(f"\nGenerated Plan:")
        print(f"  Phase: {plan.get('plan', {}).get('phase', 'N/A')}")
        print(f"  Movements: {plan.get('plan', {}).get('movements', [])}")
        print(f"  Gripper: {plan.get('plan', {}).get('gripper', 'N/A')}")
        print(f"  Confidence: {plan.get('plan', {}).get('confidence', 'N/A')}")
        print(f"  Reasoning: {plan.get('reasoning', 'N/A')}")
        print(f"\nInference time: {elapsed:.2f}s")

        # Check expectations
        checks = []
        if "expected_direction" in scenario:
            movements = plan.get('plan', {}).get('movements', [])
            if movements:
                actual_dir = movements[0].get('direction', '')
                match = actual_dir == scenario['expected_direction']
                checks.append(f"Direction: {actual_dir} (expected: {scenario['expected_direction']}) - {'✓' if match else '✗'}")
            else:
                checks.append(f"Direction: No movements (expected: {scenario['expected_direction']}) - ✗")

        if "expected_phase" in scenario:
            actual_phase = plan.get('plan', {}).get('phase', '')
            match = actual_phase == scenario['expected_phase']
            checks.append(f"Phase: {actual_phase} (expected: {scenario['expected_phase']}) - {'✓' if match else '✗'}")

        if "expected_gripper" in scenario:
            actual_gripper = plan.get('plan', {}).get('gripper', '')
            match = actual_gripper == scenario['expected_gripper']
            checks.append(f"Gripper: {actual_gripper} (expected: {scenario['expected_gripper']}) - {'✓' if match else '✗'}")

        for check in checks:
            print(f"  {check}")

        results.append({
            "name": scenario["name"],
            "plan": plan,
            "checks": checks,
            "time": elapsed,
        })

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Time: {r['time']:.2f}s")
        for check in r['checks']:
            print(f"  {check}")

    return results


def test_vlm_consistency():
    """Test that VLM gives consistent plans for same input."""
    print("\n" + "="*60)
    print("VLM Consistency Test")
    print("="*60)

    from src.vlm.qwen_planner import QwenVLPlanner

    planner = QwenVLPlanner(
        model_name="/workspace/src/models/qwen2.5-vl-7b",
        device="cuda:0",
        max_new_tokens=512,
        temperature=0.1,  # Low temperature for consistency
    )

    image = create_test_scene("left", "center", "right")
    task = "Pick up the red object on the left"

    plans = []
    for i in range(3):
        planner.reset()
        plan = planner.plan(
            image=image,
            task_description=task,
            gripper_state="open",
            steps_since_plan=0,
        )
        plans.append(plan)
        print(f"\nRun {i+1}:")
        print(f"  Phase: {plan.get('plan', {}).get('phase')}")
        print(f"  Movements: {plan.get('plan', {}).get('movements')}")

    # Check consistency
    phases = [p.get('plan', {}).get('phase') for p in plans]
    consistent = len(set(phases)) == 1
    print(f"\nPhase consistency: {'✓' if consistent else '✗'} ({set(phases)})")

    return consistent


def main():
    try:
        results = test_vlm_scenarios()
        consistency = test_vlm_consistency()

        print("\n" + "="*60)
        print("All VLM tests completed!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n❌ VLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
