#!/usr/bin/env python3
"""Test Grounding-DINO vs YOLO on grocery detection tasks.

Compares the two detectors on libero_object tasks where YOLO V4
misclassifies grocery items.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


# Target objects from libero_object tasks (the problematic grocery items)
GROCERY_TASKS = [
    (0, "alphabet_soup"),
    (1, "basket"),  # Not grocery but included for comparison
    (2, "salad_dressing"),
    (3, "bbq_sauce"),
    (4, "bbq_sauce"),
    (5, "bbq_sauce"),
    (6, "basket"),
    (7, "alphabet_soup"),
    (8, "wine_bottle"),
    (9, "basket"),
]


def make_libero_env(task_id: int):
    """Create LIBERO environment for libero_object task."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark("libero_object")()
    task = benchmark.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)
    return env, task


def get_image_from_env(env) -> np.ndarray:
    """Get RGB image from environment."""
    img = env.sim.render(
        camera_name="agentview",
        height=256,
        width=256,
        mode='offscreen'
    )
    return img[::-1]  # Flip vertically


def test_detector(detector, env, target: str, n_frames: int = 10):
    """Test detector on environment frames.

    Returns:
        Dict with detection rate, avg confidence, detected classes
    """
    detections_count = 0
    total_confidence = 0.0
    detected_classes = defaultdict(int)

    env.reset()

    for _ in range(n_frames):
        # Random action for diversity
        action = np.random.uniform(-0.3, 0.3, 7)
        action[6] = np.random.choice([-1, 1])
        try:
            env.step(action)
        except:
            break

        img = get_image_from_env(env)

        # Detect
        start = time.time()
        detections = detector.detect(img)
        elapsed = time.time() - start

        # Check for target
        for det in detections:
            detected_classes[det.class_name] += 1
            if det.class_name == target:
                detections_count += 1
                total_confidence += det.confidence

    return {
        "target": target,
        "detection_rate": detections_count / n_frames,
        "avg_confidence": total_confidence / max(1, detections_count),
        "detected_classes": dict(detected_classes),
        "n_frames": n_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Grounding-DINO vs YOLO")
    parser.add_argument("--tasks", type=int, nargs="+", default=[3, 4, 5],
                        help="Task IDs to test (default: bbq_sauce tasks)")
    parser.add_argument("--frames", type=int, default=10,
                        help="Frames per task")
    parser.add_argument("--yolo-only", action="store_true",
                        help="Only test YOLO")
    parser.add_argument("--gdino-only", action="store_true",
                        help="Only test Grounding-DINO")
    args = parser.parse_args()

    print("=" * 70)
    print("GROUNDING-DINO vs YOLO COMPARISON")
    print("=" * 70)

    # Load detectors
    yolo_detector = None
    gdino_detector = None

    if not args.gdino_only:
        print("\nLoading YOLO V4...")
        from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector
        yolo_detector = YOLOObjectDetector(model_path="models/yolo_libero_v4.pt")
        yolo_detector.warmup()

    if not args.yolo_only:
        print("\nLoading Grounding-DINO...")
        from brain_robot.perception.detection.grounding_dino_detector import GroundingDINODetector
        gdino_detector = GroundingDINODetector()
        gdino_detector.warmup()

    # Results storage
    yolo_results = []
    gdino_results = []

    for task_id in args.tasks:
        target = GROCERY_TASKS[task_id][1]
        print(f"\n{'='*70}")
        print(f"Task {task_id}: Target = {target}")
        print("=" * 70)

        env, task = make_libero_env(task_id)
        print(f"Task name: {task.name}")

        # Test YOLO
        if yolo_detector:
            print(f"\nTesting YOLO V4...")
            result = test_detector(yolo_detector, env, target, args.frames)
            yolo_results.append(result)

            status = "✓ FOUND" if result["detection_rate"] > 0.1 else "✗ MISSED"
            print(f"  {status}")
            print(f"  Detection rate: {result['detection_rate']*100:.1f}%")
            print(f"  Avg confidence: {result['avg_confidence']:.2f}")
            print(f"  Detected classes: {result['detected_classes']}")

        # Test Grounding-DINO
        if gdino_detector:
            print(f"\nTesting Grounding-DINO...")
            # Set target for detection
            gdino_detector.set_target_objects([target])
            result = test_detector(gdino_detector, env, target, args.frames)
            gdino_results.append(result)

            status = "✓ FOUND" if result["detection_rate"] > 0.1 else "✗ MISSED"
            print(f"  {status}")
            print(f"  Detection rate: {result['detection_rate']*100:.1f}%")
            print(f"  Avg confidence: {result['avg_confidence']:.2f}")
            print(f"  Detected classes: {result['detected_classes']}")

        env.close()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    if yolo_results:
        yolo_found = sum(1 for r in yolo_results if r["detection_rate"] > 0.1)
        print(f"\nYOLO V4:")
        print(f"  Tasks with target found: {yolo_found}/{len(yolo_results)}")
        print(f"  Avg detection rate: {np.mean([r['detection_rate'] for r in yolo_results])*100:.1f}%")

    if gdino_results:
        gdino_found = sum(1 for r in gdino_results if r["detection_rate"] > 0.1)
        print(f"\nGrounding-DINO:")
        print(f"  Tasks with target found: {gdino_found}/{len(gdino_results)}")
        print(f"  Avg detection rate: {np.mean([r['detection_rate'] for r in gdino_results])*100:.1f}%")

    if yolo_results and gdino_results:
        print(f"\nComparison:")
        for yr, gr in zip(yolo_results, gdino_results):
            yolo_status = "✓" if yr["detection_rate"] > 0.1 else "✗"
            gdino_status = "✓" if gr["detection_rate"] > 0.1 else "✗"
            print(f"  {yr['target']}: YOLO {yolo_status} ({yr['detection_rate']*100:.0f}%) vs GDINO {gdino_status} ({gr['detection_rate']*100:.0f}%)")


if __name__ == "__main__":
    main()
