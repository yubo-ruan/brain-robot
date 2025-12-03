#!/usr/bin/env python3
"""Compare YOLO vs Grounding-DINO on libero_object tasks.

Evaluates target detection accuracy for both detectors across all 10 tasks.

Usage:
    python scripts/compare_yolo_gdino.py
    python scripts/compare_yolo_gdino.py --detector yolo
    python scripts/compare_yolo_gdino.py --detector gdino
    python scripts/compare_yolo_gdino.py --detector both
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


# Task ID -> (target_object, task_description)
LIBERO_OBJECT_TASKS = {
    0: ("alphabet_soup", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    1: ("basket", "pick_up_the_cream_cheese_and_place_it_in_the_basket"),
    2: ("salad_dressing", "pick_up_the_salad_dressing_and_place_it_in_the_basket"),
    3: ("bbq_sauce", "pick_up_the_bbq_sauce_and_place_it_in_the_basket"),
    4: ("ketchup", "pick_up_the_ketchup_and_place_it_in_the_basket"),
    5: ("tomato_sauce", "pick_up_the_tomato_sauce_and_place_it_in_the_basket"),
    6: ("butter", "pick_up_the_butter_and_place_it_in_the_basket"),
    7: ("milk", "pick_up_the_milk_and_place_it_in_the_basket"),
    8: ("chocolate_pudding", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"),
    9: ("orange_juice", "pick_up_the_orange_juice_and_place_it_in_the_basket"),
}


@dataclass
class TaskResult:
    """Result for a single task evaluation."""
    task_id: int
    target: str
    detected: bool
    detection_rate: float  # % of frames with target detected
    avg_confidence: float
    detected_classes: Dict[str, int]  # class -> count
    inference_time_ms: float


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


def evaluate_task(
    detector,
    detector_type: str,
    task_id: int,
    target: str,
    n_episodes: int = 3,
    n_steps: int = 10,
) -> TaskResult:
    """Evaluate detector on a single task."""
    env, task = make_libero_env(task_id)

    target_detections = 0
    total_confidence = 0.0
    detected_classes = defaultdict(int)
    total_frames = 0
    total_time = 0.0

    # Set target for Grounding-DINO
    if detector_type == "gdino":
        detector.set_target_objects([target])

    for ep in range(n_episodes):
        env.reset()

        for step in range(n_steps):
            # Random action for scene diversity
            action = np.random.uniform(-0.3, 0.3, 7)
            action[6] = np.random.choice([-1, 1])
            try:
                env.step(action)
            except:
                break

            img = get_image_from_env(env)
            total_frames += 1

            # Detect
            start = time.time()
            detections = detector.detect(img)
            elapsed = time.time() - start
            total_time += elapsed

            # Check detections
            for det in detections:
                detected_classes[det.class_name] += 1
                if det.class_name == target:
                    target_detections += 1
                    total_confidence += det.confidence

    env.close()

    detection_rate = target_detections / max(1, total_frames)
    avg_confidence = total_confidence / max(1, target_detections)
    avg_time_ms = (total_time / max(1, total_frames)) * 1000

    return TaskResult(
        task_id=task_id,
        target=target,
        detected=detection_rate > 0.1,  # >10% of frames
        detection_rate=detection_rate,
        avg_confidence=avg_confidence,
        detected_classes=dict(detected_classes),
        inference_time_ms=avg_time_ms,
    )


def run_evaluation(detector_type: str, n_episodes: int = 3, n_steps: int = 10):
    """Run full evaluation on all libero_object tasks."""
    print(f"\n{'='*70}")
    print(f"LIBERO_OBJECT EVALUATION - {detector_type.upper()}")
    print(f"{'='*70}")

    # Load detector
    if detector_type == "yolo":
        from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector
        detector = YOLOObjectDetector(model_path="models/yolo_libero_v4.pt")
        detector.warmup()
    else:  # gdino
        from brain_robot.perception.detection.grounding_dino_detector import GroundingDINODetector
        detector = GroundingDINODetector()
        detector.warmup()

    results = []

    for task_id in range(10):
        target = LIBERO_OBJECT_TASKS[task_id][0]
        desc = LIBERO_OBJECT_TASKS[task_id][1]

        print(f"\nTask {task_id}: {target}")
        print(f"  {desc[:60]}...")

        result = evaluate_task(
            detector, detector_type, task_id, target,
            n_episodes=n_episodes, n_steps=n_steps
        )
        results.append(result)

        # Print result
        status = "✓ PASS" if result.detected else "✗ FAIL"
        print(f"  {status} - {result.detection_rate*100:.1f}% detection rate, conf={result.avg_confidence:.2f}")

        if not result.detected and result.detected_classes:
            # Show what was detected instead
            top_classes = sorted(result.detected_classes.items(), key=lambda x: -x[1])[:3]
            print(f"  Detected instead: {top_classes}")

    return results


def print_summary(yolo_results: Optional[List[TaskResult]], gdino_results: Optional[List[TaskResult]]):
    """Print comparison summary."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    headers = ["Task", "Target"]
    if yolo_results:
        headers.extend(["YOLO", "Conf"])
    if gdino_results:
        headers.extend(["GDINO", "Conf"])

    # Print header
    print(f"\n{'Task':<6} {'Target':<20}", end="")
    if yolo_results:
        print(f"{'YOLO':<8} {'Conf':<6}", end="")
    if gdino_results:
        print(f"{'GDINO':<8} {'Conf':<6}", end="")
    print()
    print("-" * 70)

    # Print rows
    for i in range(10):
        target = LIBERO_OBJECT_TASKS[i][0]
        print(f"{i:<6} {target:<20}", end="")

        if yolo_results:
            r = yolo_results[i]
            status = "✓" if r.detected else "✗"
            print(f"{status:<8} {r.avg_confidence:.2f}  ", end="")

        if gdino_results:
            r = gdino_results[i]
            status = "✓" if r.detected else "✗"
            print(f"{status:<8} {r.avg_confidence:.2f}  ", end="")

        print()

    # Print totals
    print("-" * 70)
    print(f"{'TOTAL':<6} {'':<20}", end="")

    if yolo_results:
        passed = sum(1 for r in yolo_results if r.detected)
        avg_time = np.mean([r.inference_time_ms for r in yolo_results])
        print(f"{passed}/10    {avg_time:.0f}ms ", end="")

    if gdino_results:
        passed = sum(1 for r in gdino_results if r.detected)
        avg_time = np.mean([r.inference_time_ms for r in gdino_results])
        print(f"{passed}/10    {avg_time:.0f}ms ", end="")

    print()

    # Final comparison
    if yolo_results and gdino_results:
        yolo_pass = sum(1 for r in yolo_results if r.detected)
        gdino_pass = sum(1 for r in gdino_results if r.detected)
        yolo_time = np.mean([r.inference_time_ms for r in yolo_results])
        gdino_time = np.mean([r.inference_time_ms for r in gdino_results])

        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"  YOLO V4:        {yolo_pass}/10 tasks ({yolo_pass*10}%), avg {yolo_time:.0f}ms")
        print(f"  Grounding-DINO: {gdino_pass}/10 tasks ({gdino_pass*10}%), avg {gdino_time:.0f}ms")
        print(f"  Improvement:    +{gdino_pass - yolo_pass} tasks, +{gdino_time - yolo_time:.0f}ms latency")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO vs Grounding-DINO")
    parser.add_argument("--detector", choices=["yolo", "gdino", "both"], default="both",
                        help="Which detector to evaluate")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per task")
    parser.add_argument("--steps", type=int, default=10,
                        help="Steps per episode")
    args = parser.parse_args()

    yolo_results = None
    gdino_results = None

    if args.detector in ["yolo", "both"]:
        yolo_results = run_evaluation("yolo", args.episodes, args.steps)

    if args.detector in ["gdino", "both"]:
        gdino_results = run_evaluation("gdino", args.episodes, args.steps)

    print_summary(yolo_results, gdino_results)


if __name__ == "__main__":
    main()
