#!/usr/bin/env python3
"""Target-aware evaluation for YOLO detector.

This script evaluates whether the detector correctly identifies the TARGET object
for each task, not just any object. It tracks:
- Target detection rate (is the correct object detected?)
- Misclassification rate (wrong class detected instead of target)
- Confusion matrix for grocery items

Usage:
    python scripts/evaluate_target_detection.py --model models/yolo_libero_v4.pt --suite libero_object
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


# Known confusion groups - classes that are visually similar
CONFUSION_GROUPS = [
    {"ketchup", "tomato_sauce", "bbq_sauce"},
    {"butter", "cream_cheese", "chocolate_pudding"},
    {"milk", "orange_juice"},
    {"alphabet_soup", "tomato_sauce", "cream_cheese"},
]

# All grocery classes for confusion matrix
GROCERY_CLASSES = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice"
]


@dataclass
class TaskResult:
    """Result for a single task evaluation."""
    task_id: int
    task_name: str
    target_object: str
    target_detected: bool = False
    target_confidence: float = 0.0
    target_detection_count: int = 0
    total_frames: int = 0
    misclassifications: List[Dict] = field(default_factory=list)
    all_detections: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


def extract_target_from_task_name(task_name: str) -> Optional[str]:
    """Extract target object from task name.

    Examples:
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket" -> "bbq_sauce"
        "pick_up_the_cream_cheese_and_place_it_in_the_basket" -> "cream_cheese"
        "put_the_bowl_on_the_stove" -> "bowl"
    """
    # Common patterns
    patterns = [
        r"pick_up_the_(\w+)_and",
        r"pick_up_the_(\w+_\w+)_and",  # Two-word objects like cream_cheese
        r"pick_up_the_(\w+_\w+_\w+)_and",  # Three-word objects
        r"put_the_(\w+)_on",
        r"put_the_(\w+_\w+)_on",
        r"put_the_(\w+)_in",
        r"put_the_(\w+_\w+)_in",
        r"open_the_(\w+)",
        r"close_the_(\w+)",
        r"turn_on_the_(\w+)",
        r"turn_off_the_(\w+)",
    ]

    task_lower = task_name.lower()

    for pattern in patterns:
        match = re.search(pattern, task_lower)
        if match:
            return match.group(1)

    # Fallback: look for known object names
    known_objects = GROCERY_CLASSES + [
        "bowl", "plate", "mug", "ramekin", "cabinet", "drawer",
        "basket", "moka_pot", "book", "caddy", "microwave",
        "white_mug", "wine_bottle", "wine_rack", "frying_pan", "stove"
    ]

    for obj in known_objects:
        if obj in task_lower:
            return obj

    return None


def is_confusable_with(detected: str, expected: str) -> bool:
    """Check if detected class is a known confusion with expected."""
    for group in CONFUSION_GROUPS:
        if detected in group and expected in group and detected != expected:
            return True
    return False


def make_libero_env(task_suite: str, task_id: int):
    """Create LIBERO environment for a task."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(task_suite)()
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


def evaluate_task(
    detector,
    suite: str,
    task_id: int,
    n_episodes: int = 5,
    n_steps: int = 30
) -> TaskResult:
    """Evaluate detection on a single task."""
    env, task = make_libero_env(suite, task_id)

    target_object = extract_target_from_task_name(task.name)

    result = TaskResult(
        task_id=task_id,
        task_name=task.name,
        target_object=target_object or "unknown"
    )

    for ep in range(n_episodes):
        env.reset()

        for step in range(n_steps):
            # Random action for diversity
            action = np.random.uniform(-0.3, 0.3, 7)
            action[6] = np.random.choice([-1, 1])

            try:
                env.step(action)
            except:
                break

            # Get image
            img = env.sim.render(
                camera_name="agentview",
                height=256,
                width=256,
                mode='offscreen'
            )
            img = img[::-1]

            # Detect
            detections = detector.detect(img)
            result.total_frames += 1

            frame_has_target = False

            for det in detections:
                # Track all detections
                result.all_detections[det.class_name].append(det.confidence)

                # Check if target detected
                if target_object and det.class_name == target_object:
                    frame_has_target = True
                    result.target_confidence = max(result.target_confidence, det.confidence)

                # Check for misclassification (detected confusable class instead of target)
                elif target_object and is_confusable_with(det.class_name, target_object):
                    result.misclassifications.append({
                        "frame": result.total_frames,
                        "detected": det.class_name,
                        "expected": target_object,
                        "confidence": det.confidence
                    })

            if frame_has_target:
                result.target_detection_count += 1

    env.close()

    # Target is considered detected if seen in >10% of frames
    result.target_detected = (result.target_detection_count / max(1, result.total_frames)) > 0.1

    return result


def generate_confusion_matrix(results: List[TaskResult]) -> np.ndarray:
    """Generate confusion matrix for grocery items."""
    n_classes = len(GROCERY_CLASSES)
    confusion = np.zeros((n_classes, n_classes))

    for r in results:
        if r.target_object not in GROCERY_CLASSES:
            continue

        expected_idx = GROCERY_CLASSES.index(r.target_object)

        # Count detections for each class
        for class_name, confidences in r.all_detections.items():
            if class_name in GROCERY_CLASSES:
                detected_idx = GROCERY_CLASSES.index(class_name)
                confusion[expected_idx, detected_idx] += len(confidences)

    return confusion


def print_confusion_matrix(confusion: np.ndarray):
    """Print confusion matrix in ASCII format."""
    n = len(GROCERY_CLASSES)

    # Normalize rows to percentages
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_pct = confusion / row_sums * 100

    # Print header
    print("\nConfusion Matrix (row=expected, col=detected, values=%):")
    print("=" * 80)

    # Abbreviated class names
    abbrev = [c[:6] for c in GROCERY_CLASSES]

    # Header row
    print(f"{'':>12}", end="")
    for a in abbrev:
        print(f"{a:>7}", end="")
    print()

    # Data rows
    for i, (name, row) in enumerate(zip(GROCERY_CLASSES, confusion_pct)):
        print(f"{name[:12]:>12}", end="")
        for j, val in enumerate(row):
            if i == j:
                # Diagonal (correct)
                print(f"\033[92m{val:6.1f}%\033[0m", end="")
            elif val > 10:
                # High confusion (>10%)
                print(f"\033[91m{val:6.1f}%\033[0m", end="")
            elif val > 0:
                print(f"{val:6.1f}%", end="")
            else:
                print(f"{'':>7}", end="")
        print()


def run_evaluation(model_path: str, suite: str, n_tasks: int = 10):
    """Run full evaluation on a suite."""
    from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector

    print(f"\n{'='*70}")
    print(f"TARGET-AWARE DETECTION EVALUATION")
    print(f"Model: {model_path}")
    print(f"Suite: {suite}")
    print(f"{'='*70}")

    detector = YOLOObjectDetector(model_path=model_path)

    results = []

    for task_id in range(n_tasks):
        print(f"\nTask {task_id}: ", end="", flush=True)

        try:
            result = evaluate_task(detector, suite, task_id)
            results.append(result)

            # Determine status
            if result.target_detected:
                status = "\033[92mTARGET FOUND\033[0m"
            elif result.misclassifications:
                status = "\033[91mMISCLASSIFIED\033[0m"
            else:
                status = "\033[93mNOT DETECTED\033[0m"

            # Detection rate
            det_rate = result.target_detection_count / max(1, result.total_frames) * 100

            print(f"{status}")
            print(f"    Task: {result.task_name[:60]}")
            print(f"    Target: {result.target_object}")
            print(f"    Target detection rate: {det_rate:.1f}% ({result.target_detection_count}/{result.total_frames} frames)")
            print(f"    Target confidence: {result.target_confidence:.2f}")

            if result.misclassifications:
                # Group misclassifications by class
                misclass_counts = defaultdict(int)
                misclass_conf = defaultdict(list)
                for m in result.misclassifications:
                    misclass_counts[m["detected"]] += 1
                    misclass_conf[m["detected"]].append(m["confidence"])

                print(f"    Misclassifications:")
                for cls, count in sorted(misclass_counts.items(), key=lambda x: -x[1]):
                    avg_conf = np.mean(misclass_conf[cls])
                    print(f"      - {cls}: {count} times (avg conf: {avg_conf:.2f})")

            # Show what was detected instead
            if not result.target_detected and result.all_detections:
                print(f"    Detected instead:")
                for cls, confs in sorted(result.all_detections.items(), key=lambda x: -len(x[1]))[:5]:
                    print(f"      - {cls}: {len(confs)} times (avg conf: {np.mean(confs):.2f})")

        except Exception as e:
            print(f"ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    targets_found = sum(1 for r in results if r.target_detected)
    misclassified = sum(1 for r in results if not r.target_detected and r.misclassifications)
    not_detected = sum(1 for r in results if not r.target_detected and not r.misclassifications)

    print(f"Target correctly detected: {targets_found}/{len(results)} tasks")
    print(f"Misclassified (wrong class): {misclassified}/{len(results)} tasks")
    print(f"Not detected at all: {not_detected}/{len(results)} tasks")

    # Misclassification rate for grocery items
    grocery_results = [r for r in results if r.target_object in GROCERY_CLASSES]
    if grocery_results:
        grocery_correct = sum(1 for r in grocery_results if r.target_detected)
        grocery_misclass = sum(1 for r in grocery_results if not r.target_detected and r.misclassifications)

        print(f"\nGrocery items specifically:")
        print(f"  Correct: {grocery_correct}/{len(grocery_results)}")
        print(f"  Misclassified: {grocery_misclass}/{len(grocery_results)}")

        if len(grocery_results) > 0:
            misclass_rate = grocery_misclass / len(grocery_results) * 100
            print(f"  Misclassification rate: {misclass_rate:.1f}%")

    # Confusion matrix
    confusion = generate_confusion_matrix(results)
    if confusion.sum() > 0:
        print_confusion_matrix(confusion)

    return results


def main():
    parser = argparse.ArgumentParser(description="Target-aware detection evaluation")
    parser.add_argument("--model", default="models/yolo_libero_v4.pt", help="Model path")
    parser.add_argument("--suite", default="libero_object", help="Task suite")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task")
    parser.add_argument("--steps", type=int, default=30, help="Steps per episode")
    args = parser.parse_args()

    results = run_evaluation(args.model, args.suite, args.tasks)

    # Return exit code based on success rate
    targets_found = sum(1 for r in results if r.target_detected)
    success_rate = targets_found / len(results) if results else 0

    if success_rate < 0.5:
        sys.exit(1)  # Failure
    sys.exit(0)


if __name__ == "__main__":
    main()
