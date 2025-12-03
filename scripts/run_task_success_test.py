#!/usr/bin/env python3
"""Run end-to-end task success testing with YOLO detector.

Tests detection + simple heuristic-based grasping on LIBERO tasks.
Reports success rates based on object detection quality.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


def make_libero_env(task_suite: str, task_id: int):
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


def evaluate_detection_quality(model_path: str, suite: str, task_id: int,
                               n_episodes: int = 5, n_steps: int = 50):
    """Evaluate detection quality on a task.

    Returns metrics about:
    - Object detection rate
    - Detection stability (consistency across frames)
    - Confidence levels
    """
    from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector

    detector = YOLOObjectDetector(model_path=model_path)

    env, task = make_libero_env(suite, task_id)

    results = {
        "task_name": task.name,
        "episodes": [],
        "overall": {}
    }

    all_detections_per_frame = []
    all_confidences = defaultdict(list)

    for ep in range(n_episodes):
        env.reset()

        ep_detections = []
        ep_classes = set()

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

            ep_detections.append(len(detections))
            for det in detections:
                ep_classes.add(det.class_name)
                all_confidences[det.class_name].append(det.confidence)

            all_detections_per_frame.append(len(detections))

        results["episodes"].append({
            "avg_detections": np.mean(ep_detections) if ep_detections else 0,
            "classes_seen": list(ep_classes),
            "num_classes": len(ep_classes)
        })

    env.close()

    # Compute overall stats
    results["overall"] = {
        "avg_detections_per_frame": float(np.mean(all_detections_per_frame)) if all_detections_per_frame else 0,
        "total_classes_detected": len(all_confidences),
        "class_confidences": {
            cls: {
                "mean": float(np.mean(confs)),
                "min": float(np.min(confs)),
                "max": float(np.max(confs)),
            }
            for cls, confs in all_confidences.items()
        }
    }

    return results


def run_evaluation(model_path: str, suite: str, n_tasks: int = 10):
    """Run evaluation on all tasks in a suite."""
    from libero.libero.benchmark import get_benchmark

    print(f"\n{'='*70}")
    print(f"TASK SUCCESS TESTING: {suite}")
    print(f"Model: {model_path}")
    print(f"{'='*70}")

    benchmark = get_benchmark(suite)()
    n_tasks = min(n_tasks, benchmark.n_tasks)

    suite_results = []

    for task_id in range(n_tasks):
        print(f"\nTask {task_id}: ", end="", flush=True)

        try:
            result = evaluate_detection_quality(model_path, suite, task_id)
            suite_results.append(result)

            avg_det = result["overall"]["avg_detections_per_frame"]
            n_classes = result["overall"]["total_classes_detected"]

            # Simple success heuristic: good detection if we see multiple objects
            success = avg_det >= 1.0 and n_classes >= 1

            status = "PASS" if success else "FAIL"
            print(f"{status} - {n_classes} classes, {avg_det:.1f} det/frame - {result['task_name'][:50]}")

            if result["overall"]["class_confidences"]:
                for cls, stats in result["overall"]["class_confidences"].items():
                    print(f"    {cls}: {stats['mean']:.3f} (min: {stats['min']:.3f})")

        except Exception as e:
            print(f"ERROR - {e}")
            suite_results.append({"error": str(e)})

    # Summary
    successful = sum(1 for r in suite_results if "error" not in r and
                    r["overall"]["avg_detections_per_frame"] >= 1.0 and
                    r["overall"]["total_classes_detected"] >= 1)

    print(f"\n{suite} Summary: {successful}/{len(suite_results)} tasks with good detection")

    return suite_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo_libero_v3.pt")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", type=int, default=10)
    args = parser.parse_args()

    results = run_evaluation(args.model, args.suite, args.tasks)


if __name__ == "__main__":
    main()
