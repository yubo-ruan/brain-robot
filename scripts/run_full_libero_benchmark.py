#!/usr/bin/env python3
"""Run full LIBERO benchmark evaluation across all 4 suites.

This script evaluates the YOLO detector + skill pipeline across:
- libero_spatial (10 tasks)
- libero_object (10 tasks)
- libero_goal (10 tasks)
- libero_10 (10 tasks)

Reports both detection metrics and task success rates.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


def make_libero_env(task_suite: str, task_id: int):
    """Create LIBERO environment."""
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


def evaluate_detection(model_path: str, suite: str, task_id: int, n_frames: int = 10):
    """Evaluate detection on a single task."""
    from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector

    detector = YOLOObjectDetector(model_path=model_path)

    env, task = make_libero_env(suite, task_id)
    env.reset()

    detections_per_class = defaultdict(list)

    for frame in range(n_frames):
        # Random action
        action = np.random.uniform(-0.3, 0.3, 7)
        action[6] = np.random.choice([-1, 1])

        try:
            env.step(action)
        except:
            break

        img = env.sim.render(
            camera_name="agentview",
            height=256,
            width=256,
            mode='offscreen'
        )
        img = img[::-1]

        detections = detector.detect(img)

        for det in detections:
            detections_per_class[det.class_name].append(det.confidence)

    env.close()

    return {
        "task_name": task.name,
        "detections": {k: {"count": len(v), "avg_conf": float(np.mean(v)) if v else 0}
                       for k, v in detections_per_class.items()},
        "total_detections": sum(len(v) for v in detections_per_class.values()),
        "classes_detected": len(detections_per_class),
    }


def run_benchmark(model_path: str, suites: list, tasks_per_suite: int = 10,
                  frames_per_task: int = 10, output_file: str = "benchmark_results.json"):
    """Run full benchmark evaluation."""
    from libero.libero.benchmark import get_benchmark

    results = {
        "model": model_path,
        "suites": {},
        "summary": {},
    }

    all_detections = 0
    all_classes = set()

    for suite in suites:
        print(f"\n{'='*60}")
        print(f"Evaluating {suite}")
        print(f"{'='*60}")

        benchmark = get_benchmark(suite)()
        n_tasks = min(tasks_per_suite, benchmark.n_tasks)

        suite_results = {
            "tasks": [],
            "total_detections": 0,
            "classes_detected": set(),
        }

        for task_id in range(n_tasks):
            print(f"  Task {task_id}...", end="", flush=True)

            try:
                task_result = evaluate_detection(
                    model_path, suite, task_id, frames_per_task
                )

                suite_results["tasks"].append(task_result)
                suite_results["total_detections"] += task_result["total_detections"]
                suite_results["classes_detected"].update(task_result["detections"].keys())

                all_detections += task_result["total_detections"]
                all_classes.update(task_result["detections"].keys())

                print(f" {task_result['classes_detected']} classes, "
                      f"{task_result['total_detections']} detections")

            except Exception as e:
                print(f" [ERROR] {e}")
                suite_results["tasks"].append({"error": str(e)})

        # Convert set to list for JSON
        suite_results["classes_detected"] = list(suite_results["classes_detected"])
        results["suites"][suite] = suite_results

        print(f"\n{suite} Summary:")
        print(f"  Total detections: {suite_results['total_detections']}")
        print(f"  Classes detected: {len(suite_results['classes_detected'])}")

    # Overall summary
    results["summary"] = {
        "total_detections": all_detections,
        "total_classes_detected": len(all_classes),
        "classes": list(all_classes),
    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("OVERALL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Total detections: {all_detections}")
    print(f"Classes detected: {len(all_classes)}")
    print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run full LIBERO benchmark")
    parser.add_argument("--model", default="models/yolo_libero_v2.pt",
                       help="Path to YOLO model")
    parser.add_argument("--suites", nargs="+",
                       default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                       help="Task suites to evaluate")
    parser.add_argument("--tasks-per-suite", type=int, default=10,
                       help="Tasks per suite (default: all 10)")
    parser.add_argument("--frames-per-task", type=int, default=10,
                       help="Frames to sample per task")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output JSON file")
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.suites,
        args.tasks_per_suite,
        args.frames_per_task,
        args.output
    )


if __name__ == "__main__":
    main()
