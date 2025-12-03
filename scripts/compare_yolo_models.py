#!/usr/bin/env python3
"""Compare two YOLO models on LIBERO detection tasks."""

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


def evaluate_model(model_path: str, suite: str, task_id: int, n_frames: int = 10):
    """Evaluate a single model on a task."""
    from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector

    detector = YOLOObjectDetector(model_path=model_path)

    env, task = make_libero_env(suite, task_id)
    env.reset()

    detections_per_class = defaultdict(list)

    for frame in range(n_frames):
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
        "high_conf_detections": sum(1 for v in detections_per_class.values() for c in v if c > 0.5),
    }


def compare_models(model_a: str, model_b: str, suites: list, tasks_per_suite: int = 10,
                   frames_per_task: int = 10, output_file: str = None):
    """Compare two YOLO models across suites."""
    from libero.libero.benchmark import get_benchmark

    results = {
        "model_a": model_a,
        "model_b": model_b,
        "suites": {},
        "comparison": {},
    }

    model_a_total = {"detections": 0, "high_conf": 0, "classes": set()}
    model_b_total = {"detections": 0, "high_conf": 0, "classes": set()}

    for suite in suites:
        print(f"\n{'='*60}")
        print(f"Evaluating {suite}")
        print(f"{'='*60}")

        benchmark = get_benchmark(suite)()
        n_tasks = min(tasks_per_suite, benchmark.n_tasks)

        suite_results = {"tasks": []}

        for task_id in range(n_tasks):
            print(f"\n  Task {task_id}:")

            # Evaluate model A
            print(f"    Model A ({Path(model_a).stem})...", end="", flush=True)
            result_a = evaluate_model(model_a, suite, task_id, frames_per_task)
            print(f" {result_a['total_detections']} det, {result_a['high_conf_detections']} high-conf")

            # Evaluate model B
            print(f"    Model B ({Path(model_b).stem})...", end="", flush=True)
            result_b = evaluate_model(model_b, suite, task_id, frames_per_task)
            print(f" {result_b['total_detections']} det, {result_b['high_conf_detections']} high-conf")

            suite_results["tasks"].append({
                "task_name": result_a["task_name"],
                "model_a": result_a,
                "model_b": result_b,
            })

            model_a_total["detections"] += result_a["total_detections"]
            model_a_total["high_conf"] += result_a["high_conf_detections"]
            model_a_total["classes"].update(result_a["detections"].keys())

            model_b_total["detections"] += result_b["total_detections"]
            model_b_total["high_conf"] += result_b["high_conf_detections"]
            model_b_total["classes"].update(result_b["detections"].keys())

        results["suites"][suite] = suite_results

    # Summary
    results["summary"] = {
        "model_a": {
            "name": Path(model_a).stem,
            "total_detections": model_a_total["detections"],
            "high_conf_detections": model_a_total["high_conf"],
            "classes_detected": len(model_a_total["classes"]),
        },
        "model_b": {
            "name": Path(model_b).stem,
            "total_detections": model_b_total["detections"],
            "high_conf_detections": model_b_total["high_conf"],
            "classes_detected": len(model_b_total["classes"]),
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{Path(model_a).stem} vs {Path(model_b).stem}")
    print(f"\n{'Metric':<25} {'Model A':<15} {'Model B':<15} {'Diff':<15}")
    print("-" * 70)

    a_det = model_a_total["detections"]
    b_det = model_b_total["detections"]
    diff_det = b_det - a_det
    print(f"{'Total Detections':<25} {a_det:<15} {b_det:<15} {diff_det:+d}")

    a_high = model_a_total["high_conf"]
    b_high = model_b_total["high_conf"]
    diff_high = b_high - a_high
    print(f"{'High-Conf (>0.5)':<25} {a_high:<15} {b_high:<15} {diff_high:+d}")

    a_cls = len(model_a_total["classes"])
    b_cls = len(model_b_total["classes"])
    diff_cls = b_cls - a_cls
    print(f"{'Classes Detected':<25} {a_cls:<15} {b_cls:<15} {diff_cls:+d}")

    if a_det > 0:
        a_rate = 100 * a_high / a_det
        b_rate = 100 * b_high / b_det if b_det > 0 else 0
        print(f"{'High-Conf Rate':<25} {a_rate:.1f}%{'':<10} {b_rate:.1f}%{'':<10} {b_rate - a_rate:+.1f}%")

    # Winner determination
    print(f"\n{'='*60}")
    if b_high > a_high:
        print(f"✓ {Path(model_b).stem} has MORE high-confidence detections (+{diff_high})")
    elif a_high > b_high:
        print(f"✓ {Path(model_a).stem} has MORE high-confidence detections (+{-diff_high})")
    else:
        print("= Both models have equal high-confidence detections")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two YOLO models")
    parser.add_argument("--model-a", default="models/yolo_libero_v4.pt",
                       help="First model path")
    parser.add_argument("--model-b", default="models/yolo_libero_v5.pt",
                       help="Second model path")
    parser.add_argument("--suites", nargs="+",
                       default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                       help="Task suites to evaluate")
    parser.add_argument("--tasks-per-suite", type=int, default=10,
                       help="Tasks per suite")
    parser.add_argument("--frames-per-task", type=int, default=10,
                       help="Frames to sample per task")
    parser.add_argument("--output", default=None,
                       help="Output JSON file")
    args = parser.parse_args()

    compare_models(
        args.model_a,
        args.model_b,
        args.suites,
        args.tasks_per_suite,
        args.frames_per_task,
        args.output
    )


if __name__ == "__main__":
    main()
