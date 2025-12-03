#!/usr/bin/env python3
"""Validate YOLO model on live LIBERO tasks with actual manipulation pipeline."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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


def validate_model(model_path: str, output_dir: str = "validation_results"):
    """Run validation on key LIBERO tasks."""
    from brain_robot.perception.detection.yolo_detector import YOLOObjectDetector

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load model via detector
    print(f"Loading model: {model_path}")
    detector = YOLOObjectDetector(model_path=model_path)
    detector.warmup()

    # Test tasks - focus on ones with bowls and new objects
    test_cases = [
        ("libero_spatial", 0, "bowl task"),
        ("libero_spatial", 1, "bowl task 2"),
        ("libero_object", 0, "object manipulation"),
        ("libero_object", 2, "object manipulation 2"),
        ("libero_goal", 0, "goal task"),
        ("libero_10", 0, "libero_10 task"),
        ("libero_10", 2, "libero_10 task 2"),
    ]

    results = {}

    for suite, task_id, desc in test_cases:
        print(f"\n--- {suite} task {task_id}: {desc} ---")

        try:
            env, task = make_libero_env(suite, task_id)
            env.reset()

            # Take a few random actions to vary the scene
            for _ in range(5):
                action = np.random.uniform(-0.2, 0.2, 7)
                action[6] = np.random.choice([-1, 1])
                env.step(action)

            # Get image
            img = env.sim.render(
                camera_name="agentview",
                height=256,
                width=256,
                mode='offscreen'
            )
            img = img[::-1]  # Flip vertically

            # Run detection
            detections = detector.detect(img)

            # Draw results
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            for det in detections:
                x1, y1, x2, y2 = [int(c) for c in det.bbox]

                # Color by confidence
                if det.confidence > 0.5:
                    color = (0, 255, 0)  # Green
                elif det.confidence > 0.3:
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (255, 128, 0)  # Orange

                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"{det.class_name}: {det.confidence:.2f}"
                draw.text((x1, max(0, y1-12)), label, fill=color)

            # Save image
            save_path = output_path / f"{suite}_{task_id}.png"
            pil_img.save(save_path)

            # Collect stats
            task_key = f"{suite}_{task_id}"
            results[task_key] = {
                "description": desc,
                "num_detections": len(detections),
                "classes": {},
            }

            for det in detections:
                cls = det.class_name
                if cls not in results[task_key]["classes"]:
                    results[task_key]["classes"][cls] = []
                results[task_key]["classes"][cls].append(det.confidence)

            # Print summary
            print(f"  Detections: {len(detections)}")
            for cls, confs in results[task_key]["classes"].items():
                avg_conf = np.mean(confs)
                print(f"    {cls}: {len(confs)}x, avg conf: {avg_conf:.3f}")

            env.close()

        except Exception as e:
            print(f"  [ERROR] {e}")
            results[f"{suite}_{task_id}"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Check bowl detection specifically
    bowl_detections = []
    for task_key, data in results.items():
        if "classes" in data and "bowl" in data["classes"]:
            bowl_detections.extend(data["classes"]["bowl"])

    if bowl_detections:
        print(f"\nBowl Detection:")
        print(f"  Total detections: {len(bowl_detections)}")
        print(f"  Avg confidence: {np.mean(bowl_detections):.3f}")
        print(f"  Min confidence: {np.min(bowl_detections):.3f}")
        print(f"  Max confidence: {np.max(bowl_detections):.3f}")
    else:
        print("\nNo bowl detections found in test tasks")

    print(f"\nVisualization images saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo_libero_v2.pt")
    parser.add_argument("--output", default="validation_results")
    args = parser.parse_args()

    validate_model(args.model, args.output)
