#!/usr/bin/env python3
"""Debug cold bootstrap to understand why it picks wrong objects.

This script runs cold bootstrap and logs detailed information about:
1. What YOLO detects (classes, confidences)
2. What IDs are assigned
3. What object is selected as source
4. Why the wrong object might be chosen

Usage:
    python scripts/debug_cold_bootstrap.py --task-id 0
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Debug Cold Bootstrap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import after path setup
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from brain_robot.perception.oracle import OraclePerception
    from brain_robot.perception.learned import LearnedPerception
    from brain_robot.utils.seeds import set_global_seed, get_episode_seed

    # Setup environment
    benchmark = get_benchmark("libero_spatial")()
    task = benchmark.get_task(args.task_id)
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
    task_description = task.language

    print("=" * 70)
    print("COLD BOOTSTRAP DEBUG")
    print("=" * 70)
    print(f"Task: {task_description}")
    print(f"Task ID: {args.task_id}")
    print()

    # Extract expected classes from task
    task_lower = task_description.lower()
    expected_source = None
    expected_target = None

    # Simple parsing
    if "bowl" in task_lower:
        expected_source = "bowl"
    if "on the plate" in task_lower:
        expected_target = "plate"

    print(f"Expected source class: {expected_source}")
    print(f"Expected target class: {expected_target}")
    print()

    # Setup perception
    oracle = OraclePerception()
    learned = LearnedPerception(
        model_path="models/yolo_libero.pt",
        confidence_threshold=0.5,
        image_size=(256, 256),
    )

    print("Warming up YOLO detector...")
    learned.detector.warmup()
    print()

    debug_results = []

    for episode_idx in range(args.n_episodes):
        episode_seed = get_episode_seed(args.seed, episode_idx)
        set_global_seed(episode_seed, env)
        env.reset()

        print(f"\n{'='*70}")
        print(f"EPISODE {episode_idx + 1} (seed={episode_seed})")
        print("=" * 70)

        # Get oracle perception for ground truth
        oracle_result = oracle.perceive(env)
        print("\n--- ORACLE (Ground Truth) ---")
        for obj_id, pos in oracle_result.objects.items():
            print(f"  {obj_id}: pos={pos[:3]}")

        # Reset learned perception and bootstrap from detections
        learned.reset()

        # Manually do cold bootstrap with detailed logging
        learned._update_camera_params(env)
        sim = learned._get_sim(env)
        rgb, depth = learned._render_images(sim)

        print("\n--- YOLO DETECTIONS ---")
        detections_2d = learned.detector.detect(rgb)
        for det in detections_2d:
            print(f"  {det.class_name}: conf={det.confidence:.3f}, bbox={det.bbox}")

        print("\n--- 3D DETECTIONS (after depth projection) ---")
        detections_3d = learned._detections_to_3d(detections_2d, depth)
        for det in detections_3d:
            print(f"  {det.class_name}: conf={det.confidence:.3f}, pos={det.position}")

        print("\n--- COLD BOOTSTRAP ID ASSIGNMENT ---")
        class_counts = {}
        known_objects = {}
        for det in detections_3d:
            cls = det.class_name
            idx = class_counts.get(cls, 0)
            class_counts[cls] = idx + 1
            instance_id = f"{cls}_{idx}_learned"
            known_objects[instance_id] = det.position
            print(f"  Assigned: {instance_id} at pos={det.position}")

        # Initialize tracker
        learned.tracker.initialize(known_objects)
        learned._bootstrapped = True

        # Get learned perception result
        learned_result = learned.perceive(env)
        print("\n--- LEARNED PERCEPTION RESULT ---")
        for obj_id, pos in learned_result.objects.items():
            print(f"  {obj_id}: pos={pos[:3]}")

        # Analyze what would be selected
        print("\n--- OBJECT SELECTION ANALYSIS ---")
        print(f"  Available objects: {list(learned_result.object_names)}")

        # Check which objects match expected classes
        bowl_candidates = [o for o in learned_result.object_names if "bowl" in o.lower()]
        plate_candidates = [o for o in learned_result.object_names if "plate" in o.lower()]

        print(f"  Bowl candidates: {bowl_candidates}")
        print(f"  Plate candidates: {plate_candidates}")

        if not bowl_candidates:
            print("\n  ⚠ NO BOWL DETECTED - this is a DETECTION failure")
        else:
            print(f"\n  ✓ Bowl detected: {bowl_candidates}")

        # What would parse_task_for_grounding select?
        from scripts.run_evaluation import parse_task_for_grounding
        source_obj, target_obj = parse_task_for_grounding(task_description, learned_result.object_names)

        print(f"\n  Selected source: {source_obj}")
        print(f"  Selected target: {target_obj}")

        # Check semantic correctness
        def get_class(obj_id):
            obj_lower = obj_id.lower()
            for cls in ['bowl', 'plate', 'stove', 'ramekin', 'mug']:
                if cls in obj_lower:
                    return cls
            return "unknown"

        source_class = get_class(source_obj) if source_obj else "none"
        target_class = get_class(target_obj) if target_obj else "none"

        source_correct = source_class == expected_source
        target_correct = target_class == expected_target

        print(f"\n  Source class: {source_class} (expected: {expected_source}) {'✓' if source_correct else '✗'}")
        print(f"  Target class: {target_class} (expected: {expected_target}) {'✓' if target_correct else '✗'}")

        # Diagnose failure
        if not source_correct:
            print("\n  ⚠ DIAGNOSIS:")
            if not bowl_candidates:
                print("    → DETECTION FAILURE: YOLO did not detect any bowl")
            elif source_obj not in bowl_candidates:
                print("    → GROUNDING/PARSING FAILURE: Bowl was detected but wrong object selected")
                print(f"      Detected bowls: {bowl_candidates}")
                print(f"      Selected source: {source_obj}")

        debug_results.append({
            "episode": episode_idx,
            "seed": episode_seed,
            "oracle_objects": list(oracle_result.object_names),
            "yolo_detections": [{"class": d.class_name, "conf": d.confidence} for d in detections_2d],
            "learned_objects": list(learned_result.object_names),
            "bowl_candidates": bowl_candidates,
            "selected_source": source_obj,
            "selected_target": target_obj,
            "source_class": source_class,
            "expected_source": expected_source,
            "source_correct": source_correct,
        })

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_episodes = len(debug_results)
    n_source_correct = sum(1 for r in debug_results if r["source_correct"])
    n_detection_failures = sum(1 for r in debug_results if not r["bowl_candidates"])

    print(f"Episodes: {n_episodes}")
    print(f"Semantic source correct: {n_source_correct}/{n_episodes}")
    print(f"Detection failures (no bowl): {n_detection_failures}/{n_episodes}")
    print(f"Grounding/parsing failures: {n_episodes - n_source_correct - n_detection_failures}/{n_episodes}")

    # Save detailed results
    output_file = f"logs/cold_bootstrap_debug_task{args.task_id}.json"
    os.makedirs("logs", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(debug_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
