#!/usr/bin/env python3
"""Visualize YOLO detection failures to diagnose bowl detection issues.

Saves side-by-side images showing:
1. RGB image with YOLO detections (red boxes)
2. Same image with oracle bowl locations (green boxes)

This helps diagnose:
- Are bowls visible but not detected?
- Are bowls too small?
- Is there a confidence threshold issue?

Usage:
    python scripts/visualize_detection_failures.py --task-id 0 --n-episodes 3
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


def project_3d_to_2d(pos_3d, camera_params):
    """Project 3D world position to 2D pixel coordinates."""
    # Transform to camera frame
    R = camera_params['rotation_matrix']
    t = camera_params['position']

    # World to camera: cam = R^T @ (world - t)
    pos_cam = R.T @ (pos_3d - t)

    # Camera to pixel
    f = camera_params['focal_length']
    cx = camera_params['cx']
    cy = camera_params['cy']

    # OpenGL convention: -Z forward
    if pos_cam[2] >= 0:
        return None  # Behind camera

    x = -f * pos_cam[0] / pos_cam[2] + cx
    y = -f * pos_cam[1] / pos_cam[2] + cy

    return (int(x), int(y))


def main():
    parser = argparse.ArgumentParser(description="Visualize Detection Failures")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/detection_viz")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--show-all-confs", action="store_true",
                        help="Also show detections below threshold")
    args = parser.parse_args()

    import cv2
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
    print("DETECTION FAILURE VISUALIZATION")
    print("=" * 70)
    print(f"Task: {task_description}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print()

    # Setup perception
    oracle = OraclePerception()
    learned = LearnedPerception(
        model_path="models/yolo_libero.pt",
        confidence_threshold=0.1 if args.show_all_confs else args.conf_threshold,
        image_size=(256, 256),
    )
    learned.detector.warmup()

    # Output directory
    output_dir = Path(args.output_dir) / f"task{args.task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track bowl detection stats
    total_bowls_oracle = 0
    total_bowls_detected = 0
    bowl_confidences = []

    for episode_idx in range(args.n_episodes):
        episode_seed = get_episode_seed(args.seed, episode_idx)
        set_global_seed(episode_seed, env)
        env.reset()

        print(f"\n--- Episode {episode_idx + 1} (seed={episode_seed}) ---")

        # Get oracle perception
        oracle_result = oracle.perceive(env)

        # Get camera params for projection
        learned._update_camera_params(env)
        sim = learned._get_sim(env)

        # Render image
        rgb, depth = learned._render_images(sim)

        # Run YOLO with low threshold to see all detections
        learned.detector.set_confidence_threshold(0.1)  # Temporarily lower threshold
        detections_all = learned.detector.detect(rgb)
        learned.detector.set_confidence_threshold(args.conf_threshold)  # Reset

        # Filter detections at specified threshold
        detections_thresh = [d for d in detections_all if d.confidence >= args.conf_threshold]

        # Create visualization
        vis_img = rgb.copy()

        # Draw all detections (below threshold in yellow, above in red)
        for det in detections_all:
            if det.bbox is None:
                continue
            x1, y1, x2, y2 = [int(c) for c in det.bbox]

            if det.confidence >= args.conf_threshold:
                color = (255, 0, 0)  # Red for above threshold
                thickness = 2
            else:
                color = (255, 255, 0)  # Yellow for below threshold
                thickness = 1

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Project oracle bowl positions to 2D and draw green boxes
        camera_params = {
            'position': learned.camera_params.position,
            'rotation_matrix': learned.camera_params.rotation_matrix,
            'focal_length': learned.camera_params.focal_length,
            'cx': learned.camera_params.cx,
            'cy': learned.camera_params.cy,
        }

        oracle_bowls = []
        for obj_id, pos in oracle_result.objects.items():
            if 'bowl' in obj_id.lower():
                oracle_bowls.append((obj_id, pos[:3]))
                total_bowls_oracle += 1

                # Project to 2D
                pos_2d = project_3d_to_2d(pos[:3], camera_params)
                if pos_2d:
                    # Draw green circle at oracle bowl location
                    cv2.circle(vis_img, pos_2d, 15, (0, 255, 0), 2)
                    cv2.putText(vis_img, "BOWL (oracle)", (pos_2d[0] - 40, pos_2d[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Check if YOLO detected any bowls
        detected_bowls = [d for d in detections_all if 'bowl' in d.class_name.lower()]
        detected_bowls_thresh = [d for d in detected_bowls if d.confidence >= args.conf_threshold]

        print(f"  Oracle bowls: {len(oracle_bowls)}")
        for obj_id, pos in oracle_bowls:
            print(f"    {obj_id}: {pos}")

        print(f"  YOLO bowl detections (all): {len(detected_bowls)}")
        for det in detected_bowls:
            print(f"    conf={det.confidence:.3f}, bbox={det.bbox}")
            bowl_confidences.append(det.confidence)
            if det.confidence >= args.conf_threshold:
                total_bowls_detected += 1

        print(f"  YOLO bowl detections (≥{args.conf_threshold}): {len(detected_bowls_thresh)}")

        # Check what was detected instead
        other_detections = [d for d in detections_thresh if 'bowl' not in d.class_name.lower()]
        if other_detections:
            print(f"  Other detections (≥{args.conf_threshold}):")
            for det in other_detections:
                print(f"    {det.class_name}: conf={det.confidence:.3f}")

        # Diagnosis
        if len(oracle_bowls) > 0 and len(detected_bowls_thresh) == 0:
            if len(detected_bowls) > 0:
                max_conf = max(d.confidence for d in detected_bowls)
                print(f"  ⚠ DIAGNOSIS: Bowl detected but below threshold (max conf: {max_conf:.3f})")
            else:
                print(f"  ⚠ DIAGNOSIS: Bowl completely missed by YOLO")

        # Save visualization
        output_file = output_dir / f"ep{episode_idx}_seed{episode_seed}.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {output_file}")

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("BOWL DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total oracle bowls: {total_bowls_oracle}")
    print(f"Total detected (≥{args.conf_threshold}): {total_bowls_detected}")
    recall = total_bowls_detected / total_bowls_oracle if total_bowls_oracle > 0 else 0
    print(f"Bowl recall: {recall:.1%}")

    if bowl_confidences:
        print(f"\nBowl confidence distribution:")
        print(f"  Min: {min(bowl_confidences):.3f}")
        print(f"  Max: {max(bowl_confidences):.3f}")
        print(f"  Mean: {np.mean(bowl_confidences):.3f}")

        # Histogram
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(bowl_confidences, bins=bins)
        print(f"\n  Confidence histogram:")
        for i in range(len(bins) - 1):
            bar = "█" * hist[i]
            print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:2d} {bar}")
    else:
        print("\n⚠ NO BOWL DETECTIONS AT ANY CONFIDENCE LEVEL")
        print("   This suggests the model never learned to detect bowls properly.")

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
