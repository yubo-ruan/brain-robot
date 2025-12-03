#!/usr/bin/env python3
"""Test perception accuracy: YOLO + depth estimation vs oracle positions.

Measures how well the learned perception (YOLO detection + depth-based 3D estimation)
matches ground truth positions from the simulator.

Metrics:
- Detection rate: % of objects detected by YOLO
- Position error: 3D Euclidean distance from oracle
- XY error: Horizontal distance (most important for grasping)
- Z error: Vertical distance
"""

import argparse
import torch
import numpy as np
from collections import defaultdict

# Patch torch.load for LIBERO compatibility
_original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = patched_load

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

import sys
sys.path.insert(0, '/workspace/brain_robot')

from brain_robot.perception.perception_module import PerceptionModule
from brain_robot.perception.depth_estimation import (
    CameraInfo,
    normalize_depth_to_meters,
    estimate_object_position_from_bbox,
)


def get_oracle_positions(env):
    """Get ground truth positions for all objects from simulator."""
    positions = {}
    sim = env.sim

    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        if body_name is None:
            continue

        # Check for common object patterns
        name_lower = body_name.lower()
        is_object = any(obj in name_lower for obj in [
            'bowl', 'plate', 'mug', 'cup', 'ramekin', 'cookie',
            'cream_cheese', 'butter', 'milk', 'ketchup', 'can', 'bottle'
        ])

        if is_object and '_main' in body_name:
            positions[body_name] = sim.data.body_xpos[i].copy()

    return positions


def test_single_frame(env, obs, perception, oracle_positions):
    """Test perception accuracy on a single frame."""
    results = {
        'oracle_count': len(oracle_positions),
        'detected_count': 0,
        'matched_count': 0,
        'errors': [],
        'detections': [],
    }

    # Get RGB and depth
    rgb = obs.get('agentview_image')
    depth = obs.get('agentview_depth')

    if rgb is None or depth is None:
        results['error'] = 'Missing RGB or depth'
        return results

    # Run YOLO detection
    detections = perception.detector.detect(rgb)
    results['detected_count'] = len(detections)

    # Estimate 3D positions
    detections = perception.depth_estimator.estimate_positions(
        detections, depth, env.sim
    )

    # Match detections to oracle positions
    for det in detections:
        det_info = {
            'class': det.class_name,
            'confidence': det.confidence,
            'position': det.position.tolist() if det.position is not None else None,
            'bbox': det.bbox,
        }

        # Find closest oracle object of same class
        best_match = None
        best_dist = float('inf')

        for oracle_name, oracle_pos in oracle_positions.items():
            # Check class match
            if det.class_name not in oracle_name.lower():
                continue

            dist = np.linalg.norm(det.position - oracle_pos)
            if dist < best_dist:
                best_dist = dist
                best_match = (oracle_name, oracle_pos)

        if best_match is not None:
            oracle_name, oracle_pos = best_match
            xy_error = np.linalg.norm(det.position[:2] - oracle_pos[:2])
            z_error = abs(det.position[2] - oracle_pos[2])

            det_info['matched_to'] = oracle_name
            det_info['oracle_pos'] = oracle_pos.tolist()
            det_info['error_3d'] = float(best_dist)
            det_info['error_xy'] = float(xy_error)
            det_info['error_z'] = float(z_error)

            results['errors'].append({
                'class': det.class_name,
                'error_3d': best_dist,
                'error_xy': xy_error,
                'error_z': z_error,
            })
            results['matched_count'] += 1

        results['detections'].append(det_info)

    return results


def run_perception_test(
    task_suite: str = 'libero_spatial',
    task_ids: list = None,
    n_frames: int = 5,
    verbose: bool = True,
):
    """Run perception accuracy test across multiple tasks and frames."""

    if task_ids is None:
        task_ids = [0, 1, 2, 5]  # Tasks with known good performance

    # Initialize perception module
    perception = PerceptionModule.from_config(
        model_path='models/yolo_libero_v4.pt',
        use_oracle=False,
    )

    # Aggregate results
    all_results = defaultdict(list)

    benchmark = get_benchmark(task_suite)()

    for task_id in task_ids:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task {task_id}: {benchmark.get_task(task_id).language}")
            print('='*60)

        # Setup environment
        task = benchmark.get_task(task_id)
        bddl_file = benchmark.get_task_bddl_file_path(task_id)
        init_states = benchmark.get_task_init_states(task_id)

        env_args = {
            'bddl_file_name': bddl_file,
            'camera_heights': 128,
            'camera_widths': 128,
            'camera_depths': True,
        }

        env = OffScreenRenderEnv(**env_args)
        env.reset()
        env.set_init_state(init_states[0])

        # Initialize perception with environment
        perception.depth_estimator.update_camera_info(env.sim)

        # Run multiple frames
        for frame in range(n_frames):
            # Take random action to vary scene slightly
            action = np.random.uniform(-0.1, 0.1, 7)
            action[6] = 0  # Don't change gripper
            obs, _, _, _ = env.step(action)

            # Get oracle positions
            oracle_positions = get_oracle_positions(env)

            # Test perception
            results = test_single_frame(env, obs, perception, oracle_positions)

            if verbose and frame == 0:
                print(f"\nOracle objects: {list(oracle_positions.keys())}")
                print(f"Detections: {results['detected_count']}")
                print(f"Matched: {results['matched_count']}")

                for det in results['detections']:
                    if 'error_3d' in det:
                        print(f"  {det['class']}: "
                              f"3D={det['error_3d']*100:.1f}cm, "
                              f"XY={det['error_xy']*100:.1f}cm, "
                              f"Z={det['error_z']*100:.1f}cm "
                              f"(conf={det['confidence']:.2f})")
                    else:
                        print(f"  {det['class']}: NO MATCH (conf={det['confidence']:.2f})")

            # Aggregate
            all_results['task_id'].append(task_id)
            all_results['frame'].append(frame)
            all_results['oracle_count'].append(results['oracle_count'])
            all_results['detected_count'].append(results['detected_count'])
            all_results['matched_count'].append(results['matched_count'])

            for err in results['errors']:
                all_results['errors_3d'].append(err['error_3d'])
                all_results['errors_xy'].append(err['error_xy'])
                all_results['errors_z'].append(err['error_z'])
                all_results['error_classes'].append(err['class'])

        env.close()

    # Print summary
    print("\n" + "="*60)
    print("PERCEPTION ACCURACY SUMMARY")
    print("="*60)

    total_oracle = sum(all_results['oracle_count'])
    total_detected = sum(all_results['detected_count'])
    total_matched = sum(all_results['matched_count'])

    print(f"\nDetection Rate:")
    print(f"  Total oracle objects: {total_oracle}")
    print(f"  Total detections: {total_detected}")
    print(f"  Matched detections: {total_matched}")

    if all_results['errors_3d']:
        errors_3d = np.array(all_results['errors_3d']) * 100  # cm
        errors_xy = np.array(all_results['errors_xy']) * 100
        errors_z = np.array(all_results['errors_z']) * 100

        print(f"\nPosition Errors (cm):")
        print(f"  3D Error:  mean={errors_3d.mean():.1f}, std={errors_3d.std():.1f}, "
              f"median={np.median(errors_3d):.1f}, max={errors_3d.max():.1f}")
        print(f"  XY Error:  mean={errors_xy.mean():.1f}, std={errors_xy.std():.1f}, "
              f"median={np.median(errors_xy):.1f}, max={errors_xy.max():.1f}")
        print(f"  Z Error:   mean={errors_z.mean():.1f}, std={errors_z.std():.1f}, "
              f"median={np.median(errors_z):.1f}, max={errors_z.max():.1f}")

        # Per-class breakdown
        print(f"\nPer-Class XY Errors:")
        classes = set(all_results['error_classes'])
        for cls in sorted(classes):
            cls_errors = [
                all_results['errors_xy'][i] * 100
                for i, c in enumerate(all_results['error_classes'])
                if c == cls
            ]
            if cls_errors:
                print(f"  {cls}: mean={np.mean(cls_errors):.1f}cm, n={len(cls_errors)}")

        # Grasp success threshold analysis
        print(f"\nGrasp Success Threshold Analysis (XY):")
        for threshold in [3, 5, 7, 10]:
            pct = (errors_xy < threshold).mean() * 100
            print(f"  <{threshold}cm: {pct:.1f}%")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test perception accuracy')
    parser.add_argument('--tasks', type=int, nargs='+', default=[0, 1, 2, 5],
                        help='Task IDs to test')
    parser.add_argument('--frames', type=int, default=5,
                        help='Frames per task')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output')

    args = parser.parse_args()

    results = run_perception_test(
        task_ids=args.tasks,
        n_frames=args.frames,
        verbose=not args.quiet,
    )
