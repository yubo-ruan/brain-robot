#!/usr/bin/env python3
"""
Pre-compute VLM plans for LIBERO demonstrations.

This script:
1. Loads LIBERO demonstration HDF5 files
2. Extracts images from each timestep
3. Runs VLM on images to generate motion plans
4. Saves (state, action, vlm_plan) tuples for training
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import h5py
import json
import pickle
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
import time


def load_demo_images_and_states(
    demo_path: str,
    n_demos: int = 50,
    sample_interval: int = 5,  # Sample every N frames
) -> List[Dict]:
    """
    Load images, states, and actions from LIBERO demo file.

    Args:
        demo_path: Path to HDF5 demo file
        n_demos: Number of demonstrations to load
        sample_interval: Sample every N frames (VLM is slow)

    Returns:
        List of (image, state, action, step_idx) tuples
    """
    samples = []

    with h5py.File(demo_path, 'r') as f:
        demo_keys = sorted(
            [k for k in f['data'].keys() if k.startswith('demo_')],
            key=lambda x: int(x.split('_')[1])
        )[:n_demos]

        for dk in tqdm(demo_keys, desc="Loading demos"):
            demo = f['data'][dk]

            # Get images
            images = demo['obs/agentview_rgb'][:]  # (T, H, W, 3)

            # Get states
            ee_pos = demo['obs/ee_pos'][:]
            ee_ori = demo['obs/ee_ori'][:]
            gripper = demo['obs/gripper_states'][:]
            joints = demo['obs/joint_states'][:]
            states = np.concatenate([ee_pos, ee_ori, gripper, joints], axis=1)

            # Get actions
            actions = demo['actions'][:]

            # Sample frames
            T = len(images)
            for t in range(0, T, sample_interval):
                samples.append({
                    'image': images[t],
                    'state': states[t],
                    'action': actions[t],
                    'step_idx': t,
                    'demo_idx': dk,
                    'total_steps': T,
                })

    print(f"Loaded {len(samples)} samples from {len(demo_keys)} demos")
    return samples


def determine_phase(step_idx: int, total_steps: int, gripper_state: float) -> str:
    """
    Heuristically determine task phase based on progress.

    This is a simple heuristic - VLM should learn to do better.
    """
    progress = step_idx / total_steps

    if progress < 0.2:
        return "approach"
    elif progress < 0.3:
        return "align"
    elif progress < 0.4:
        return "descend"
    elif progress < 0.5:
        return "grasp"
    elif progress < 0.6:
        return "lift"
    elif progress < 0.8:
        return "move"
    elif progress < 0.9:
        return "place"
    else:
        return "release"


def run_vlm_on_samples(
    samples: List[Dict],
    task_description: str,
    vlm_planner,
    batch_size: int = 1,
    save_interval: int = 10,
    output_dir: str = None,
) -> List[Dict]:
    """
    Run VLM planner on all samples.

    Args:
        samples: List of sample dictionaries
        task_description: Task description for VLM
        vlm_planner: QwenVLPlanner instance
        batch_size: Batch size (currently 1 for VLM)
        save_interval: Save checkpoint every N samples
        output_dir: Directory to save checkpoints

    Returns:
        Samples augmented with VLM plans
    """
    augmented_samples = []

    for i, sample in enumerate(tqdm(samples, desc="Running VLM")):
        # Determine gripper state from state vector
        gripper_val = sample['state'][6]  # gripper position
        gripper_state = "closed" if gripper_val < 0 else "open"

        # Determine previous phase heuristically
        previous_phase = determine_phase(
            sample['step_idx'],
            sample['total_steps'],
            gripper_val,
        )

        # Run VLM
        try:
            start_time = time.time()
            plan = vlm_planner.plan(
                image=sample['image'],
                task_description=task_description,
                gripper_state=gripper_state,
                steps_since_plan=sample['step_idx'],
            )
            vlm_time = time.time() - start_time
        except Exception as e:
            print(f"[VLM Error] Sample {i}: {e}")
            plan = {
                "observation": {"distance_to_target": "unknown"},
                "plan": {
                    "phase": previous_phase,
                    "movements": [{"direction": "forward", "speed": "slow", "steps": 1}],
                    "gripper": "maintain",
                    "confidence": 0.3,
                },
                "reasoning": "VLM error - using heuristic",
            }
            vlm_time = 0

        # Add to sample
        sample['vlm_plan'] = plan
        sample['vlm_time'] = vlm_time
        sample['gripper_state'] = gripper_state
        augmented_samples.append(sample)

        # Save checkpoint
        if output_dir and (i + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(augmented_samples, f)
            print(f"Saved checkpoint: {checkpoint_path}")

    return augmented_samples


def create_mock_vlm_plans(samples: List[Dict], task_description: str) -> List[Dict]:
    """
    Create mock VLM plans using heuristics (for testing without VLM).

    This is useful for testing the training pipeline without GPU.
    """
    augmented_samples = []

    for sample in tqdm(samples, desc="Creating mock plans"):
        # Determine gripper state
        gripper_val = sample['state'][6]
        gripper_state = "closed" if gripper_val < 0 else "open"

        # Determine phase
        phase = determine_phase(
            sample['step_idx'],
            sample['total_steps'],
            gripper_val,
        )

        # Create mock plan based on phase
        if phase == "approach":
            movements = [{"direction": "forward", "speed": "fast", "steps": 2}]
            gripper = "open"
        elif phase == "align":
            movements = [{"direction": "forward", "speed": "slow", "steps": 1}]
            gripper = "open"
        elif phase == "descend":
            movements = [{"direction": "down", "speed": "slow", "steps": 2}]
            gripper = "open"
        elif phase == "grasp":
            movements = []
            gripper = "close"
        elif phase == "lift":
            movements = [{"direction": "up", "speed": "medium", "steps": 2}]
            gripper = "maintain"
        elif phase == "move":
            movements = [{"direction": "right", "speed": "fast", "steps": 2}]
            gripper = "maintain"
        elif phase == "place":
            movements = [{"direction": "down", "speed": "slow", "steps": 2}]
            gripper = "maintain"
        else:  # release
            movements = []
            gripper = "open"

        plan = {
            "observation": {
                "target_object": "black bowl",
                "distance_to_target": "medium",
            },
            "plan": {
                "phase": phase,
                "movements": movements,
                "gripper": gripper,
                "confidence": 0.8,
            },
            "reasoning": f"Mock plan for {phase} phase",
        }

        sample['vlm_plan'] = plan
        sample['vlm_time'] = 0
        sample['gripper_state'] = gripper_state
        augmented_samples.append(sample)

    return augmented_samples


def save_training_data(
    samples: List[Dict],
    output_path: str,
    include_images: bool = False,
):
    """
    Save augmented samples for training.

    Args:
        samples: List of augmented samples
        output_path: Path to save file
        include_images: Whether to include images (large!)
    """
    # Convert to training format
    training_data = {
        'states': [],
        'actions': [],
        'vlm_plans': [],
        'gripper_states': [],
        'step_indices': [],
    }

    if include_images:
        training_data['images'] = []

    for sample in samples:
        training_data['states'].append(sample['state'])
        training_data['actions'].append(sample['action'])
        training_data['vlm_plans'].append(sample['vlm_plan'])
        training_data['gripper_states'].append(sample['gripper_state'])
        training_data['step_indices'].append(sample['step_idx'])

        if include_images:
            training_data['images'].append(sample['image'])

    # Convert to numpy
    training_data['states'] = np.array(training_data['states'])
    training_data['actions'] = np.array(training_data['actions'])

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"Saved training data to {output_path}")
    print(f"  States: {training_data['states'].shape}")
    print(f"  Actions: {training_data['actions'].shape}")
    print(f"  VLM plans: {len(training_data['vlm_plans'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--n_demos", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--use_mock_vlm", action="store_true",
                        help="Use mock VLM plans (for testing without GPU)")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/src/data/vlm_plans")
    parser.add_argument("--model_path", type=str,
                        default="/workspace/src/models/qwen2.5-vl-7b")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get task info
    from libero.libero.benchmark import get_benchmark
    benchmark = get_benchmark(args.task_suite)()
    task = benchmark.get_task(args.task_id)
    task_name = task.name
    task_description = task.language

    print(f"Task: {task_name}")
    print(f"Description: {task_description}")

    # Find demo file
    demo_path = f"/workspace/data/libero/{args.task_suite}/{task_name}_demo.hdf5"
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        return

    # Load samples
    samples = load_demo_images_and_states(
        demo_path,
        n_demos=args.n_demos,
        sample_interval=args.sample_interval,
    )

    # Run VLM or create mock plans
    if args.use_mock_vlm:
        print("Using mock VLM plans (heuristic)")
        augmented_samples = create_mock_vlm_plans(samples, task_description)
    else:
        print("Loading VLM planner...")
        from src.vlm.qwen_planner import QwenVLPlanner
        vlm = QwenVLPlanner(model_name=args.model_path)

        augmented_samples = run_vlm_on_samples(
            samples,
            task_description,
            vlm,
            save_interval=50,
            output_dir=args.output_dir,
        )

    # Save training data
    output_path = os.path.join(
        args.output_dir,
        f"{args.task_suite}_{task_name}_vlm_plans.pkl"
    )
    save_training_data(augmented_samples, output_path)

    # Print stats
    if not args.use_mock_vlm:
        vlm_times = [s['vlm_time'] for s in augmented_samples if s['vlm_time'] > 0]
        if vlm_times:
            print(f"\nVLM timing stats:")
            print(f"  Avg: {np.mean(vlm_times):.2f}s")
            print(f"  Min: {np.min(vlm_times):.2f}s")
            print(f"  Max: {np.max(vlm_times):.2f}s")
            print(f"  FPS: {1/np.mean(vlm_times):.2f}")


if __name__ == "__main__":
    main()
