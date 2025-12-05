#!/usr/bin/env python3
"""
Regenerate LIBERO demonstrations at 512x512 resolution with depth.

This script replays the original LIBERO demonstrations and captures
new observations at higher resolution with depth enabled.

REQUIREMENTS:
- Downloaded LIBERO HDF5 files (use download_libero_datasets.py)
- Working OpenGL/rendering (OSMesa, EGL, or display)
- GPU recommended for faster processing

Usage:
    python regenerate_libero_512.py --input-dir /path/to/libero_datasets
"""

import os
import sys

# Set rendering backend BEFORE any mujoco/robosuite imports
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import h5py
import numpy as np
from tqdm import tqdm

# LIBERO imports
sys.path.insert(0, '/workspace/LIBERO')
from libero.libero import get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.utils.utils as libero_utils
import robosuite.utils.transform_utils as T


DEFAULT_RESOLUTION = 512
DEFAULT_CAMERAS = ["agentview", "robot0_eye_in_hand"]


def regenerate_single_file(
    input_file: Path,
    output_dir: Path,
    resolution: int = 512,
    cameras: List[str] = None,
    enable_depth: bool = True,
    skip_existing: bool = True,
    verbose: bool = False,
) -> Dict:
    """Regenerate a single HDF5 demonstration file.

    Args:
        input_file: Path to original HDF5 file
        output_dir: Directory for output
        resolution: Image resolution
        cameras: Camera names
        enable_depth: Enable depth capture
        skip_existing: Skip if output exists
        verbose: Verbose output

    Returns:
        Result dictionary
    """
    if cameras is None:
        cameras = DEFAULT_CAMERAS

    output_file = output_dir / input_file.name

    # Skip if already exists
    if skip_existing and output_file.exists():
        if verbose:
            print(f"[SKIP] {input_file.name}")
        return {"status": "skipped", "file": str(input_file)}

    try:
        with h5py.File(input_file, "r") as f_in:
            # Get environment configuration
            env_args = json.loads(f_in["data"].attrs.get("env_args", "{}"))
            problem_info = json.loads(f_in["data"].attrs.get("problem_info", "{}"))

            # Find BDDL file
            bddl_file = f_in["data"].attrs.get("bddl_file_name", b"")
            if isinstance(bddl_file, bytes):
                bddl_file = bddl_file.decode()

            if not bddl_file:
                bddl_file = env_args.get("bddl_file", "")

            if not bddl_file or not os.path.exists(bddl_file):
                return {
                    "status": "error",
                    "file": str(input_file),
                    "error": f"BDDL file not found: {bddl_file}"
                }

            # Get problem name
            problem_name = problem_info.get("problem_name", "")
            if not problem_name:
                problem_name = env_args.get("problem_name", "")

            if problem_name not in TASK_MAPPING:
                return {
                    "status": "error",
                    "file": str(input_file),
                    "error": f"Unknown problem: {problem_name}"
                }

            # Configure environment
            env_kwargs = env_args.get("env_kwargs", {}).copy()
            env_kwargs.update({
                "bddl_file_name": bddl_file,
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "use_camera_obs": True,
                "camera_depths": enable_depth,
                "camera_names": cameras,
                "camera_heights": resolution,
                "camera_widths": resolution,
                "ignore_done": True,
                "reward_shaping": True,
                "control_freq": 20,
            })

            # Create environment
            env = TASK_MAPPING[problem_name](**env_kwargs)

            # Get demo list
            demos = sorted([k for k in f_in["data"].keys() if k.startswith("demo")])

            # Create output directory and file
            output_dir.mkdir(parents=True, exist_ok=True)

            with h5py.File(output_file, "w") as f_out:
                # Create data group and copy attributes
                grp_out = f_out.create_group("data")
                for attr_name, attr_val in f_in["data"].attrs.items():
                    grp_out.attrs[attr_name] = attr_val

                # Add regeneration metadata
                grp_out.attrs["resolution"] = resolution
                grp_out.attrs["depth_enabled"] = enable_depth
                grp_out.attrs["regenerated"] = True
                grp_out.attrs["regenerated_at"] = datetime.now().isoformat()

                total_frames = 0
                total_demos = 0

                # Process each demo
                for demo_key in demos:
                    demo_in = f_in[f"data/{demo_key}"]

                    states = demo_in["states"][()]
                    actions = demo_in["actions"][()]
                    model_xml = demo_in.attrs.get("model_file", b"")
                    if isinstance(model_xml, bytes):
                        model_xml = model_xml.decode()

                    # Reset environment
                    env.reset()
                    if model_xml:
                        try:
                            model_xml = libero_utils.postprocess_model_xml(model_xml, {})
                            env.reset_from_xml_string(model_xml)
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: Could not load model XML: {e}")

                    env.sim.reset()
                    env.sim.set_state_from_flattened(states[0])
                    env.sim.forward()

                    # Collect observations
                    obs_lists = {
                        "agentview_rgb": [],
                        "eye_in_hand_rgb": [],
                        "ee_states": [],
                        "gripper_states": [],
                        "joint_states": [],
                    }
                    if enable_depth:
                        obs_lists["agentview_depth"] = []
                        obs_lists["eye_in_hand_depth"] = []

                    robot_states_list = []
                    rewards_list = []
                    dones_list = []

                    # Skip initial frames (force sensor stabilization)
                    skip_n = 5

                    # Replay actions
                    for idx, action in enumerate(actions):
                        obs, reward, done, info = env.step(action)

                        if idx < skip_n:
                            continue

                        # RGB observations
                        obs_lists["agentview_rgb"].append(obs["agentview_image"])
                        obs_lists["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])

                        # Depth observations
                        if enable_depth:
                            obs_lists["agentview_depth"].append(
                                obs.get("agentview_depth", np.zeros((resolution, resolution)))
                            )
                            obs_lists["eye_in_hand_depth"].append(
                                obs.get("robot0_eye_in_hand_depth", np.zeros((resolution, resolution)))
                            )

                        # Proprioception
                        if "robot0_gripper_qpos" in obs:
                            obs_lists["gripper_states"].append(obs["robot0_gripper_qpos"])
                        if "robot0_joint_pos" in obs:
                            obs_lists["joint_states"].append(obs["robot0_joint_pos"])
                        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                            ee = np.hstack((
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"])
                            ))
                            obs_lists["ee_states"].append(ee)

                        robot_states_list.append(env.get_robot_state_vector(obs))
                        rewards_list.append(reward)
                        dones_list.append(done)

                    # Save demo
                    n_frames = len(obs_lists["agentview_rgb"])
                    if n_frames == 0:
                        continue

                    demo_out = grp_out.create_group(demo_key)
                    obs_out = demo_out.create_group("obs")

                    # Save observations with compression
                    for key, data in obs_lists.items():
                        if data:
                            obs_out.create_dataset(
                                key,
                                data=np.stack(data, axis=0),
                                compression="gzip",
                                compression_opts=4
                            )

                    # Save actions, states, etc.
                    valid_actions = actions[skip_n:skip_n + n_frames]
                    valid_states = states[skip_n:skip_n + n_frames]

                    demo_out.create_dataset("actions", data=valid_actions)
                    demo_out.create_dataset("states", data=valid_states)
                    demo_out.create_dataset(
                        "robot_states",
                        data=np.stack(robot_states_list, axis=0)
                    )
                    demo_out.create_dataset("rewards", data=np.array(rewards_list))
                    demo_out.create_dataset(
                        "dones",
                        data=np.array(dones_list).astype(np.uint8)
                    )

                    demo_out.attrs["num_samples"] = n_frames
                    if model_xml:
                        demo_out.attrs["model_file"] = model_xml
                    demo_out.attrs["init_state"] = states[0]

                    total_frames += n_frames
                    total_demos += 1

                grp_out.attrs["num_demos"] = total_demos
                grp_out.attrs["total_frames"] = total_frames

            env.close()

            if verbose:
                print(f"[OK] {input_file.name}: {total_demos} demos, {total_frames} frames")

            return {
                "status": "success",
                "file": str(input_file),
                "demos": total_demos,
                "frames": total_frames,
            }

    except Exception as e:
        import traceback
        if verbose:
            print(f"[ERROR] {input_file.name}: {e}")
            traceback.print_exc()
        return {
            "status": "error",
            "file": str(input_file),
            "error": str(e),
        }


def regenerate_suite(
    suite_name: str,
    input_dir: Path,
    output_dir: Path,
    resolution: int = 512,
    enable_depth: bool = True,
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Regenerate all files in a suite."""
    print(f"\n{'='*60}")
    print(f"Regenerating: {suite_name}")
    print(f"Resolution: {resolution}x{resolution}, Depth: {enable_depth}")
    print(f"{'='*60}")

    # Find HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    if max_files:
        hdf5_files = hdf5_files[:max_files]

    if not hdf5_files:
        print(f"[WARNING] No HDF5 files found in {input_dir}")
        return {"suite": suite_name, "status": "no_files"}

    print(f"Found {len(hdf5_files)} files")

    suite_output = output_dir / suite_name
    results = []

    for hdf5_file in tqdm(hdf5_files, desc=suite_name):
        result = regenerate_single_file(
            input_file=hdf5_file,
            output_dir=suite_output,
            resolution=resolution,
            enable_depth=enable_depth,
            verbose=verbose,
        )
        results.append(result)

    # Summary
    success = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if r.get("status") == "error")
    total_demos = sum(r.get("demos", 0) for r in results)
    total_frames = sum(r.get("frames", 0) for r in results)

    print(f"\n{suite_name}: {success} success, {skipped} skipped, {errors} errors")
    print(f"Total: {total_demos} demos, {total_frames} frames")

    return {
        "suite": suite_name,
        "files": len(hdf5_files),
        "success": success,
        "skipped": skipped,
        "errors": errors,
        "demos": total_demos,
        "frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate LIBERO demonstrations at 512x512 with depth"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/workspace/new_experiment/libero_datasets",
        help="Directory containing downloaded LIBERO HDF5 files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/new_experiment/libero_512",
        help="Output directory for regenerated files"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="Specific suites to process (default: all found)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (default: 512)"
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth capture"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files per suite (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find suites to process
    if args.suites:
        suites = args.suites
    else:
        # Auto-detect from input directory
        suites = [d.name for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("libero")]

    if not suites:
        print(f"[ERROR] No LIBERO suites found in {input_dir}")
        print("Run download_libero_datasets.py first")
        return

    print("="*60)
    print("LIBERO Regeneration at High Resolution")
    print("="*60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Suites: {suites}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Depth: {not args.no_depth}")
    print("="*60)

    # Process each suite
    all_results = []

    for suite in suites:
        suite_input = input_dir / suite
        if not suite_input.exists():
            print(f"[SKIP] {suite}: directory not found")
            continue

        result = regenerate_suite(
            suite_name=suite,
            input_dir=suite_input,
            output_dir=output_dir,
            resolution=args.resolution,
            enable_depth=not args.no_depth,
            max_files=args.max_files,
            verbose=args.verbose,
        )
        all_results.append(result)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    total_demos = sum(r.get("demos", 0) for r in all_results)
    total_frames = sum(r.get("frames", 0) for r in all_results)

    for r in all_results:
        print(f"  {r['suite']}: {r.get('success', 0)}/{r.get('files', 0)} files, {r.get('demos', 0)} demos")

    print("-"*60)
    print(f"  TOTAL: {total_demos} demos, {total_frames} frames")
    print("="*60)

    # Save summary
    summary_file = output_dir / "regeneration_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "resolution": args.resolution,
            "depth": not args.no_depth,
            "results": all_results,
            "total_demos": total_demos,
            "total_frames": total_frames,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
