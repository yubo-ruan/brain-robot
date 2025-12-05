#!/usr/bin/env python3
"""
Download LIBERO demonstration datasets and regenerate at 512x512 with depth.

This script:
1. Downloads all LIBERO demonstration HDF5 files from HuggingFace
2. Replays each demonstration in the environment at 512x512 resolution
3. Saves new HDF5 files with high-resolution RGB + depth observations

Total: ~6,500 demonstrations across 130 tasks
- libero_spatial: 10 tasks × 50 demos = 500
- libero_object: 10 tasks × 50 demos = 500
- libero_goal: 10 tasks × 50 demos = 500
- libero_100: 100 tasks × 50 demos = 5,000
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
import multiprocessing as mp
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

# LIBERO imports
sys.path.insert(0, '/workspace/LIBERO')
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import TASK_MAPPING
import libero.libero.utils.utils as libero_utils

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


# =============================================================================
# Configuration
# =============================================================================

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

DATASET_SUITES = {
    "libero_spatial": {"num_tasks": 10, "demos_per_task": 50},
    "libero_object": {"num_tasks": 10, "demos_per_task": 50},
    "libero_goal": {"num_tasks": 10, "demos_per_task": 50},
    "libero_90": {"num_tasks": 90, "demos_per_task": 50},
    "libero_10": {"num_tasks": 10, "demos_per_task": 50},
}

DEFAULT_RESOLUTION = 512
DEFAULT_CAMERAS = ["agentview", "robot0_eye_in_hand"]


# =============================================================================
# Download Functions
# =============================================================================

def download_libero_datasets(
    suites: List[str],
    download_dir: str,
    force_download: bool = False
) -> Dict[str, Path]:
    """Download LIBERO datasets from HuggingFace.

    Args:
        suites: List of suite names to download
        download_dir: Directory to save datasets
        force_download: If True, re-download even if exists

    Returns:
        Dictionary mapping suite names to their paths
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")

    os.makedirs(download_dir, exist_ok=True)
    suite_paths = {}

    for suite in suites:
        suite_dir = os.path.join(download_dir, suite)

        # Check if already downloaded
        if os.path.exists(suite_dir) and not force_download:
            hdf5_count = len(list(Path(suite_dir).glob("*.hdf5")))
            expected = DATASET_SUITES.get(suite, {}).get("num_tasks", 0)
            if hdf5_count >= expected:
                print(f"[SKIP] {suite} already exists with {hdf5_count} files")
                suite_paths[suite] = Path(suite_dir)
                continue

        print(f"\n[DOWNLOAD] {suite} from HuggingFace...")

        # Map suite names for download
        download_pattern = suite
        if suite in ["libero_90", "libero_10"]:
            # These are part of libero_100 but split differently
            download_pattern = "libero_100"

        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=download_dir,
                allow_patterns=f"{download_pattern}/*",
                local_dir_use_symlinks=False,
            )

            # Verify download
            actual_dir = os.path.join(download_dir, download_pattern)
            if os.path.exists(actual_dir):
                hdf5_count = len(list(Path(actual_dir).glob("*.hdf5")))
                print(f"[OK] Downloaded {hdf5_count} files for {suite}")
                suite_paths[suite] = Path(actual_dir)
            else:
                print(f"[ERROR] Failed to download {suite}")

        except Exception as e:
            print(f"[ERROR] Failed to download {suite}: {e}")

    return suite_paths


# =============================================================================
# Regeneration Functions
# =============================================================================

def regenerate_single_demo(
    demo_file: Path,
    output_dir: Path,
    resolution: int = 512,
    cameras: List[str] = None,
    enable_depth: bool = True,
    verbose: bool = False,
) -> Optional[Dict]:
    """Regenerate a single demonstration at new resolution with depth.

    Args:
        demo_file: Path to original HDF5 demo file
        output_dir: Directory to save regenerated demo
        resolution: Image resolution (height = width)
        cameras: List of camera names
        enable_depth: Whether to capture depth images
        verbose: Print progress

    Returns:
        Dictionary with stats, or None if failed
    """
    if cameras is None:
        cameras = DEFAULT_CAMERAS

    output_file = output_dir / demo_file.name

    # Skip if already exists
    if output_file.exists():
        if verbose:
            print(f"[SKIP] {demo_file.name} already regenerated")
        return {"status": "skipped", "file": str(demo_file)}

    try:
        # Open original file
        with h5py.File(demo_file, "r") as f_in:
            # Get environment info
            env_args = json.loads(f_in["data"].attrs.get("env_args", "{}"))
            problem_info = json.loads(f_in["data"].attrs.get("problem_info", "{}"))
            bddl_file = f_in["data"].attrs.get("bddl_file_name", "")

            if not bddl_file or not os.path.exists(bddl_file):
                # Try to find BDDL file
                if "bddl_file" in env_args:
                    bddl_file = env_args["bddl_file"]

            if not bddl_file or not os.path.exists(bddl_file):
                print(f"[ERROR] BDDL file not found for {demo_file.name}")
                return {"status": "error", "file": str(demo_file), "error": "BDDL not found"}

            # Get problem name
            problem_name = problem_info.get("problem_name", "")
            if not problem_name:
                problem_name = env_args.get("problem_name", "")

            if problem_name not in TASK_MAPPING:
                print(f"[ERROR] Unknown problem: {problem_name}")
                return {"status": "error", "file": str(demo_file), "error": f"Unknown problem: {problem_name}"}

            # Get env kwargs and update for new resolution
            env_kwargs = env_args.get("env_kwargs", {})
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

            # Get list of demos
            demos = sorted([k for k in f_in["data"].keys() if k.startswith("demo")])

            # Create output file
            output_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(output_file, "w") as f_out:
                # Copy top-level attributes
                grp_out = f_out.create_group("data")
                for attr_name, attr_val in f_in["data"].attrs.items():
                    grp_out.attrs[attr_name] = attr_val

                # Update resolution info
                grp_out.attrs["camera_height"] = resolution
                grp_out.attrs["camera_width"] = resolution
                grp_out.attrs["camera_depth"] = enable_depth
                grp_out.attrs["regenerated"] = True
                grp_out.attrs["regenerated_timestamp"] = datetime.now().isoformat()

                total_frames = 0

                # Process each demo
                for demo_idx, demo_key in enumerate(demos):
                    demo_grp_in = f_in[f"data/{demo_key}"]

                    # Get states and actions
                    states = demo_grp_in["states"][()]
                    actions = demo_grp_in["actions"][()]

                    # Get model XML
                    model_xml = demo_grp_in.attrs.get("model_file", "")

                    # Reset environment
                    env.reset()
                    if model_xml:
                        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
                        env.reset_from_xml_string(model_xml)

                    env.sim.reset()
                    env.sim.set_state_from_flattened(states[0])
                    env.sim.forward()

                    # Collect observations
                    obs_data = {
                        "agentview_rgb": [],
                        "eye_in_hand_rgb": [],
                        "ee_states": [],
                        "gripper_states": [],
                        "joint_states": [],
                    }
                    if enable_depth:
                        obs_data["agentview_depth"] = []
                        obs_data["eye_in_hand_depth"] = []

                    robot_states = []
                    rewards = []
                    dones = []

                    # Skip first few frames (force sensor stabilization)
                    skip_frames = 5

                    # Replay actions
                    for step_idx, action in enumerate(actions):
                        obs, reward, done, info = env.step(action)

                        if step_idx < skip_frames:
                            continue

                        # Collect RGB
                        obs_data["agentview_rgb"].append(obs["agentview_image"])
                        obs_data["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])

                        # Collect depth
                        if enable_depth:
                            obs_data["agentview_depth"].append(obs.get("agentview_depth", np.zeros((resolution, resolution))))
                            obs_data["eye_in_hand_depth"].append(obs.get("robot0_eye_in_hand_depth", np.zeros((resolution, resolution))))

                        # Collect proprioception
                        if "robot0_gripper_qpos" in obs:
                            obs_data["gripper_states"].append(obs["robot0_gripper_qpos"])
                        if "robot0_joint_pos" in obs:
                            obs_data["joint_states"].append(obs["robot0_joint_pos"])
                        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
                            import robosuite.utils.transform_utils as T
                            ee_state = np.hstack((
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"])
                            ))
                            obs_data["ee_states"].append(ee_state)

                        robot_states.append(env.get_robot_state_vector(obs))
                        rewards.append(reward)
                        dones.append(done)

                    # Save demo
                    demo_grp_out = grp_out.create_group(demo_key)
                    obs_grp = demo_grp_out.create_group("obs")

                    # Save observations
                    valid_actions = actions[skip_frames:]
                    valid_states = states[skip_frames:]

                    for obs_key, obs_list in obs_data.items():
                        if obs_list:
                            obs_grp.create_dataset(
                                obs_key,
                                data=np.stack(obs_list, axis=0),
                                compression="gzip",
                                compression_opts=4
                            )

                    # Save other data
                    demo_grp_out.create_dataset("actions", data=valid_actions)
                    demo_grp_out.create_dataset("states", data=valid_states)
                    demo_grp_out.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                    demo_grp_out.create_dataset("rewards", data=np.array(rewards))
                    demo_grp_out.create_dataset("dones", data=np.array(dones).astype(np.uint8))

                    # Copy attributes
                    demo_grp_out.attrs["num_samples"] = len(obs_data["agentview_rgb"])
                    if model_xml:
                        demo_grp_out.attrs["model_file"] = model_xml
                    demo_grp_out.attrs["init_state"] = states[0]

                    total_frames += len(obs_data["agentview_rgb"])

                    if verbose:
                        print(f"  Demo {demo_idx + 1}/{len(demos)}: {len(obs_data['agentview_rgb'])} frames")

                grp_out.attrs["num_demos"] = len(demos)
                grp_out.attrs["total_frames"] = total_frames

            env.close()

            if verbose:
                print(f"[OK] Regenerated {demo_file.name}: {len(demos)} demos, {total_frames} frames")

            return {
                "status": "success",
                "file": str(demo_file),
                "num_demos": len(demos),
                "total_frames": total_frames
            }

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to regenerate {demo_file.name}: {e}")
        if verbose:
            traceback.print_exc()
        return {"status": "error", "file": str(demo_file), "error": str(e)}


def regenerate_suite(
    suite_name: str,
    input_dir: Path,
    output_dir: Path,
    resolution: int = 512,
    enable_depth: bool = True,
    num_workers: int = 1,
    max_tasks: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Regenerate all demos in a suite at new resolution.

    Args:
        suite_name: Name of the suite
        input_dir: Directory containing original HDF5 files
        output_dir: Directory to save regenerated files
        resolution: Image resolution
        enable_depth: Whether to capture depth
        num_workers: Number of parallel workers (1 = sequential)
        max_tasks: Maximum number of tasks to process (for testing)
        verbose: Print progress

    Returns:
        Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"Regenerating {suite_name} at {resolution}x{resolution} (depth={enable_depth})")
    print(f"{'='*60}")

    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))

    if max_tasks:
        hdf5_files = hdf5_files[:max_tasks]

    print(f"Found {len(hdf5_files)} task files")

    suite_output_dir = output_dir / suite_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if num_workers > 1:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            func = partial(
                regenerate_single_demo,
                output_dir=suite_output_dir,
                resolution=resolution,
                enable_depth=enable_depth,
                verbose=verbose,
            )
            results = list(tqdm(
                pool.imap(func, hdf5_files),
                total=len(hdf5_files),
                desc=f"Processing {suite_name}"
            ))
    else:
        # Sequential processing
        for hdf5_file in tqdm(hdf5_files, desc=f"Processing {suite_name}"):
            result = regenerate_single_demo(
                demo_file=hdf5_file,
                output_dir=suite_output_dir,
                resolution=resolution,
                enable_depth=enable_depth,
                verbose=verbose,
            )
            results.append(result)

    # Summarize results
    success = sum(1 for r in results if r and r.get("status") == "success")
    skipped = sum(1 for r in results if r and r.get("status") == "skipped")
    errors = sum(1 for r in results if r and r.get("status") == "error")
    total_demos = sum(r.get("num_demos", 0) for r in results if r)
    total_frames = sum(r.get("total_frames", 0) for r in results if r)

    summary = {
        "suite": suite_name,
        "total_files": len(hdf5_files),
        "success": success,
        "skipped": skipped,
        "errors": errors,
        "total_demos": total_demos,
        "total_frames": total_frames,
    }

    print(f"\n{suite_name} Summary:")
    print(f"  Files: {success} success, {skipped} skipped, {errors} errors")
    print(f"  Total: {total_demos} demos, {total_frames} frames")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and regenerate LIBERO datasets")

    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"],
        help="Dataset suites to process"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/workspace/new_experiment/libero_datasets_original",
        help="Directory for original downloads"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/new_experiment/libero_datasets_512",
        help="Directory for regenerated datasets"
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
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks per suite (for testing)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step"
    )
    parser.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Skip regeneration step"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("="*60)
    print("LIBERO Dataset Download & Regeneration")
    print("="*60)
    print(f"Suites: {args.suites}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Depth: {not args.no_depth}")
    print(f"Download dir: {args.download_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Step 1: Download
    if not args.skip_download:
        print("\n[STEP 1] Downloading datasets from HuggingFace...")
        suite_paths = download_libero_datasets(
            suites=args.suites,
            download_dir=args.download_dir,
            force_download=args.force_download
        )
    else:
        print("\n[STEP 1] Skipping download...")
        suite_paths = {}
        for suite in args.suites:
            # Handle libero_100 split
            if suite in ["libero_90", "libero_10"]:
                path = Path(args.download_dir) / "libero_100"
            else:
                path = Path(args.download_dir) / suite
            if path.exists():
                suite_paths[suite] = path

    # Step 2: Regenerate
    if not args.skip_regenerate:
        print("\n[STEP 2] Regenerating at new resolution...")

        all_summaries = []

        for suite in args.suites:
            input_dir = suite_paths.get(suite)
            if not input_dir or not input_dir.exists():
                print(f"[SKIP] {suite}: input directory not found")
                continue

            summary = regenerate_suite(
                suite_name=suite,
                input_dir=input_dir,
                output_dir=Path(args.output_dir),
                resolution=args.resolution,
                enable_depth=not args.no_depth,
                num_workers=args.num_workers,
                max_tasks=args.max_tasks,
                verbose=args.verbose,
            )
            all_summaries.append(summary)

        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        total_demos = sum(s["total_demos"] for s in all_summaries)
        total_frames = sum(s["total_frames"] for s in all_summaries)
        total_success = sum(s["success"] for s in all_summaries)
        total_errors = sum(s["errors"] for s in all_summaries)

        for s in all_summaries:
            print(f"  {s['suite']}: {s['success']}/{s['total_files']} files, {s['total_demos']} demos")

        print(f"\nTotal: {total_success} files, {total_demos} demos, {total_frames} frames")
        if total_errors > 0:
            print(f"Errors: {total_errors}")

        # Save summary
        summary_file = Path(args.output_dir) / "regeneration_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "resolution": args.resolution,
                "depth": not args.no_depth,
                "suites": all_summaries,
                "total_demos": total_demos,
                "total_frames": total_frames,
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
