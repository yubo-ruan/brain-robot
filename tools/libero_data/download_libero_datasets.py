#!/usr/bin/env python3
"""
Download LIBERO demonstration datasets from HuggingFace.

This script downloads the raw HDF5 demonstration files.
No rendering/OpenGL required - just downloads the files.

After downloading, use regenerate_libero_512.py on a machine
with rendering capabilities to convert to 512x512 with depth.

Total: ~6,500 demonstrations across 130 tasks
- libero_spatial: 10 tasks × 50 demos = 500
- libero_object: 10 tasks × 50 demos = 500
- libero_goal: 10 tasks × 50 demos = 500
- libero_100: 100 tasks × 50 demos = 5,000 (split as libero_90 + libero_10)
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

DATASET_INFO = {
    "libero_spatial": {"num_tasks": 10, "demos_per_task": 50, "total_demos": 500},
    "libero_object": {"num_tasks": 10, "demos_per_task": 50, "total_demos": 500},
    "libero_goal": {"num_tasks": 10, "demos_per_task": 50, "total_demos": 500},
    "libero_100": {"num_tasks": 100, "demos_per_task": 50, "total_demos": 5000},
}


def download_datasets(
    suites: list,
    download_dir: str,
    force_download: bool = False,
) -> dict:
    """Download LIBERO datasets from HuggingFace.

    Args:
        suites: List of suite names to download
        download_dir: Directory to save datasets
        force_download: If True, re-download even if exists

    Returns:
        Dictionary with download statistics
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub required. Install with: pip install huggingface_hub"
        )

    os.makedirs(download_dir, exist_ok=True)
    results = {}

    for suite in suites:
        suite_dir = os.path.join(download_dir, suite)

        # Check if already downloaded
        if os.path.exists(suite_dir) and not force_download:
            hdf5_files = list(Path(suite_dir).glob("*.hdf5"))
            expected = DATASET_INFO.get(suite, {}).get("num_tasks", 0)

            if len(hdf5_files) >= expected:
                print(f"[SKIP] {suite}: already exists with {len(hdf5_files)} files")
                results[suite] = {
                    "status": "skipped",
                    "files": len(hdf5_files),
                    "path": suite_dir,
                }
                continue

        print(f"\n[DOWNLOAD] {suite}...")

        try:
            # Download from HuggingFace
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=download_dir,
                allow_patterns=f"{suite}/*",
                local_dir_use_symlinks=False,
            )

            # Verify download
            if os.path.exists(suite_dir):
                hdf5_files = list(Path(suite_dir).glob("*.hdf5"))
                print(f"[OK] Downloaded {len(hdf5_files)} files for {suite}")
                results[suite] = {
                    "status": "success",
                    "files": len(hdf5_files),
                    "path": suite_dir,
                }
            else:
                print(f"[ERROR] Directory not created for {suite}")
                results[suite] = {"status": "error", "error": "Directory not created"}

        except Exception as e:
            print(f"[ERROR] Failed to download {suite}: {e}")
            results[suite] = {"status": "error", "error": str(e)}

    return results


def print_dataset_info():
    """Print information about available datasets."""
    print("\nLIBERO Dataset Information:")
    print("="*60)

    total_tasks = 0
    total_demos = 0

    for suite, info in DATASET_INFO.items():
        print(f"  {suite}:")
        print(f"    Tasks: {info['num_tasks']}")
        print(f"    Demos per task: {info['demos_per_task']}")
        print(f"    Total demos: {info['total_demos']}")
        total_tasks += info['num_tasks']
        total_demos += info['total_demos']

    print("-"*60)
    print(f"  TOTAL: {total_tasks} tasks, {total_demos} demonstrations")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download LIBERO datasets from HuggingFace"
    )

    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_100"],
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_100", "all"],
        help="Dataset suites to download"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/workspace/new_experiment/libero_datasets",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit"
    )

    args = parser.parse_args()

    if args.info:
        print_dataset_info()
        return

    # Handle 'all' option
    if "all" in args.suites:
        suites = list(DATASET_INFO.keys())
    else:
        suites = args.suites

    print("="*60)
    print("LIBERO Dataset Download")
    print("="*60)
    print(f"Suites: {suites}")
    print(f"Download directory: {args.download_dir}")
    print("="*60)

    # Download
    results = download_datasets(
        suites=suites,
        download_dir=args.download_dir,
        force_download=args.force,
    )

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    total_files = 0
    for suite, result in results.items():
        status = result.get("status", "unknown")
        files = result.get("files", 0)
        total_files += files

        if status == "success":
            print(f"  {suite}: {files} files downloaded")
        elif status == "skipped":
            print(f"  {suite}: {files} files (already exists)")
        else:
            print(f"  {suite}: ERROR - {result.get('error', 'unknown')}")

    print("-"*60)
    print(f"  Total: {total_files} HDF5 files")
    print("="*60)

    # Save results
    results_file = Path(args.download_dir) / "download_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "suites": results,
            "total_files": total_files,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n[NEXT STEPS]")
    print("To regenerate at 512x512 with depth, run on a machine with GPU/rendering:")
    print(f"  python regenerate_libero_512.py --input-dir {args.download_dir}")


if __name__ == "__main__":
    main()
