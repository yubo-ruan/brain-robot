#!/usr/bin/env python3
"""Convert collected perception data to YOLO training format.

YOLO requires:
- images/ directory with images
- labels/ directory with corresponding .txt files
- Each .txt line: class_id x_center y_center width height (normalized 0-1)
- data.yaml file with class names and paths

Usage:
    python scripts/convert_to_yolo_format.py --data-dir data/perception_v1 --output-dir data/yolo_libero
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


# Class name to ID mapping (must be consistent with detector)
CLASS_NAME_TO_ID = {
    "bowl": 0,
    "plate": 1,
    "mug": 2,
    "ramekin": 3,
    "cabinet": 4,
    "drawer": 5,
    "cookie_box": 6,
    "can": 7,
    "bottle": 8,
    "stove": 9,
}

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


def convert_bbox_to_yolo(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """Convert [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height].

    All values normalized to [0, 1].

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox

    # Convert to center format
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def process_frame(
    frame: dict,
    source_images_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
    image_size: Tuple[int, int],
    min_bbox_size: int = 8,
) -> dict:
    """Process a single frame, copying image and creating label file.

    Args:
        frame: Frame dict from our JSON format
        source_images_dir: Source directory for images
        dest_images_dir: Destination images directory
        dest_labels_dir: Destination labels directory
        image_size: (width, height) of images
        min_bbox_size: Minimum bbox dimension to include

    Returns:
        Stats dict with counts
    """
    stats = {
        "n_objects": 0,
        "n_valid_bboxes": 0,
        "n_skipped_small": 0,
        "n_skipped_invalid": 0,
        "class_counts": defaultdict(int),
    }

    # Construct image path
    image_path = source_images_dir.parent / frame["image_path"]
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return stats

    # Create unique filename
    base_name = f"task{frame['task_id']}_ep{frame['episode_idx']}_step{frame['step_idx']}"
    image_dest = dest_images_dir / f"{base_name}.png"
    label_dest = dest_labels_dir / f"{base_name}.txt"

    # Copy image
    shutil.copy2(image_path, image_dest)

    # Create label file
    label_lines = []
    image_width, image_height = image_size

    for obj_id, obj_data in frame["objects"].items():
        obj_class = obj_data.get("class", "unknown")
        bbox = obj_data.get("bbox", [])

        stats["n_objects"] += 1

        # Skip unknown classes
        if obj_class not in CLASS_NAME_TO_ID:
            stats["n_skipped_invalid"] += 1
            continue

        # Skip invalid bboxes
        if not bbox or len(bbox) != 4:
            stats["n_skipped_invalid"] += 1
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Skip small bboxes
        if width < min_bbox_size or height < min_bbox_size:
            stats["n_skipped_small"] += 1
            continue

        # Skip bboxes outside image
        if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
            stats["n_skipped_invalid"] += 1
            continue

        # Convert to YOLO format
        class_id = CLASS_NAME_TO_ID[obj_class]
        x_center, y_center, norm_width, norm_height = convert_bbox_to_yolo(
            bbox, image_width, image_height
        )

        # YOLO format: class_id x_center y_center width height
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )

        stats["n_valid_bboxes"] += 1
        stats["class_counts"][obj_class] += 1

    # Write label file (even if empty - YOLO handles empty labels)
    with open(label_dest, "w") as f:
        f.write("\n".join(label_lines))

    return stats


def create_data_yaml(
    output_dir: Path,
    train_images_dir: Path,
    val_images_dir: Path,
) -> Path:
    """Create YOLO data.yaml configuration file."""
    data_config = {
        "path": str(output_dir.absolute()),
        "train": str(train_images_dir.relative_to(output_dir)),
        "val": str(val_images_dir.relative_to(output_dir)),
        "names": CLASS_ID_TO_NAME,
        "nc": len(CLASS_NAME_TO_ID),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert to YOLO Format")
    parser.add_argument("--data-dir", type=str, default="data/perception_v1",
                        help="Source data directory with train.json and images/")
    parser.add_argument("--output-dir", type=str, default="data/yolo_libero",
                        help="Output directory for YOLO format data")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (assumed square)")
    parser.add_argument("--min-bbox-size", type=int, default=8,
                        help="Minimum bbox dimension (pixels)")
    args = parser.parse_args()

    source_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    image_size = (args.image_size, args.image_size)

    print("=" * 70)
    print("CONVERTING TO YOLO FORMAT")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Image size: {image_size}")
    print(f"Min bbox size: {args.min_bbox_size}px")
    print("=" * 70)

    # Create output directories
    train_images_dir = output_dir / "images" / "train"
    train_labels_dir = output_dir / "labels" / "train"
    val_images_dir = output_dir / "images" / "val"
    val_labels_dir = output_dir / "labels" / "val"

    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "train_class_counts": defaultdict(int),
        "val_class_counts": defaultdict(int),
    }

    # Process train set
    train_json = source_dir / "train.json"
    if train_json.exists():
        print(f"\nProcessing train set...")
        with open(train_json) as f:
            train_frames = json.load(f)

        for i, frame in enumerate(train_frames):
            stats = process_frame(
                frame,
                source_dir / "images",
                train_images_dir,
                train_labels_dir,
                image_size,
                args.min_bbox_size,
            )

            total_stats["train"]["n_frames"] += 1
            total_stats["train"]["n_objects"] += stats["n_objects"]
            total_stats["train"]["n_valid_bboxes"] += stats["n_valid_bboxes"]
            total_stats["train"]["n_skipped_small"] += stats["n_skipped_small"]
            total_stats["train"]["n_skipped_invalid"] += stats["n_skipped_invalid"]

            for cls, count in stats["class_counts"].items():
                total_stats["train_class_counts"][cls] += count

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(train_frames)} frames")

        print(f"  Train: {total_stats['train']['n_frames']} frames, "
              f"{total_stats['train']['n_valid_bboxes']} bboxes")

    # Process val set
    val_json = source_dir / "val.json"
    if val_json.exists():
        print(f"\nProcessing val set...")
        with open(val_json) as f:
            val_frames = json.load(f)

        for i, frame in enumerate(val_frames):
            stats = process_frame(
                frame,
                source_dir / "images",
                val_images_dir,
                val_labels_dir,
                image_size,
                args.min_bbox_size,
            )

            total_stats["val"]["n_frames"] += 1
            total_stats["val"]["n_objects"] += stats["n_objects"]
            total_stats["val"]["n_valid_bboxes"] += stats["n_valid_bboxes"]
            total_stats["val"]["n_skipped_small"] += stats["n_skipped_small"]
            total_stats["val"]["n_skipped_invalid"] += stats["n_skipped_invalid"]

            for cls, count in stats["class_counts"].items():
                total_stats["val_class_counts"][cls] += count

        print(f"  Val: {total_stats['val']['n_frames']} frames, "
              f"{total_stats['val']['n_valid_bboxes']} bboxes")

    # Create data.yaml
    yaml_path = create_data_yaml(output_dir, train_images_dir, val_images_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"\nTrain set:")
    print(f"  Frames: {total_stats['train']['n_frames']}")
    print(f"  Valid bboxes: {total_stats['train']['n_valid_bboxes']}")
    print(f"  Skipped (small): {total_stats['train']['n_skipped_small']}")
    print(f"  Skipped (invalid): {total_stats['train']['n_skipped_invalid']}")

    print(f"\nVal set:")
    print(f"  Frames: {total_stats['val']['n_frames']}")
    print(f"  Valid bboxes: {total_stats['val']['n_valid_bboxes']}")
    print(f"  Skipped (small): {total_stats['val']['n_skipped_small']}")
    print(f"  Skipped (invalid): {total_stats['val']['n_skipped_invalid']}")

    print(f"\nClass distribution (train):")
    for cls in sorted(total_stats["train_class_counts"].keys()):
        train_count = total_stats["train_class_counts"][cls]
        val_count = total_stats["val_class_counts"].get(cls, 0)
        print(f"  {cls}: {train_count} train, {val_count} val")

    print(f"\nOutput files:")
    print(f"  {yaml_path}")
    print(f"  {train_images_dir}")
    print(f"  {train_labels_dir}")
    print(f"  {val_images_dir}")
    print(f"  {val_labels_dir}")


if __name__ == "__main__":
    main()
