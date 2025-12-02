#!/usr/bin/env python3
"""Visualize collected perception data with bounding box overlays.

Validates that 3D→2D projection is working correctly by overlaying
ground-truth bounding boxes on the collected images.

Usage:
    python scripts/visualize_perception_data.py --data-dir data/perception_test --n-samples 5
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# Colors for different object classes
CLASS_COLORS = {
    "bowl": (255, 0, 0),       # Red
    "plate": (0, 255, 0),      # Green
    "mug": (0, 0, 255),        # Blue
    "ramekin": (255, 255, 0),  # Yellow
    "cabinet": (128, 0, 128),  # Purple
    "drawer": (0, 128, 128),   # Teal
    "cookie_box": (255, 128, 0),  # Orange
    "can": (128, 128, 0),      # Olive
    "bottle": (0, 128, 0),     # Dark green
    "stove": (128, 0, 0),      # Dark red
    "unknown": (128, 128, 128),  # Gray
}


def draw_bbox(draw, bbox, label, color, thickness=2):
    """Draw a bounding box with label on the image."""
    if not bbox or len(bbox) != 4:
        return

    x1, y1, x2, y2 = bbox

    # Draw rectangle
    for i in range(thickness):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x1, y1 - 12), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, y1 - 12), label, fill=(255, 255, 255), font=font)


def visualize_frame(data_dir: Path, frame: dict, output_dir: Path) -> dict:
    """Visualize a single frame with bbox overlays.

    Returns:
        Dict with bbox statistics for this frame.
    """
    # Load image
    image_path = data_dir / frame["image_path"]
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return {}

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    stats = {
        "n_objects": 0,
        "n_valid_bboxes": 0,
        "bbox_sizes": [],
        "class_counts": defaultdict(int),
    }

    # Draw each object's bbox
    for obj_id, obj_data in frame["objects"].items():
        obj_class = obj_data.get("class", "unknown")
        bbox = obj_data.get("bbox", [])

        stats["n_objects"] += 1
        stats["class_counts"][obj_class] += 1

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            if width > 0 and height > 0:
                stats["n_valid_bboxes"] += 1
                stats["bbox_sizes"].append({
                    "class": obj_class,
                    "width": width,
                    "height": height,
                    "min_dim": min(width, height),
                })

                # Draw bbox
                color = CLASS_COLORS.get(obj_class, CLASS_COLORS["unknown"])
                label = f"{obj_class}"
                draw_bbox(draw, bbox, label, color)

    # Draw spatial relations as text
    y_offset = 5
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except:
        font = ImageFont.load_default()

    if frame.get("on_relations"):
        for obj, surface in list(frame["on_relations"].items())[:3]:
            obj_short = obj.split("_")[0] if "_" in obj else obj
            surf_short = surface.split("_")[0] if "_" in surface else surface
            text = f"{obj_short} ON {surf_short}"
            draw.text((5, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 12

    if frame.get("inside_relations"):
        for obj, container in list(frame["inside_relations"].items())[:3]:
            obj_short = obj.split("_")[0] if "_" in obj else obj
            cont_short = container.split("_")[0] if "_" in container else container
            text = f"{obj_short} IN {cont_short}"
            draw.text((5, y_offset), text, fill=(255, 255, 0), font=font)
            y_offset += 12

    # Save visualized image
    output_filename = f"vis_task{frame['task_id']}_ep{frame['episode_idx']}_step{frame['step_idx']}.png"
    output_path = output_dir / output_filename
    img.save(output_path)

    return stats


def compute_dataset_statistics(data_dir: Path, split: str = "train") -> dict:
    """Compute statistics over the entire dataset."""
    json_path = data_dir / f"{split}.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found")
        return {}

    with open(json_path) as f:
        frames = json.load(f)

    all_stats = {
        "n_frames": len(frames),
        "n_episodes": len(set((f["task_id"], f["episode_idx"]) for f in frames)),
        "class_counts": defaultdict(int),
        "bbox_sizes_by_class": defaultdict(list),
        "spatial_on_counts": defaultdict(int),
        "spatial_inside_counts": defaultdict(int),
        "missing_bboxes": 0,
        "total_objects": 0,
    }

    for frame in frames:
        for obj_id, obj_data in frame["objects"].items():
            obj_class = obj_data.get("class", "unknown")
            bbox = obj_data.get("bbox", [])

            all_stats["total_objects"] += 1
            all_stats["class_counts"][obj_class] += 1

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                if width > 0 and height > 0:
                    all_stats["bbox_sizes_by_class"][obj_class].append(min(width, height))
            else:
                all_stats["missing_bboxes"] += 1

        # Count spatial relations
        for obj, surface in frame.get("on_relations", {}).items():
            all_stats["spatial_on_counts"][surface] += 1
        for obj, container in frame.get("inside_relations", {}).items():
            all_stats["spatial_inside_counts"][container] += 1

    return all_stats


def print_statistics(stats: dict, split: str):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS ({split})")
    print("=" * 60)

    print(f"\nFrames: {stats['n_frames']}")
    print(f"Episodes: {stats['n_episodes']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Missing bboxes: {stats['missing_bboxes']}")

    print(f"\nClass distribution:")
    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    print(f"\nBounding box sizes (min dimension in pixels):")
    for cls, sizes in sorted(stats["bbox_sizes_by_class"].items()):
        if sizes:
            min_size = min(sizes)
            max_size = max(sizes)
            avg_size = sum(sizes) / len(sizes)
            print(f"  {cls}: min={min_size:.1f}, max={max_size:.1f}, avg={avg_size:.1f}")
            if min_size < 16:
                print(f"    ⚠️  WARNING: Some {cls} boxes are <16px (may be too small)")

    print(f"\nSpatial ON relations:")
    for surface, count in sorted(stats["spatial_on_counts"].items(), key=lambda x: -x[1]):
        print(f"  on {surface}: {count}")

    if stats["spatial_inside_counts"]:
        print(f"\nSpatial INSIDE relations:")
        for container, count in sorted(stats["spatial_inside_counts"].items(), key=lambda x: -x[1]):
            print(f"  inside {container}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Perception Data")
    parser.add_argument("--data-dir", type=str, default="data/perception_test")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations (default: data-dir/visualizations)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of frames to visualize")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to visualize (train/val)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute statistics, don't save visualizations")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "visualizations"

    if not args.stats_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    json_path = data_dir / f"{args.split}.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return 1

    with open(json_path) as f:
        frames = json.load(f)

    print(f"Loaded {len(frames)} frames from {json_path}")

    # Compute and print statistics
    stats = compute_dataset_statistics(data_dir, args.split)
    print_statistics(stats, args.split)

    if args.stats_only:
        return 0

    # Visualize sample frames
    print(f"\n{'='*60}")
    print(f"VISUALIZING {min(args.n_samples, len(frames))} FRAMES")
    print("=" * 60)

    # Sample frames evenly across the dataset
    indices = np.linspace(0, len(frames) - 1, min(args.n_samples, len(frames)), dtype=int)

    for i, idx in enumerate(indices):
        frame = frames[idx]
        frame_stats = visualize_frame(data_dir, frame, output_dir)

        print(f"\nFrame {i+1}/{len(indices)}: task{frame['task_id']}_ep{frame['episode_idx']}_step{frame['step_idx']}")
        print(f"  Objects: {frame_stats.get('n_objects', 0)}, Valid bboxes: {frame_stats.get('n_valid_bboxes', 0)}")

        if frame_stats.get("bbox_sizes"):
            min_bbox = min(s["min_dim"] for s in frame_stats["bbox_sizes"])
            max_bbox = max(s["min_dim"] for s in frame_stats["bbox_sizes"])
            print(f"  Bbox sizes: {min_bbox:.1f} - {max_bbox:.1f} px")

    print(f"\nVisualizations saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
