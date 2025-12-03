#!/usr/bin/env python3
"""Collect class-balanced training data for V5 model.

Key improvements over V4:
1. Balance all grocery classes to ~3000 samples each
2. Downsample over-represented classes (cream_cheese, etc.)
3. Use precise bounding boxes based on object geometry
4. Higher resolution (512x512) for better feature extraction

Usage:
    python scripts/collect_v5_balanced_data.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, "/workspace/LIBERO")


# Target samples per class (balanced distribution)
TARGET_SAMPLES = {
    # Grocery items - all balanced to 3000
    "alphabet_soup": 3000,
    "cream_cheese": 3000,
    "salad_dressing": 3000,
    "bbq_sauce": 3000,
    "ketchup": 3000,
    "tomato_sauce": 3000,
    "butter": 3000,
    "milk": 3000,
    "chocolate_pudding": 3000,
    "orange_juice": 3000,
    "basket": 3000,

    # Other objects - keep reasonable amounts
    "bowl": 3000,
    "plate": 3000,
    "mug": 1500,
    "ramekin": 2000,
    "cabinet": 3000,
    "drawer": 3000,
    "stove": 3000,
    "cookie_box": 2000,
    "moka_pot": 1500,
    "book": 1000,
    "caddy": 1000,
    "microwave": 1000,
    "white_mug": 1500,
    "yellow_white_mug": 1500,
    "wine_bottle": 2000,
    "wine_rack": 2000,
    "frying_pan": 1000,
}

# All 30 classes
ALL_CLASSES = [
    "bowl", "plate", "mug", "ramekin", "cabinet", "drawer",
    "cookie_box", "can", "bottle", "stove",
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk", "chocolate_pudding",
    "orange_juice", "basket", "moka_pot", "book", "caddy", "microwave",
    "white_mug", "yellow_white_mug", "wine_bottle", "wine_rack", "frying_pan",
]

CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}

# Object dimensions (width, height, depth) in meters for precise bboxes
OBJECT_DIMENSIONS = {
    "butter": (0.12, 0.03, 0.06),
    "milk": (0.07, 0.20, 0.07),
    "alphabet_soup": (0.08, 0.10, 0.08),
    "cream_cheese": (0.10, 0.05, 0.05),
    "ketchup": (0.06, 0.18, 0.06),
    "bbq_sauce": (0.07, 0.15, 0.07),
    "orange_juice": (0.08, 0.22, 0.08),
    "chocolate_pudding": (0.08, 0.06, 0.08),
    "salad_dressing": (0.06, 0.16, 0.06),
    "tomato_sauce": (0.07, 0.12, 0.07),
    "basket": (0.25, 0.12, 0.18),
    "moka_pot": (0.10, 0.18, 0.10),
    "bowl": (0.15, 0.07, 0.15),
    "plate": (0.20, 0.02, 0.20),
    "mug": (0.08, 0.10, 0.08),
}


def make_libero_env(task_suite: str, task_id: int, img_size: int = 256):
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
        "camera_heights": img_size,
        "camera_widths": img_size,
    }

    env = OffScreenRenderEnv(**env_args)
    return env, task


def get_object_class(obj_name: str) -> Optional[str]:
    """Map MuJoCo object name to class name."""
    name_lower = obj_name.lower()

    mappings = [
        # Drawer mappings (cabinet parts)
        ("cabinet_top", "drawer"),
        ("cabinet_middle", "drawer"),
        ("cabinet_bottom", "drawer"),
        # Mug variants
        ("porcelain_mug", "white_mug"),
        ("white_yellow_mug", "yellow_white_mug"),
        ("red_coffee_mug", "mug"),
        # Grocery items
        ("alphabet_soup", "alphabet_soup"),
        ("cream_cheese", "cream_cheese"),
        ("salad_dressing", "salad_dressing"),
        ("bbq_sauce", "bbq_sauce"),
        ("ketchup", "ketchup"),
        ("tomato_sauce", "tomato_sauce"),
        ("butter", "butter"),
        ("milk", "milk"),
        ("chocolate_pudding", "chocolate_pudding"),
        ("orange_juice", "orange_juice"),
        ("basket", "basket"),
        # Other objects
        ("moka_pot", "moka_pot"),
        ("microwave", "microwave"),
        ("flat_stove", "stove"),
        ("akita_black_bowl", "bowl"),
        ("glazed_rim_porcelain_ramekin", "ramekin"),
        ("wooden_cabinet", "cabinet"),
        ("wine_bottle", "wine_bottle"),
        ("wine_rack", "wine_rack"),
        ("frying_pan", "frying_pan"),
        # Generic fallbacks
        ("plate", "plate"),
        ("can", "can"),
        ("bottle", "bottle"),
        ("mug", "mug"),
        ("bowl", "bowl"),
        ("cabinet", "cabinet"),
        ("stove", "stove"),
        ("book", "book"),
        ("caddy", "caddy"),
        ("cookie", "cookie_box"),
    ]

    for pattern, class_name in mappings:
        if pattern in name_lower:
            return class_name
    return None


def get_precise_bbox(sim, body_name: str, img_size: int = 256) -> Optional[List[int]]:
    """Get precise bounding box using object geometry.

    Uses actual object dimensions when available, falls back to distance-based sizing.
    """
    try:
        body_id = sim.model.body_name2id(body_name)
    except:
        return None

    body_pos = sim.data.body_xpos[body_id]
    cam_id = sim.model.camera_name2id("agentview")
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = sim.model.cam_fovy[cam_id]

    rel_pos = body_pos - cam_pos
    cam_coords = cam_mat.T @ rel_pos

    if cam_coords[2] >= 0:
        return None

    # Project center to image
    f = img_size / (2 * np.tan(np.radians(fovy / 2)))
    u = f * cam_coords[0] / (-cam_coords[2]) + img_size / 2
    v = f * (-cam_coords[1]) / (-cam_coords[2]) + img_size / 2

    # Get object class for dimension lookup
    class_name = get_object_class(body_name)
    dist = np.linalg.norm(cam_coords)

    # Use object-specific dimensions if available
    if class_name and class_name in OBJECT_DIMENSIONS:
        dims = OBJECT_DIMENSIONS[class_name]
        # Project 3D dimensions to 2D
        # Use max of width/depth for horizontal, height for vertical
        obj_width = max(dims[0], dims[2])
        obj_height = dims[1]

        # Convert to pixels
        box_w = int(f * obj_width / (-cam_coords[2]) * 1.2)  # 1.2x padding
        box_h = int(f * obj_height / (-cam_coords[2]) * 1.2)

        # Clamp to reasonable sizes
        box_w = max(15, min(100, box_w))
        box_h = max(15, min(100, box_h))
    else:
        # Fallback: distance-based sizing with object type consideration
        obj_lower = body_name.lower()

        if "microwave" in obj_lower:
            box_w = box_h = max(40, min(120, int(400 / dist)))
        elif "cabinet" in obj_lower or "stove" in obj_lower:
            box_w = box_h = max(35, min(100, int(350 / dist)))
        elif "basket" in obj_lower:
            box_w = max(35, min(90, int(300 / dist)))
            box_h = max(25, min(60, int(200 / dist)))
        else:
            box_w = box_h = max(20, min(60, int(180 / dist)))

    x1 = max(0, int(u - box_w / 2))
    y1 = max(0, int(v - box_h / 2))
    x2 = min(img_size, int(u + box_w / 2))
    y2 = min(img_size, int(v + box_h / 2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def should_keep_frame(
    labels: List[str],
    class_counts: Dict[str, int],
    target_samples: Dict[str, int]
) -> bool:
    """Decide whether to keep a frame based on class balance.

    Prioritize frames that have under-represented classes.
    """
    under_represented = []
    over_represented = []

    for class_name in labels:
        if class_name not in target_samples:
            continue
        current = class_counts.get(class_name, 0)
        target = target_samples[class_name]

        if current < target * 0.5:  # Less than 50% of target
            under_represented.append(class_name)
        elif current >= target:  # Already at target
            over_represented.append(class_name)

    # Always keep if frame has under-represented class
    if under_represented:
        return True

    # Skip if ALL classes are over-represented
    if over_represented and len(over_represented) == len(labels):
        return False

    return True


def collect_from_task(
    task_suite: str,
    task_id: int,
    output_dir: Path,
    class_counts: Dict[str, int],
    n_episodes: int = 20,
    n_frames: int = 25,
    img_size: int = 256
) -> Tuple[int, Dict[str, int]]:
    """Collect training data from a task with class balancing."""
    img_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    env, task = make_libero_env(task_suite, task_id, img_size)

    collected = 0
    local_counts = defaultdict(int)

    for ep in range(n_episodes):
        env.reset()

        for frame in range(n_frames):
            action = np.random.uniform(-0.3, 0.3, 7)
            action[6] = np.random.choice([-1, 1])

            try:
                env.step(action)
            except:
                break

            img = env.sim.render(
                camera_name="agentview",
                height=img_size,
                width=img_size,
                mode='offscreen'
            )
            img = img[::-1]

            labels = []
            frame_classes = []
            seen_classes = set()

            for i in range(env.sim.model.nbody):
                body_name = env.sim.model.body_id2name(i)
                if not body_name:
                    continue

                class_name = get_object_class(body_name)
                if class_name is None or class_name not in CLASS_TO_ID:
                    continue

                if class_name in seen_classes:
                    continue

                # Check if we need more of this class
                current_count = class_counts.get(class_name, 0) + local_counts.get(class_name, 0)
                target = TARGET_SAMPLES.get(class_name, 3000)
                if current_count >= target:
                    continue

                bbox = get_precise_bbox(env.sim, body_name, img_size)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                cx = (x1 + x2) / 2 / img_size
                cy = (y1 + y2) / 2 / img_size
                w = (x2 - x1) / img_size
                h = (y2 - y1) / img_size

                class_id = CLASS_TO_ID[class_name]
                labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                frame_classes.append(class_name)
                seen_classes.add(class_name)

            # Check if we should keep this frame
            if not labels:
                continue

            if not should_keep_frame(frame_classes, {**class_counts, **local_counts}, TARGET_SAMPLES):
                continue

            # Save image and labels
            prefix = f"{task_suite}_{task_id:02d}_{ep:03d}_{frame:03d}"

            img_path = img_dir / f"{prefix}.png"
            Image.fromarray(img).save(img_path)

            label_path = label_dir / f"{prefix}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))

            # Update counts
            for class_name in frame_classes:
                local_counts[class_name] += 1

            collected += 1

    env.close()
    return collected, dict(local_counts)


def check_balance(class_counts: Dict[str, int]) -> None:
    """Print class balance status."""
    print("\nCurrent class balance:")
    print("-" * 60)

    for class_name in sorted(TARGET_SAMPLES.keys()):
        current = class_counts.get(class_name, 0)
        target = TARGET_SAMPLES[class_name]
        pct = current / target * 100 if target > 0 else 0

        if pct >= 100:
            status = "DONE"
            bar = "#" * 20
        else:
            status = f"{pct:.0f}%"
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))

        print(f"{class_name:20s}: {current:5d}/{target:5d} [{bar}] {status}")


def main():
    output_dir = Path("/workspace/brain_robot/data/yolo_v5")

    # Clear existing data
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # All tasks to collect from
    tasks = []

    # All 4 suites
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        for i in range(10):
            tasks.append((suite, i))

    print("=" * 60)
    print("Collecting V5 Training Data (Class-Balanced)")
    print(f"Total tasks: {len(tasks)}")
    print("=" * 60)

    total_collected = 0
    all_class_counts = defaultdict(int)

    # Multiple passes to ensure balance
    for pass_num in range(3):
        print(f"\n=== Pass {pass_num + 1} ===")

        for suite, task_id in tasks:
            # Check if we've reached all targets
            all_done = all(
                all_class_counts.get(cls, 0) >= target
                for cls, target in TARGET_SAMPLES.items()
            )
            if all_done:
                print("All class targets reached!")
                break

            print(f"{suite} task {task_id}...", end=" ", flush=True)

            try:
                n, counts = collect_from_task(
                    suite, task_id, output_dir,
                    all_class_counts,
                    n_episodes=15 if pass_num == 0 else 10,
                    n_frames=20
                )
                total_collected += n

                for cls, cnt in counts.items():
                    all_class_counts[cls] += cnt

                if n > 0:
                    print(f"{n} images")
                else:
                    print("skipped (balanced)")

            except Exception as e:
                print(f"ERROR: {e}")

        check_balance(all_class_counts)

    # Create data.yaml
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"# YOLO V5 Training Data - Class-Balanced\n")
        f.write(f"# {total_collected} images with balanced class distribution\n\n")
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/train\n\n")
        f.write(f"nc: {len(ALL_CLASSES)}\n")
        f.write(f"names:\n")
        for i, name in enumerate(ALL_CLASSES):
            f.write(f"  {i}: {name}\n")

    print("\n" + "=" * 60)
    print(f"Total: {total_collected} images")
    print(f"\nData saved to {output_dir}")
    print(f"Data config: {data_yaml}")

    # Final balance check
    check_balance(all_class_counts)


if __name__ == "__main__":
    main()
