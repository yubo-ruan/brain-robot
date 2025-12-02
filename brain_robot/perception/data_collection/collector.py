"""Perception data collector for training learned perception models.

Collects RGB images with ground-truth labels from oracle perception
during episode execution.
"""

import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..interface import PerceptionResult


# LIBERO object classes for detection
LIBERO_OBJECT_CLASSES = [
    "bowl",
    "plate",
    "mug",
    "ramekin",
    "cabinet",
    "drawer",
    "cookie_box",
    "can",
    "bottle",
    "stove",
]


def instance_id_to_class(instance_id: str) -> str:
    """Extract object class from LIBERO instance ID.

    Examples:
        "akita_black_bowl_1_main" → "bowl"
        "plate_1_main" → "plate"
        "cookies_1_main" → "cookie_box"
        "wooden_cabinet_1_base" → "cabinet"
    """
    name_lower = instance_id.lower()

    # Special mappings
    if "cookies" in name_lower:
        return "cookie_box"

    # Check each class
    for cls in LIBERO_OBJECT_CLASSES:
        if cls in name_lower:
            return cls

    return "unknown"


@dataclass
class PerceptionDataPoint:
    """Single training example for learned perception."""

    # Metadata (for episode-level splitting)
    task_id: int
    task_suite: str
    episode_idx: int
    step_idx: int
    timestamp: float
    camera_name: str

    # Image path (stored separately to save memory)
    image_path: str = ""

    # Ground truth from oracle
    # Objects: {instance_id: {"pose": [7D], "class": str, "bbox": [4D]}}
    objects: Dict[str, dict] = field(default_factory=dict)

    # Gripper state
    gripper_pose: Optional[List[float]] = None
    gripper_width: float = 0.0

    # Spatial relations
    on_relations: Dict[str, str] = field(default_factory=dict)
    inside_relations: Dict[str, str] = field(default_factory=dict)

    # Episode context
    task_description: str = ""
    episode_success: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "task_id": self.task_id,
            "task_suite": self.task_suite,
            "episode_idx": self.episode_idx,
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "camera_name": self.camera_name,
            "image_path": self.image_path,
            "objects": self.objects,
            "gripper_pose": self.gripper_pose,
            "gripper_width": self.gripper_width,
            "on_relations": self.on_relations,
            "inside_relations": self.inside_relations,
            "task_description": self.task_description,
            "episode_success": self.episode_success,
        }


class PerceptionDataCollector:
    """Collect perception training data during episode execution.

    Usage:
        collector = PerceptionDataCollector("data/perception")

        for episode in episodes:
            collector.start_episode(task_id, episode_idx, task_description)

            for step in episode_steps:
                oracle_result = oracle.perceive(env)
                collector.collect_frame(env, oracle_result, step)

            collector.end_episode(success=True)

        collector.save_dataset()
    """

    def __init__(
        self,
        output_dir: str,
        task_suite: str = "libero_spatial",
        cameras: List[str] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_suite = task_suite
        self.cameras = cameras or ["agentview"]
        self.image_size = image_size

        # Storage
        self.data_points: List[PerceptionDataPoint] = []

        # Current episode state
        self._current_task_id: Optional[int] = None
        self._current_episode_idx: Optional[int] = None
        self._current_task_description: str = ""
        self._episode_data_points: List[PerceptionDataPoint] = []

        # Image storage
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # Statistics
        self.stats = {
            "total_frames": 0,
            "total_episodes": 0,
            "frames_per_task": defaultdict(int),
        }

    def start_episode(
        self,
        task_id: int,
        episode_idx: int,
        task_description: str = ""
    ):
        """Start collecting data for a new episode."""
        self._current_task_id = task_id
        self._current_episode_idx = episode_idx
        self._current_task_description = task_description
        self._episode_data_points = []

    def collect_frame(
        self,
        env,
        oracle_result: PerceptionResult,
        step_idx: int,
    ):
        """Collect a single frame with ground truth labels.

        Args:
            env: LIBERO environment
            oracle_result: Ground truth from oracle perception
            step_idx: Current step in episode
        """
        if self._current_task_id is None:
            raise RuntimeError("Must call start_episode() before collect_frame()")

        timestamp = time.time()

        for camera in self.cameras:
            # Render image
            try:
                rgb = env.sim.render(
                    width=self.image_size[0],
                    height=self.image_size[1],
                    camera_name=camera,
                )
                # Flip vertically (MuJoCo renders upside down)
                rgb = rgb[::-1, :, :]
            except Exception as e:
                print(f"Warning: Failed to render {camera}: {e}")
                continue

            # Save image
            image_filename = (
                f"task{self._current_task_id}_"
                f"ep{self._current_episode_idx}_"
                f"step{step_idx}_{camera}.png"
            )
            image_path = self.images_dir / image_filename
            self._save_image(rgb, image_path)

            # Build object annotations
            objects = {}
            for instance_id, pose in oracle_result.objects.items():
                obj_class = instance_id_to_class(instance_id)

                # Compute 2D bounding box via projection
                bbox = self._compute_bbox(env, pose[:3], camera)

                objects[instance_id] = {
                    "pose": pose.tolist() if isinstance(pose, np.ndarray) else pose,
                    "class": obj_class,
                    "bbox": bbox,
                }

            # Create data point
            data_point = PerceptionDataPoint(
                task_id=self._current_task_id,
                task_suite=self.task_suite,
                episode_idx=self._current_episode_idx,
                step_idx=step_idx,
                timestamp=timestamp,
                camera_name=camera,
                image_path=str(image_path.relative_to(self.output_dir)),
                objects=objects,
                gripper_pose=(
                    oracle_result.gripper_pose.tolist()
                    if oracle_result.gripper_pose is not None
                    else None
                ),
                gripper_width=oracle_result.gripper_width,
                on_relations=dict(oracle_result.on),
                inside_relations=dict(oracle_result.inside),
                task_description=self._current_task_description,
            )

            self._episode_data_points.append(data_point)
            self.stats["total_frames"] += 1
            self.stats["frames_per_task"][self._current_task_id] += 1

    def end_episode(self, success: bool):
        """End current episode and mark success/failure."""
        # Mark all frames with episode outcome
        for dp in self._episode_data_points:
            dp.episode_success = success

        # Add to main collection
        self.data_points.extend(self._episode_data_points)
        self.stats["total_episodes"] += 1

        # Reset
        self._current_task_id = None
        self._current_episode_idx = None
        self._episode_data_points = []

    def _save_image(self, rgb: np.ndarray, path: Path):
        """Save RGB image to disk."""
        from PIL import Image
        img = Image.fromarray(rgb.astype(np.uint8))
        img.save(path)

    def _compute_bbox(
        self,
        env,
        position_3d: np.ndarray,
        camera_name: str,
        object_size: float = 0.05,
    ) -> List[float]:
        """Project 3D position to 2D bounding box.

        Uses MuJoCo camera projection with estimated object size.

        Returns:
            [x1, y1, x2, y2] in pixel coordinates, or empty list if off-screen.
        """
        try:
            # Get camera matrix from MuJoCo
            sim = env.sim

            # Find camera ID
            camera_id = sim.model.camera_name2id(camera_name)

            # Get camera properties
            fovy = sim.model.cam_fovy[camera_id]

            # Camera pose in world frame
            cam_pos = sim.data.cam_xpos[camera_id]
            cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3)

            # Transform point to camera frame
            point_world = np.array(position_3d)
            point_cam = cam_mat.T @ (point_world - cam_pos)

            # Check if point is behind camera
            if point_cam[2] >= 0:
                return []

            # Project to image plane
            # MuJoCo uses OpenGL conventions: -Z is forward
            f = self.image_size[1] / (2 * np.tan(np.radians(fovy) / 2))

            u = f * point_cam[0] / (-point_cam[2]) + self.image_size[0] / 2
            v = f * point_cam[1] / (-point_cam[2]) + self.image_size[1] / 2

            # Estimate box size based on distance
            dist = -point_cam[2]
            box_pixels = f * object_size / dist

            # Clamp to image bounds
            x1 = max(0, u - box_pixels)
            y1 = max(0, v - box_pixels)
            x2 = min(self.image_size[0], u + box_pixels)
            y2 = min(self.image_size[1], v + box_pixels)

            # Check if box is visible
            if x2 <= x1 or y2 <= y1:
                return []

            return [float(x1), float(y1), float(x2), float(y2)]

        except Exception as e:
            # Projection failed - return empty bbox
            return []

    def save_dataset(self, split_ratio: float = 0.9, seed: int = 42):
        """Save collected data with episode-level train/val split.

        CRITICAL: Splits at episode level to avoid data leakage.
        """
        random.seed(seed)

        # Group frames by episode
        episodes = defaultdict(list)
        for dp in self.data_points:
            key = (dp.task_id, dp.episode_idx)
            episodes[key].append(dp)

        # Shuffle and split at episode level
        episode_keys = list(episodes.keys())
        random.shuffle(episode_keys)
        split = int(len(episode_keys) * split_ratio)

        train_keys = episode_keys[:split]
        val_keys = episode_keys[split:]

        # Save train set
        train_data = []
        for key in train_keys:
            for dp in episodes[key]:
                train_data.append(dp.to_dict())

        # Save val set
        val_data = []
        for key in val_keys:
            for dp in episodes[key]:
                val_data.append(dp.to_dict())

        # Write to disk
        with open(self.output_dir / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        with open(self.output_dir / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)

        # Save metadata
        metadata = {
            "task_suite": self.task_suite,
            "cameras": self.cameras,
            "image_size": self.image_size,
            "split_ratio": split_ratio,
            "seed": seed,
            "n_train_episodes": len(train_keys),
            "n_val_episodes": len(val_keys),
            "n_train_frames": len(train_data),
            "n_val_frames": len(val_data),
            "object_classes": LIBERO_OBJECT_CLASSES,
            "stats": dict(self.stats),
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {self.output_dir}")
        print(f"  Train: {len(train_keys)} episodes, {len(train_data)} frames")
        print(f"  Val: {len(val_keys)} episodes, {len(val_data)} frames")
        print(f"  Object classes: {LIBERO_OBJECT_CLASSES}")

    def summary(self) -> dict:
        """Get collection statistics."""
        return {
            "total_frames": self.stats["total_frames"],
            "total_episodes": self.stats["total_episodes"],
            "frames_per_task": dict(self.stats["frames_per_task"]),
            "cameras": self.cameras,
        }
