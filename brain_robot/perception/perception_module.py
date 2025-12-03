"""Unified Perception Module for real-time object detection and tracking.

Combines YOLO detection with depth-based 3D estimation to provide
real-time object positions. This replaces oracle perception for skills.

Architecture:
    RGB Image + Depth Image
           ↓
    YOLO Detector (2D bbox + class)
           ↓
    Depth Position Estimator (3D position)
           ↓
    Object Tracker (persistent IDs)
           ↓
    WorldState Update

Usage:
    perception = PerceptionModule.from_config(model_path="models/yolo_libero_v4.pt")

    # Initialize with environment
    perception.initialize(env)

    # Each step
    objects = perception.update(obs)  # Returns {instance_id: position}
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .detection.yolo_detector import YOLOObjectDetector
from .depth_estimation import DepthPositionEstimator, CameraInfo, extract_object_pointcloud
from .tracking.interface import Detection, ObjectTrack, TrackingResult


@dataclass
class PerceptionConfig:
    """Configuration for perception module."""

    # YOLO config
    yolo_model_path: str = "models/yolo_libero_v4.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "cuda"

    # Depth config
    camera_name: str = "agentview"
    depth_percentile: float = 25.0

    # Tracking config
    association_threshold: float = 0.15  # Max distance (m) to associate detection to track
    track_max_age: int = 30  # Frames before deleting lost track
    track_min_hits: int = 3  # Min detections before track is reliable

    # Bootstrap config
    use_oracle_bootstrap: bool = True  # Use oracle positions for initial association
    bootstrap_frames: int = 5  # Number of frames for bootstrap


class SimpleObjectTracker:
    """Simple nearest-neighbor tracker for LIBERO objects.

    Designed for the relatively static LIBERO environment where:
    - Objects don't move much (except when manipulated)
    - We have oracle bootstrap (initial positions from sim)
    - Main challenge is maintaining instance IDs

    Uses Hungarian algorithm for optimal assignment.
    """

    def __init__(
        self,
        association_threshold: float = 0.15,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.association_threshold = association_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: Dict[str, ObjectTrack] = {}
        self.next_track_id = 0
        self.frame_count = 0

    def initialize(self, known_objects: Dict[str, np.ndarray]):
        """Initialize with known objects from oracle.

        Args:
            known_objects: {instance_id: position}
        """
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0

        for instance_id, position in known_objects.items():
            # Infer class from instance ID (e.g., "akita_black_bowl_1_main" -> "bowl")
            class_name = self._infer_class_from_id(instance_id)

            track = ObjectTrack(
                track_id=self.next_track_id,
                instance_id=instance_id,
                class_name=class_name,
                position=np.array(position),
                hits=self.min_hits,  # Start as reliable
                created_at=0.0,
            )
            self.tracks[instance_id] = track
            self.next_track_id += 1

    def _infer_class_from_id(self, instance_id: str) -> str:
        """Infer class name from LIBERO instance ID."""
        id_lower = instance_id.lower()

        # Common patterns
        class_patterns = [
            "bowl", "plate", "mug", "cup", "ramekin",
            "drawer", "cabinet", "stove",
            "cream_cheese", "butter", "milk", "ketchup",
            "alphabet_soup", "salad_dressing", "bbq_sauce",
            "tomato_sauce", "chocolate_pudding", "orange_juice",
            "basket", "moka_pot", "book", "caddy", "microwave",
            "wine_bottle", "wine_rack", "frying_pan",
            "cookie", "can", "bottle",
        ]

        for pattern in class_patterns:
            if pattern in id_lower:
                return pattern

        return "unknown"

    def update(
        self,
        detections: List[Detection],
        timestamp: float = 0.0,
    ) -> TrackingResult:
        """Update tracks with new detections.

        Uses greedy nearest-neighbor matching by class.
        """
        self.frame_count += 1
        result = TrackingResult(timestamp=timestamp, n_detections=len(detections))

        # Group detections by class for matching
        detections_by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            if det.class_name not in detections_by_class:
                detections_by_class[det.class_name] = []
            detections_by_class[det.class_name].append(det)

        matched_tracks = set()
        matched_detections = set()

        # Match detections to existing tracks
        for instance_id, track in self.tracks.items():
            if not track.is_active:
                continue

            # Get detections of same class
            class_detections = detections_by_class.get(track.class_name, [])

            # Find nearest detection
            best_det = None
            best_dist = float('inf')
            best_idx = -1

            for idx, det in enumerate(class_detections):
                if idx in matched_detections:
                    continue

                dist = np.linalg.norm(det.position - track.position)
                if dist < best_dist and dist < self.association_threshold:
                    best_dist = dist
                    best_det = det
                    best_idx = idx

            if best_det is not None:
                # Update track with detection
                # Simple exponential moving average for position
                alpha = 0.7
                track.position = alpha * best_det.position + (1 - alpha) * track.position
                track.confidence = best_det.confidence
                track.hits += 1
                track.misses = 0
                track.last_seen = timestamp
                matched_tracks.add(instance_id)
                matched_detections.add(best_idx)
                result.n_matched += 1

        # Handle unmatched tracks (increase miss count)
        for instance_id, track in self.tracks.items():
            if instance_id not in matched_tracks and track.is_active:
                track.misses += 1
                track.confidence *= 0.95  # Decay confidence

                if track.misses > self.max_age:
                    track.is_active = False
                    result.n_lost_tracks += 1

        # For LIBERO, we don't create new tracks from unmatched detections
        # since all objects are known at initialization
        # (Could add this for unknown object handling)

        result.tracks = list(self.tracks.values())
        return result

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0

    def get_position(self, instance_id: str) -> Optional[np.ndarray]:
        """Get current position of tracked object."""
        track = self.tracks.get(instance_id)
        if track and track.is_active:
            return track.position.copy()
        return None

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get all active object positions."""
        return {
            tid: track.position.copy()
            for tid, track in self.tracks.items()
            if track.is_active
        }


class PerceptionModule:
    """Unified perception module combining detection, depth, and tracking.

    Provides real-time object positions for skill execution.
    Can operate in oracle mode (ground truth) or learned mode (YOLO + depth).
    """

    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        use_oracle: bool = False,
    ):
        """Initialize perception module.

        Args:
            config: Perception configuration
            use_oracle: If True, use oracle positions from simulator
        """
        self.config = config or PerceptionConfig()
        self.use_oracle = use_oracle

        # Components
        self.detector: Optional[YOLOObjectDetector] = None
        self.depth_estimator: Optional[DepthPositionEstimator] = None
        self.tracker: Optional[SimpleObjectTracker] = None

        # State
        self.sim = None
        self.camera_info: Optional[CameraInfo] = None
        self.known_objects: Dict[str, str] = {}  # instance_id -> class_name
        self.initialized = False

        if not use_oracle:
            self._init_learned_components()

    def _init_learned_components(self):
        """Initialize learned perception components."""
        # YOLO detector
        self.detector = YOLOObjectDetector(
            model_path=self.config.yolo_model_path,
            confidence_threshold=self.config.yolo_confidence,
            device=self.config.yolo_device,
        )

        # Depth estimator
        self.depth_estimator = DepthPositionEstimator(
            camera_name=self.config.camera_name,
            depth_percentile=self.config.depth_percentile,
        )

        # Tracker
        self.tracker = SimpleObjectTracker(
            association_threshold=self.config.association_threshold,
            max_age=self.config.track_max_age,
            min_hits=self.config.track_min_hits,
        )

    def initialize(self, env, known_objects: Optional[Dict[str, str]] = None):
        """Initialize perception with environment.

        Args:
            env: LIBERO environment
            known_objects: Optional {instance_id: class_name} mapping
        """
        self.sim = env.sim

        # Extract known objects from environment if not provided
        if known_objects is None:
            known_objects = self._extract_known_objects(env)

        self.known_objects = known_objects

        # Get initial positions for tracker bootstrap
        if not self.use_oracle and self.tracker:
            initial_positions = {}
            for instance_id in known_objects.keys():
                pos = self._get_oracle_position(instance_id)
                if pos is not None:
                    initial_positions[instance_id] = pos

            self.tracker.initialize(initial_positions)

        # Update camera info
        if self.depth_estimator:
            self.depth_estimator.update_camera_info(self.sim)
            self.camera_info = self.depth_estimator.camera_info

        self.initialized = True

    def _extract_known_objects(self, env) -> Dict[str, str]:
        """Extract known object IDs from environment."""
        known = {}

        # Get all body names from simulation
        for i in range(self.sim.model.nbody):
            body_name = self.sim.model.body_id2name(i)
            if body_name is None:
                continue

            # Check if it's a manipulable object (has "_main" suffix typically)
            name_lower = body_name.lower()

            # Common LIBERO object patterns
            for obj_type in ["bowl", "plate", "mug", "cup", "ramekin",
                             "cream_cheese", "butter", "milk", "ketchup",
                             "cookie", "can", "bottle"]:
                if obj_type in name_lower:
                    known[body_name] = obj_type
                    break

        return known

    def _get_oracle_position(self, instance_id: str) -> Optional[np.ndarray]:
        """Get ground truth position from simulator."""
        try:
            # Try exact name
            try:
                body_id = self.sim.model.body_name2id(instance_id)
            except ValueError:
                # Try with/without _main suffix
                if instance_id.endswith("_main"):
                    body_id = self.sim.model.body_name2id(instance_id[:-5])
                else:
                    body_id = self.sim.model.body_name2id(instance_id + "_main")

            return self.sim.data.body_xpos[body_id].copy()

        except (ValueError, AttributeError):
            return None

    def update(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Update perception with new observation.

        Args:
            obs: Observation dict from environment

        Returns:
            {instance_id: position} for all tracked objects
        """
        if self.use_oracle:
            return self._update_oracle()
        else:
            return self._update_learned(obs)

    def _update_oracle(self) -> Dict[str, np.ndarray]:
        """Update using oracle (ground truth) positions."""
        positions = {}
        for instance_id in self.known_objects.keys():
            pos = self._get_oracle_position(instance_id)
            if pos is not None:
                positions[instance_id] = pos
        return positions

    def _update_learned(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Update using learned perception (YOLO + depth)."""
        if not self.initialized or self.detector is None:
            return {}

        # Get images from observation
        rgb = obs.get("agentview_image")
        depth = obs.get("agentview_depth")

        if rgb is None or depth is None:
            # Fall back to oracle if no images
            return self._update_oracle()

        # Detect objects
        detections = self.detector.detect(rgb)

        # Estimate 3D positions from depth
        if self.depth_estimator:
            detections = self.depth_estimator.estimate_positions(
                detections, depth, self.sim
            )

        # Update tracker
        if self.tracker:
            result = self.tracker.update(detections)
            return self.tracker.get_all_positions()

        return {}

    def get_position(self, instance_id: str) -> Optional[np.ndarray]:
        """Get position of specific object.

        Args:
            instance_id: LIBERO instance ID

        Returns:
            3D position or None if not found
        """
        if self.use_oracle:
            return self._get_oracle_position(instance_id)
        elif self.tracker:
            return self.tracker.get_position(instance_id)
        return None

    def get_pointcloud(
        self,
        instance_id: str,
        obs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract point cloud for an object.

        Useful for GIGA grasp prediction.

        Args:
            instance_id: Object instance ID
            obs: Current observation

        Returns:
            Tuple of (points, colors)
        """
        if self.depth_estimator is None or self.camera_info is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Get bbox for this object from tracker
        if self.tracker and instance_id in self.tracker.tracks:
            track = self.tracker.tracks[instance_id]
            # Project 3D position to get approximate bbox
            # (This is a simplification - ideally we'd cache YOLO bboxes)
            bbox = self._project_position_to_bbox(track.position, obs)
        else:
            # Use full image if no bbox
            bbox = [0, 0, self.camera_info.width, self.camera_info.height]

        rgb = obs.get("agentview_image")
        depth = obs.get("agentview_depth")

        if rgb is None or depth is None:
            return np.zeros((0, 3)), np.zeros((0, 3))

        return extract_object_pointcloud(
            bbox, depth, rgb, self.camera_info
        )

    def _project_position_to_bbox(
        self,
        position: np.ndarray,
        obs: Dict[str, Any],
        box_size: int = 40,
    ) -> List[float]:
        """Project 3D position to 2D bbox (approximate)."""
        if self.camera_info is None or self.camera_info.extrinsic is None:
            return [0, 0, box_size, box_size]

        # Transform to camera frame
        T_cam_world = np.linalg.inv(self.camera_info.extrinsic)
        pos_homo = np.array([*position, 1.0])
        pos_cam = T_cam_world @ pos_homo

        # Project to image
        if pos_cam[2] > 0:
            u = self.camera_info.fx * pos_cam[0] / pos_cam[2] + self.camera_info.cx
            v = self.camera_info.fy * pos_cam[1] / pos_cam[2] + self.camera_info.cy

            # Create bbox around projected point
            half = box_size // 2
            return [
                max(0, u - half),
                max(0, v - half),
                min(self.camera_info.width, u + half),
                min(self.camera_info.height, v + half),
            ]

        return [0, 0, box_size, box_size]

    def reset(self):
        """Reset perception state for new episode."""
        if self.tracker:
            self.tracker.reset()
        self.initialized = False

    @classmethod
    def from_config(
        cls,
        model_path: str = "models/yolo_libero_v4.pt",
        use_oracle: bool = False,
        **kwargs,
    ) -> "PerceptionModule":
        """Create perception module from config.

        Args:
            model_path: Path to YOLO model
            use_oracle: Use oracle perception
            **kwargs: Additional config options

        Returns:
            PerceptionModule instance
        """
        config = PerceptionConfig(yolo_model_path=model_path, **kwargs)
        return cls(config=config, use_oracle=use_oracle)
