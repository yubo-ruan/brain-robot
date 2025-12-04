"""Learned perception using detection + depth-based pose estimation.

This is the pragmatic "Option B" approach:
- Detection for 2D detection (class + bbox)
- Depth image for 3D position (bbox center → depth lookup → 3D)
- Tracker for instance ID persistence
- Reuse oracle's spatial relation computation

Supported detectors:
- yolo: YOLO-based detection (fast, requires training)
- gdino: Grounding-DINO (open-vocabulary, ~120ms)
- gsam: Grounded-SAM (open-vocabulary + masks, ~130ms)

Why depth-based instead of learned pose regression:
1. Simpler - no additional training required
2. Depth is already available from LIBERO
3. Works well for tabletop (objects at similar heights)
4. Can always upgrade to learned pose later
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .interface import PerceptionInterface, PerceptionResult
from .oracle import OraclePerception
from .detection.yolo_detector import YOLOObjectDetector
from .tracking.interface import Detection, TrackingResult
from .tracking.nearest_neighbor import NearestNeighborTracker


@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics for depth projection."""

    # Image dimensions
    width: int = 256
    height: int = 256

    # Camera pose (updated each frame from sim)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Field of view (vertical)
    fovy: float = 45.0

    @property
    def focal_length(self) -> float:
        """Compute focal length in pixels from FOV."""
        return self.height / (2 * np.tan(np.radians(self.fovy) / 2))

    @property
    def cx(self) -> float:
        """Principal point x (image center)."""
        return self.width / 2

    @property
    def cy(self) -> float:
        """Principal point y (image center)."""
        return self.height / 2


class LearnedPerception(PerceptionInterface):
    """Learned perception using YOLO + depth.

    Architecture:
        RGB image → YOLO detector → 2D detections (class, bbox)
                                          ↓
        Depth image + bbox center → 3D position
                                          ↓
        3D detections → Tracker → instance IDs
                                          ↓
        Object poses → Spatial relations → PerceptionResult

    Usage:
        perception = LearnedPerception(model_path="models/yolo_libero.pt")

        # At episode start, bootstrap tracker with oracle knowledge
        perception.bootstrap_from_oracle(oracle_result, env)

        # Each frame
        result = perception.perceive(env)
    """

    def __init__(
        self,
        model_path: str = "models/yolo_libero.pt",
        confidence_threshold: float = 0.5,
        depth_patch_size: int = 5,  # Size of patch for robust depth sampling
        use_median_depth: bool = True,  # Use median vs mean for depth
        camera_name: str = "agentview",
        image_size: Tuple[int, int] = (256, 256),
        detector_type: str = "yolo",  # yolo, gdino, gsam
        target_objects: Optional[List[str]] = None,  # For open-vocab detectors
    ):
        """Initialize learned perception.

        Args:
            model_path: Path to trained YOLO model (only used if detector_type="yolo")
            confidence_threshold: Detection confidence threshold
            depth_patch_size: Size of patch around bbox center for depth sampling
            use_median_depth: Use median (robust) vs mean for depth
            camera_name: Camera to use for perception
            image_size: Rendering resolution
            detector_type: Type of detector ("yolo", "gdino", "gsam")
            target_objects: List of target object names for open-vocab detectors
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.depth_patch_size = depth_patch_size
        self.use_median_depth = use_median_depth
        self.camera_name = camera_name
        self.image_size = image_size
        self.detector_type = detector_type
        self.target_objects = target_objects or []

        # Initialize detector based on type
        self.detector = self._create_detector(detector_type, model_path, confidence_threshold)

        # Initialize tracker
        # More tolerant settings for robustness:
        # - Higher association threshold (objects can move between frames)
        # - More max_misses (detector may miss objects for several frames)
        # - For open-vocab detectors (gsam/gdino), preserve bootstrap positions
        #   since depth-based 3D is less calibrated
        self.tracker = NearestNeighborTracker(
            association_threshold=0.20,  # 20cm for association
            max_misses=30,  # Allow 30 misses before marking inactive
            min_confidence=0.05,  # Lower confidence threshold to keep tracks alive
            preserve_bootstrap_positions=(detector_type in ["gsam", "gdino"]),
        )

        # Camera parameters (updated from sim)
        self.camera_params = CameraParams(
            width=image_size[0],
            height=image_size[1],
        )

        # For spatial relations - reuse oracle logic
        self._oracle = OraclePerception()

        # State
        self._bootstrapped = False
        self._last_perceive_time = 0.0

        # Depth conversion scale (calibrated from environment)
        # Default value for LIBERO: extent * znear ≈ 0.011
        self._depth_scale = 0.011

    def _create_detector(self, detector_type: str, model_path: str, confidence_threshold: float):
        """Create detector based on type.

        Args:
            detector_type: Type of detector ("yolo", "gdino", "gsam")
            model_path: Path to YOLO model (only used for yolo)
            confidence_threshold: Detection confidence threshold

        Returns:
            Detector instance implementing detect() method
        """
        if detector_type == "yolo":
            return YOLOObjectDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
            )
        elif detector_type == "gdino":
            from .detection.grounding_dino_detector import GroundingDINODetector
            detector = GroundingDINODetector(
                box_threshold=confidence_threshold,
            )
            # Set targets if provided
            if self.target_objects:
                detector.set_target_objects(self.target_objects)
            return detector
        elif detector_type == "gsam":
            from .detection.grounded_sam_detector import GroundedSAMDetector
            detector = GroundedSAMDetector(
                box_threshold=confidence_threshold,
            )
            # Set targets if provided
            if self.target_objects:
                detector.set_target_objects(self.target_objects)
            return detector
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Use 'yolo', 'gdino', or 'gsam'.")

    def set_target_objects(self, target_objects: List[str]):
        """Set target objects for open-vocabulary detectors.

        Args:
            target_objects: List of object names to detect (e.g., ["bbq_sauce", "basket"])
        """
        self.target_objects = target_objects
        if self.detector_type in ["gdino", "gsam"]:
            self.detector.set_target_objects(target_objects)

    def bootstrap_from_oracle(
        self,
        oracle_result: PerceptionResult,
        env,
    ):
        """Bootstrap tracker with oracle knowledge at episode start.

        This is the "initialization" phase where we use privileged info
        to establish instance IDs. After this, we track using detections.

        Args:
            oracle_result: Ground truth from oracle.perceive(env)
            env: LIBERO environment (for camera params)
        """
        # Extract positions for tracker initialization
        known_objects = {}
        for instance_id, pose in oracle_result.objects.items():
            known_objects[instance_id] = pose[:3]  # Position only

        # Initialize tracker
        self.tracker.initialize(known_objects)

        # Update camera params from env
        self._update_camera_params(env)

        self._bootstrapped = True

    def bootstrap_from_detections(self, env):
        """Cold-start bootstrap using only detections (no oracle).

        This is an alternative to bootstrap_from_oracle that doesn't use
        privileged information. It runs detection on the first frame and
        assigns instance IDs based on class + position.

        Args:
            env: LIBERO environment
        """
        # Update camera params
        self._update_camera_params(env)

        # Get sim
        sim = self._get_sim(env)
        if sim is None:
            return

        # Render and detect
        rgb, depth = self._render_images(sim)
        if rgb is None:
            return

        detections_2d = self.detector.detect(rgb)
        detections_3d = self._detections_to_3d(detections_2d, depth)

        # Assign instance IDs based on class + index
        # e.g., "bowl_0", "bowl_1", "plate_0"
        class_counts = {}
        known_objects = {}

        for det in detections_3d:
            cls = det.class_name
            idx = class_counts.get(cls, 0)
            class_counts[cls] = idx + 1

            # Generate synthetic instance ID
            instance_id = f"{cls}_{idx}_learned"
            known_objects[instance_id] = det.position

        # Initialize tracker with detected objects
        self.tracker.initialize(known_objects)
        self._bootstrapped = True

    def perceive(self, env) -> PerceptionResult:
        """Run learned perception pipeline.

        Args:
            env: LIBERO environment

        Returns:
            PerceptionResult with detected object poses
        """
        result = PerceptionResult(timestamp=time.time())

        if not self._bootstrapped:
            # Fall back to oracle if not bootstrapped
            return self._oracle.perceive(env)

        # Get sim and base env
        sim = self._get_sim(env)
        base_env = self._get_base_env(env)

        if sim is None:
            return result

        # Update camera params
        self._update_camera_params(env)

        # 1. Render RGB and depth
        rgb, depth = self._render_images(sim)

        if rgb is None:
            return result

        # 2. Run YOLO detection
        detections_2d = self.detector.detect(rgb)

        # 3. Convert 2D detections to 3D using depth
        detections_3d = self._detections_to_3d(detections_2d, depth)

        # 4. Update tracker
        tracking_result = self.tracker.update(
            detections_3d,
            timestamp=result.timestamp
        )

        # 5. Build object poses from tracks
        result.objects = {}
        result.object_names = []

        for track in tracking_result.get_active_tracks():
            # Use identity quaternion for orientation (depth doesn't give us this)
            pose = np.concatenate([
                track.position,
                np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            ])
            result.objects[track.instance_id] = pose
            result.object_names.append(track.instance_id)

        # 6. Compute spatial relations using oracle's logic
        result.on, result.inside = self._oracle._extract_spatial_relations(result.objects)

        # 7. Extract gripper state (still use oracle for this - it's proprioception)
        result.gripper_pose = self._oracle._extract_gripper_pose(sim, base_env, env)
        result.gripper_width = self._oracle._extract_gripper_width(sim, base_env, env)

        # 8. Extract joint state (proprioception)
        result.joint_positions, result.joint_velocities = self._oracle._extract_joint_state(sim, base_env)

        self._last_perceive_time = result.timestamp
        return result

    def _render_images(self, sim) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Render RGB and depth images from camera.

        Returns:
            (rgb, depth) tuple, or (None, None) on failure
        """
        try:
            # Render RGB
            rgb = sim.render(
                width=self.image_size[0],
                height=self.image_size[1],
                camera_name=self.camera_name,
            )
            # Flip vertically (MuJoCo renders upside down)
            rgb = np.asarray(rgb)[::-1, :, :].copy()

            # Render depth - MuJoCo returns (rgb, depth) tuple when depth=True
            result = sim.render(
                width=self.image_size[0],
                height=self.image_size[1],
                camera_name=self.camera_name,
                depth=True,
            )

            # Handle different return formats
            if isinstance(result, tuple):
                # Returns (rgb, depth) tuple
                depth = np.asarray(result[1])
            else:
                # Returns just depth array
                depth = np.asarray(result)

            # Flip vertically
            depth = depth[::-1].copy()

            return rgb, depth

        except Exception as e:
            print(f"Warning: Failed to render: {e}")
            return None, None

    def _detections_to_3d(
        self,
        detections_2d: List[Detection],
        depth: np.ndarray,
    ) -> List[Detection]:
        """Convert 2D detections to 3D using depth image.

        For each detection:
        1. If mask available (gsam): use mask-based depth sampling for accuracy
        2. Otherwise: sample depth from bbox with multi-point strategy
        3. Back-project to 3D using camera intrinsics
        4. Transform to world frame
        5. Validate position is within workspace bounds

        Args:
            detections_2d: List of 2D detections (may include masks for gsam)
            depth: Depth image from MuJoCo

        Returns:
            List of Detection with 3D positions
        """
        detections_3d = []

        # LIBERO workspace bounds (approximate)
        # Different scenes have different coordinate systems:
        # - libero_spatial: table at ~0.9, objects at Z: 0.8-1.3
        # - libero_object: floor-level scene, objects at Z: -0.1 to 0.5
        # Use permissive bounds that cover all LIBERO scenes
        WORKSPACE_BOUNDS = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (-0.2, 1.5),  # Cover both floor-level and table-level scenes
        }

        for det in detections_2d:
            if det.bbox is None or len(det.bbox) != 4:
                continue

            # Check if detection has a mask (from gsam detector)
            has_mask = hasattr(det, 'mask') and det.mask is not None

            if has_mask:
                # Use mask-based 3D position estimation for precision
                point_world = self._mask_to_3d_position(det.mask, depth)
                if point_world is None:
                    # Fallback to bbox-based sampling if mask fails
                    depth_result = self._sample_depth_bbox(depth, det.bbox)
                    if depth_result is None:
                        continue
                    z, cx, cy = depth_result
                    if z <= 0 or z >= 0.995:
                        continue
                    point_cam = self._pixel_to_camera(cx, cy, z)
                    point_world = self._camera_to_world(point_cam)
            else:
                # Standard bbox-based depth sampling
                depth_result = self._sample_depth_bbox(depth, det.bbox)
                if depth_result is None:
                    continue
                z, cx, cy = depth_result
                if z <= 0 or z >= 0.995:
                    continue
                point_cam = self._pixel_to_camera(cx, cy, z)
                point_world = self._camera_to_world(point_cam)

            # Validate position is within workspace
            x, y, z_world = point_world
            if not (WORKSPACE_BOUNDS['x'][0] <= x <= WORKSPACE_BOUNDS['x'][1] and
                    WORKSPACE_BOUNDS['y'][0] <= y <= WORKSPACE_BOUNDS['y'][1] and
                    WORKSPACE_BOUNDS['z'][0] <= z_world <= WORKSPACE_BOUNDS['z'][1]):
                # Position outside workspace - skip this detection
                continue

            # Create 3D detection
            det_3d = Detection(
                class_name=det.class_name,
                position=point_world,
                confidence=det.confidence,
                bbox=det.bbox,
            )
            detections_3d.append(det_3d)

        return detections_3d

    def _mask_to_3d_position(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute 3D position from segmentation mask and depth.

        Uses the mask to sample depth only from object pixels, then computes
        the 3D centroid of the masked region. This is more accurate than
        bbox-based sampling for small/thin objects.

        Args:
            mask: Binary mask (H, W), True for object pixels
            depth: Normalized depth image (H, W) from MuJoCo

        Returns:
            3D position in world frame, or None if no valid depth
        """
        if mask is None or not mask.any():
            return None

        # Get all pixel coordinates where mask is True
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return None

        # Get depth values at mask pixels
        mask_depths = depth[ys, xs]

        # Filter valid depth values (not at far plane, not zero)
        valid_mask = (mask_depths > 0) & (mask_depths < 0.995) & np.isfinite(mask_depths)
        valid_depths = mask_depths[valid_mask]
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]

        if len(valid_depths) == 0:
            return None

        # Strategy: Use the front surface (minimum depth) for grasp target
        # This ensures we target the visible/graspable part of the object
        min_depth_idx = np.argmin(valid_depths)
        z_normalized = valid_depths[min_depth_idx]

        # Use centroid of pixels near the minimum depth (front surface)
        # This gives a more stable position than a single pixel
        depth_threshold = z_normalized + 0.02  # Within 2% of min depth
        front_mask = valid_depths <= depth_threshold
        front_xs = valid_xs[front_mask]
        front_ys = valid_ys[front_mask]

        # Compute centroid of front surface pixels
        cx = float(np.mean(front_xs))
        cy = float(np.mean(front_ys))
        z = float(np.mean(valid_depths[front_mask]))

        # Back-project to camera frame
        point_cam = self._pixel_to_camera(cx, cy, z)

        # Transform to world frame
        point_world = self._camera_to_world(point_cam)

        return point_world

    def _sample_depth_bbox(
        self,
        depth: np.ndarray,
        bbox: list,
    ) -> Optional[tuple]:
        """Sample depth within a bounding box with multi-point strategy.

        For small objects like bowls, sampling just the bbox center often
        picks up background. Instead, sample multiple points within
        the bbox and return the point with the closest (minimum) valid depth
        along with its pixel coordinates for accurate back-projection.

        Args:
            depth: Depth image (H, W)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Tuple of (depth, cx, cy) where cx,cy are pixel coords of the
            sampled point, or None if no valid depth found
        """
        h, w = depth.shape
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Multi-point sampling: sample a grid of points within the bbox
        # This gives us better coverage than just the center
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Sample points: center + 4 points at 25% and 75% of bbox dimensions
        # For a 3x3 grid, gives 9 sample points
        sample_fracs = [0.25, 0.5, 0.75]
        sample_points = []

        for fx in sample_fracs:
            for fy in sample_fracs:
                px = int(x1 + bbox_w * fx)
                py = int(y1 + bbox_h * fy)
                if 0 <= px < w and 0 <= py < h:
                    sample_points.append((px, py))

        # Find the point with minimum valid depth (closest to camera = on object)
        best_depth = None
        best_cx, best_cy = None, None

        for px, py in sample_points:
            # Sample a small 3x3 patch around each point for robustness
            patch_half = 1
            py_min = max(0, py - patch_half)
            py_max = min(h, py + patch_half + 1)
            px_min = max(0, px - patch_half)
            px_max = min(w, px + patch_half + 1)

            patch = depth[py_min:py_max, px_min:px_max]

            # Filter valid depth values (not at far plane)
            # MuJoCo depth: 0 = near plane, 1 = far plane (normalized)
            valid_mask = (patch > 0) & (patch < 0.995) & np.isfinite(patch)
            valid_depths = patch[valid_mask]

            if len(valid_depths) > 0:
                min_depth = float(np.min(valid_depths))
                if best_depth is None or min_depth < best_depth:
                    best_depth = min_depth
                    best_cx = float(px)
                    best_cy = float(py)

        if best_depth is None:
            return None

        return (best_depth, best_cx, best_cy)

    def _sample_depth(
        self,
        depth: np.ndarray,
        cx: float,
        cy: float,
    ) -> Optional[float]:
        """Sample depth at pixel location with robustness.

        Uses a patch around the center and takes median/mean
        to handle noise and edge cases.

        Args:
            depth: Depth image (H, W)
            cx, cy: Center pixel coordinates

        Returns:
            Depth value or None if invalid
        """
        h, w = depth.shape

        # Get patch bounds
        half = self.depth_patch_size // 2
        x_min = max(0, int(cx) - half)
        x_max = min(w, int(cx) + half + 1)
        y_min = max(0, int(cy) - half)
        y_max = min(h, int(cy) + half + 1)

        # Extract patch
        patch = depth[y_min:y_max, x_min:x_max]

        # Filter valid depth values
        # MuJoCo depth: 0 = near plane, 1 = far plane (normalized)
        # We need to convert and filter invalid values
        valid_mask = (patch > 0) & (patch < 1) & np.isfinite(patch)
        valid_depths = patch[valid_mask]

        if len(valid_depths) == 0:
            return None

        # Use median (robust) or mean
        if self.use_median_depth:
            return float(np.median(valid_depths))
        else:
            return float(np.mean(valid_depths))

    def _pixel_to_camera(
        self,
        u: float,
        v: float,
        z_normalized: float,
    ) -> np.ndarray:
        """Back-project pixel to camera frame.

        MuJoCo depth buffer is normalized [0, 1] but with non-linear mapping.
        Empirically calibrated formula: z = k / (1 - d) where k = extent * znear.

        Args:
            u, v: Pixel coordinates
            z_normalized: Normalized depth from MuJoCo

        Returns:
            3D point in camera frame [x, y, z]
        """
        # Empirically calibrated depth conversion for LIBERO
        # The depth buffer follows approximately: z = k / (1 - d)
        # where k = extent * znear (typically ~0.011 for LIBERO)
        # This gives ~2cm mean error for tabletop objects
        k = self._depth_scale
        if z_normalized >= 1.0:
            z_normalized = 0.9999  # Avoid division by zero
        z = k / (1.0 - z_normalized)

        # Back-project using camera intrinsics
        f = self.camera_params.focal_length
        cx = self.camera_params.cx
        cy = self.camera_params.cy

        # Camera coordinates (OpenGL convention: -Z forward)
        x = (u - cx) * z / f
        y = (v - cy) * z / f

        return np.array([x, y, -z])  # -z because MuJoCo uses -Z forward

    def _camera_to_world(self, point_cam: np.ndarray) -> np.ndarray:
        """Transform point from camera frame to world frame.

        Args:
            point_cam: 3D point in camera frame

        Returns:
            3D point in world frame
        """
        R = self.camera_params.rotation_matrix
        t = self.camera_params.position

        # World = R @ cam + t
        return R @ point_cam + t

    def _update_camera_params(self, env):
        """Update camera parameters from environment.

        Gets camera pose, FOV, and depth scale from MuJoCo sim.
        """
        sim = self._get_sim(env)
        if sim is None:
            return

        try:
            camera_id = sim.model.camera_name2id(self.camera_name)

            # Get FOV
            self.camera_params.fovy = sim.model.cam_fovy[camera_id]

            # Get camera pose
            self.camera_params.position = sim.data.cam_xpos[camera_id].copy()
            self.camera_params.rotation_matrix = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()

            # Calibrate depth scale from model parameters
            # Empirical formula: depth_scale ≈ extent * znear * 1.065
            # The 1.065 factor accounts for rendering pipeline quirks
            extent = sim.model.stat.extent
            znear = sim.model.vis.map.znear
            self._depth_scale = extent * znear * 1.065

        except Exception as e:
            print(f"Warning: Failed to update camera params: {e}")

    def _get_sim(self, env):
        """Get MuJoCo sim object from wrapped environment."""
        if hasattr(env, 'sim'):
            return env.sim
        if hasattr(env, '_env'):
            return self._get_sim(env._env)
        if hasattr(env, 'env'):
            return self._get_sim(env.env)
        return None

    def _get_base_env(self, env):
        """Get base robosuite environment."""
        if hasattr(env, 'robots'):
            return env
        if hasattr(env, '_env'):
            return self._get_base_env(env._env)
        if hasattr(env, 'env'):
            return self._get_base_env(env.env)
        return env

    def reset(self):
        """Reset perception state for new episode."""
        self.tracker.reset()
        self._bootstrapped = False

    def set_confidence_threshold(self, threshold: float):
        """Update detection confidence threshold."""
        self.confidence_threshold = threshold
        self.detector.set_confidence_threshold(threshold)
