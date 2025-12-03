"""Grasp point selection module.

Provides pluggable grasp selection strategies:
- HeuristicGraspSelector: Rule-based (rim grasps for hollow objects, center for solid)
- GIGAGraspSelector: 6-DoF grasp affordance model integration (TSDF input)
- ContactGraspNetSelector: 6-DoF Contact-GraspNet (point cloud input)

Usage:
    from brain_robot.skills.grasp_selection import get_grasp_selector

    # Use default (heuristic) selector
    selector = get_grasp_selector("heuristic")

    # Use GIGA selector (requires GIGA model)
    selector = get_grasp_selector("giga", model_path="path/to/giga.pth")

    # Use Contact-GraspNet selector (requires contact_graspnet_pytorch)
    selector = get_grasp_selector("contact_graspnet")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


@dataclass
class GraspPose:
    """6-DoF grasp pose with metadata."""

    # Position (3D)
    position: np.ndarray  # [x, y, z]

    # Orientation (quaternion [w, x, y, z] or rotation matrix)
    orientation: np.ndarray  # [qw, qx, qy, qz] or 3x3 matrix

    # Gripper width at grasp (meters, 0 = closed, ~0.08 = fully open)
    gripper_width: float = 0.04

    # Confidence score (0-1, higher is better)
    confidence: float = 1.0

    # Approach direction (unit vector pointing toward object)
    approach_direction: Optional[np.ndarray] = None

    # Strategy name for debugging
    strategy: str = "unknown"

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

    def to_7d(self) -> np.ndarray:
        """Convert to 7D pose [x, y, z, qw, qx, qy, qz]."""
        pose = np.zeros(7)
        pose[:3] = self.position

        if self.orientation.shape == (4,):
            # Already quaternion
            pose[3:7] = self.orientation
        elif self.orientation.shape == (3, 3):
            # Rotation matrix -> quaternion
            from scipy.spatial.transform import Rotation as R
            quat = R.from_matrix(self.orientation).as_quat()  # [x, y, z, w]
            pose[3:7] = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
        else:
            # Default orientation (gripper down)
            pose[3:7] = [-0.02, 0.707, 0.707, -0.02]

        return pose


class GraspSelector(ABC):
    """Abstract base class for grasp selection strategies."""

    @abstractmethod
    def select_grasp(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        obj_type: Optional[str] = None,
        gripper_pose: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        approach_strategy: str = "top_down",
        **kwargs,
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Select optimal grasp pose for an object.

        Args:
            obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
            obj_name: Object name (for heuristic lookup)
            obj_type: Object type (bowl, mug, cube, etc.)
            gripper_pose: Current gripper pose (for reachability)
            point_cloud: Object point cloud (N, 3) for learned methods
            rgb_image: RGB image for visual methods
            depth_image: Depth image for visual methods
            camera_intrinsics: Camera intrinsics (3, 3)
            approach_strategy: Approach direction hint
            **kwargs: Additional method-specific arguments

        Returns:
            Tuple of (GraspPose, info_dict)
        """
        pass

    @abstractmethod
    def select_multiple_grasps(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        n_grasps: int = 5,
        **kwargs,
    ) -> List[Tuple[GraspPose, Dict[str, Any]]]:
        """Select multiple candidate grasp poses, ranked by confidence.

        Args:
            obj_pose: Object pose
            obj_name: Object name
            n_grasps: Number of grasps to return
            **kwargs: Same as select_grasp

        Returns:
            List of (GraspPose, info_dict) tuples, sorted by confidence
        """
        pass


class HeuristicGraspSelector(GraspSelector):
    """Rule-based grasp selection for known object types.

    Implements the rim-grasp strategy for hollow objects (bowls, mugs)
    and center grasp for solid objects.
    """

    # Object types that are hollow and need rim grasping
    HOLLOW_OBJECTS = {'bowl', 'mug', 'cup', 'ramekin'}

    # Approximate radii for hollow objects (meters)
    OBJECT_RADII = {
        'bowl': 0.045,
        'mug': 0.035,
        'cup': 0.035,
        'ramekin': 0.030,
    }

    # Gripper finger half-width (meters)
    GRIPPER_FINGER_HALF_WIDTH = 0.005

    # Workspace limits
    MIN_WORKSPACE_Y = 0.12

    # Default gripper-down orientation [w, x, y, z]
    DEFAULT_ORIENTATION = np.array([-0.02, 0.707, 0.707, -0.02])

    def __init__(
        self,
        grasp_height_offset: float = 0.04,
        default_gripper_width: float = 0.04,
    ):
        """Initialize heuristic selector.

        Args:
            grasp_height_offset: Height above object center to grasp
            default_gripper_width: Default gripper width for grasps
        """
        self.grasp_height_offset = grasp_height_offset
        self.default_gripper_width = default_gripper_width

    def select_grasp(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        obj_type: Optional[str] = None,
        gripper_pose: Optional[np.ndarray] = None,
        approach_strategy: str = "top_down",
        **kwargs,
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Select grasp using heuristic rules."""
        info = {"grasp_strategy": "center", "rim_offset": 0.0}

        # Infer object type from name if not provided
        if obj_type is None:
            obj_name_lower = obj_name.lower()
            for hollow_type in self.HOLLOW_OBJECTS:
                if hollow_type in obj_name_lower:
                    obj_type = hollow_type
                    break

        # Check if hollow object needing rim grasp
        if obj_type and obj_type.lower() in self.HOLLOW_OBJECTS:
            grasp_pose, rim_info = self._compute_rim_grasp(
                obj_pose, obj_type, approach_strategy
            )
            info.update(rim_info)
            info["object_type"] = obj_type
        else:
            # Solid object: center grasp
            grasp_pose = self._compute_center_grasp(obj_pose)
            info["object_type"] = obj_type or "solid"

        return grasp_pose, info

    def select_multiple_grasps(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        n_grasps: int = 5,
        **kwargs,
    ) -> List[Tuple[GraspPose, Dict[str, Any]]]:
        """Return multiple grasp candidates.

        For heuristic selector, returns variations around the primary grasp.
        """
        primary, info = self.select_grasp(obj_pose, obj_name, **kwargs)
        results = [(primary, info)]

        # Add variations (different rim positions for hollow objects)
        obj_type = info.get("object_type", "solid")
        if obj_type in self.HOLLOW_OBJECTS:
            # Add grasps at different rim positions (0, 90, 180, 270 degrees)
            for angle in [np.pi/2, np.pi, -np.pi/2]:
                varied = self._vary_rim_angle(primary, angle)
                varied_info = info.copy()
                varied_info["rim_angle"] = float(angle)
                varied_info["confidence"] = 0.8  # Lower than primary
                results.append((varied, varied_info))
                if len(results) >= n_grasps:
                    break

        return results[:n_grasps]

    def _compute_rim_grasp(
        self,
        obj_pose: np.ndarray,
        obj_type: str,
        approach_strategy: str,
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Compute rim grasp for hollow object."""
        info = {"grasp_strategy": "rim"}

        radius = self.OBJECT_RADII.get(obj_type.lower(), 0.05)
        rim_offset = radius - self.GRIPPER_FINGER_HALF_WIDTH
        info["rim_offset"] = rim_offset

        # Determine offset direction based on approach
        if approach_strategy in ('front_angled', 'front_angled_steep'):
            direction = np.array([0.0, 1.0])  # Toward front (away from robot)
        else:
            direction = np.array([0.0, -1.0])  # Toward robot base

        info["offset_direction"] = direction.tolist()

        # Compute grasp XY
        grasp_xy = obj_pose[:2] + direction * rim_offset

        # Clamp Y for top-down approaches
        if approach_strategy not in ('front_angled', 'front_angled_steep'):
            if grasp_xy[1] < self.MIN_WORKSPACE_Y:
                grasp_xy[1] = self.MIN_WORKSPACE_Y
                info["y_clamped"] = True

        # Compute grasp Z based on approach
        if approach_strategy in ('front_angled', 'front_angled_steep', 'front_horizontal'):
            grasp_z = obj_pose[2] + 0.02  # Just above bowl center
        else:
            grasp_z = obj_pose[2] + self.grasp_height_offset

        grasp_pose = GraspPose(
            position=np.array([grasp_xy[0], grasp_xy[1], grasp_z]),
            orientation=self.DEFAULT_ORIENTATION.copy(),
            gripper_width=self.default_gripper_width,
            confidence=0.9,
            strategy="rim",
            metadata=info,
        )

        return grasp_pose, info

    def _compute_center_grasp(self, obj_pose: np.ndarray) -> GraspPose:
        """Compute center grasp for solid object."""
        return GraspPose(
            position=np.array([
                obj_pose[0],
                obj_pose[1],
                obj_pose[2] + self.grasp_height_offset,
            ]),
            orientation=self.DEFAULT_ORIENTATION.copy(),
            gripper_width=self.default_gripper_width,
            confidence=0.9,
            strategy="center",
        )

    def _vary_rim_angle(self, grasp: GraspPose, angle: float) -> GraspPose:
        """Create a grasp variation at different rim angle."""
        # Rotate the XY offset around object center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Get offset from original (approximate from metadata)
        if grasp.metadata and "rim_offset" in grasp.metadata:
            offset = grasp.metadata["rim_offset"]
            direction = np.array(grasp.metadata.get("offset_direction", [0, -1]))
            new_direction = rotation @ direction

            # Compute new position (need original obj center)
            # This is approximate since we don't have obj_pose here
            original_offset = direction * offset
            new_offset = new_direction * offset
            delta = new_offset - original_offset
            new_pos = grasp.position.copy()
            new_pos[:2] += delta
        else:
            new_pos = grasp.position.copy()

        return GraspPose(
            position=new_pos,
            orientation=grasp.orientation.copy(),
            gripper_width=grasp.gripper_width,
            confidence=grasp.confidence * 0.9,
            strategy=grasp.strategy,
            metadata=grasp.metadata,
        )


class GIGAGraspSelector(GraspSelector):
    """6-DoF grasp affordance model (GIGA) integration.

    GIGA predicts dense grasp affordances from point clouds or depth images.
    This selector loads a pre-trained GIGA model and queries it for grasp poses.

    Paper: https://arxiv.org/abs/2104.01542

    Note: This is a stub implementation. Full integration requires:
    1. GIGA model weights (giga.pth)
    2. Point cloud extraction from LIBERO observations
    3. Coordinate frame transformations (camera -> world)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        voxel_size: float = 0.005,
        score_threshold: float = 0.5,
        grasp_height_offset: float = 0.04,  # Ignored by GIGA, but accepted for compatibility
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        """Initialize GIGA selector.

        Args:
            model_path: Path to GIGA model weights
            device: Device for inference ("cuda" or "cpu")
            voxel_size: Voxel size for point cloud processing
            score_threshold: Minimum grasp quality score
        """
        self.model_path = model_path
        self.device = device
        self.voxel_size = voxel_size
        self.score_threshold = score_threshold

        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load GIGA model from checkpoint.

        Note: GIGA model loading is not yet implemented.
        This selector will fall back to heuristic grasp selection.
        """
        # GIGA model loading would go here when implemented:
        # self.model = GIGANet.load(model_path)
        # self.model.to(self.device)
        # self.model.eval()
        print(f"[GIGAGraspSelector] Model loading not implemented: {model_path}")

    def select_grasp(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        obj_type: Optional[str] = None,
        gripper_pose: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        approach_strategy: str = "top_down",
        **kwargs,
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Select grasp using GIGA model.

        Requires point_cloud or (depth_image + camera_intrinsics).
        Falls back to heuristic if model not loaded or inputs missing.
        """
        info = {"method": "giga", "fallback": False}

        # Check if we can run GIGA
        if self.model is None:
            info["fallback"] = True
            info["fallback_reason"] = "model_not_loaded"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        if point_cloud is None:
            if depth_image is not None and camera_intrinsics is not None:
                point_cloud = self._depth_to_pointcloud(depth_image, camera_intrinsics)
            else:
                info["fallback"] = True
                info["fallback_reason"] = "no_point_cloud"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        # Run GIGA inference
        try:
            grasps = self._run_giga_inference(point_cloud)
            if len(grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = "no_grasps_found"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Return best grasp
            best_grasp = grasps[0]
            info["n_candidates"] = len(grasps)
            info["best_score"] = best_grasp.confidence
            return best_grasp, info

        except Exception as e:
            info["fallback"] = True
            info["fallback_reason"] = f"inference_error: {e}"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

    def select_multiple_grasps(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        n_grasps: int = 5,
        **kwargs,
    ) -> List[Tuple[GraspPose, Dict[str, Any]]]:
        """Select multiple grasp candidates from GIGA."""
        # For now, just return variations from single grasp
        primary, info = self.select_grasp(obj_pose, obj_name, **kwargs)

        if info.get("fallback"):
            # Use heuristic fallback for multiple grasps
            heuristic = HeuristicGraspSelector()
            return heuristic.select_multiple_grasps(obj_pose, obj_name, n_grasps, **kwargs)

        # TODO: Implement full GIGA multi-grasp selection
        return [(primary, info)]

    def _run_giga_inference(self, point_cloud: np.ndarray) -> List[GraspPose]:
        """Run GIGA inference on point cloud.

        This is a stub - actual implementation would:
        1. Voxelize point cloud
        2. Run through GIGA network
        3. Extract top-k grasp poses
        4. Transform to world frame
        """
        # Placeholder - return empty list to trigger fallback
        return []

    def _depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Convert depth image to point cloud."""
        # Standard depth -> pointcloud conversion
        h, w = depth_image.shape
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = z.flatten() > 0
        return points[valid]

    def _fallback_heuristic(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        approach_strategy: str,
        info: Dict[str, Any],
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Fall back to heuristic grasp selection."""
        heuristic = HeuristicGraspSelector()
        grasp, heuristic_info = heuristic.select_grasp(
            obj_pose, obj_name, approach_strategy=approach_strategy
        )
        info.update(heuristic_info)
        return grasp, info


# Global cache for Contact-GraspNet model to avoid reloading every episode
_CGN_MODEL_CACHE = {
    "grasp_estimator": None,
    "loaded": False,
    "error": None,
}


def _get_cached_cgn_model(checkpoint_dir: Optional[str] = None):
    """Get or load the Contact-GraspNet model from cache.

    This avoids reloading the ~200MB model every episode, saving ~2-3 seconds per grasp.
    """
    global _CGN_MODEL_CACHE

    if _CGN_MODEL_CACHE["loaded"]:
        return _CGN_MODEL_CACHE["grasp_estimator"], None

    if _CGN_MODEL_CACHE["error"] is not None:
        return None, _CGN_MODEL_CACHE["error"]

    try:
        import sys
        cgn_path = "/tmp/contact_graspnet_pytorch"
        if cgn_path not in sys.path:
            sys.path.insert(0, cgn_path)

        from contact_graspnet_pytorch import config_utils
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

        if checkpoint_dir is None:
            checkpoint_dir = "/tmp/contact_graspnet_pytorch/checkpoints/contact_graspnet"

        config = config_utils.load_config(checkpoint_dir, batch_size=1)
        grasp_estimator = GraspEstimator(config)

        _CGN_MODEL_CACHE["grasp_estimator"] = grasp_estimator
        _CGN_MODEL_CACHE["loaded"] = True
        print(f"[ContactGraspNetSelector] Model loaded successfully on {grasp_estimator.device}")

        return grasp_estimator, None

    except ImportError as e:
        error = f"contact_graspnet_pytorch not installed: {e}"
        _CGN_MODEL_CACHE["error"] = error
        print(f"[ContactGraspNetSelector] {error}")
        print("  Install from: https://github.com/elchun/contact_graspnet_pytorch")
        return None, error

    except Exception as e:
        error = f"Failed to load model: {e}"
        _CGN_MODEL_CACHE["error"] = error
        print(f"[ContactGraspNetSelector] {error}")
        return None, error


class ContactGraspNetSelector(GraspSelector):
    """Contact-GraspNet 6-DoF grasp selector.

    Uses the Contact-GraspNet model for dense 6-DoF grasp prediction from
    point clouds. Falls back to heuristic grasp selection if the model
    is not available.

    Paper: "Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes"
    https://github.com/NVlabs/contact_graspnet

    Input: Point cloud (N, 3) in world frame
    Output: 6-DoF grasp poses (position + rotation) with confidence scores

    Note: Requires contact_graspnet_pytorch package to be installed.
    Install from: https://github.com/elchun/contact_graspnet_pytorch

    The model is cached globally to avoid reloading every episode (~2-3s savings).
    """

    # LIBERO table height is approximately 0.8m
    # Objects typically sit 0.82-0.95m above ground
    DEFAULT_Z_RANGE = (0.75, 1.2)  # Covers table surface to raised objects

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        forward_passes: int = 1,
        z_range: Optional[Tuple[float, float]] = None,
        score_threshold: float = 0.5,
        filter_grasps: bool = True,
        grasp_height_offset: float = 0.04,
        **kwargs,
    ):
        """Initialize Contact-GraspNet selector.

        Args:
            model_path: Path to model checkpoint (if None, uses default)
            checkpoint_dir: Directory containing checkpoints
            device: Device for inference ("cuda" or "cpu")
            forward_passes: Number of forward passes for sampling
            z_range: Z range for filtering point cloud (min, max).
                     Default: (0.75, 1.2) tuned for LIBERO table (~0.8m).
            score_threshold: Minimum grasp confidence score
            filter_grasps: Whether to filter grasps to object surface
            grasp_height_offset: Ignored (for API compatibility)
            **kwargs: Additional arguments (ignored)
        """
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.forward_passes = forward_passes
        self.z_range = z_range if z_range is not None else self.DEFAULT_Z_RANGE
        self.score_threshold = score_threshold
        self.filter_grasps = filter_grasps

        # Use global cache instead of loading fresh each time
        self.grasp_estimator, self._load_error = _get_cached_cgn_model(checkpoint_dir)
        self._model_loaded = self.grasp_estimator is not None

    def select_grasp(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        obj_type: Optional[str] = None,
        gripper_pose: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        approach_strategy: str = "top_down",
        **kwargs,
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Select grasp using Contact-GraspNet.

        Args:
            obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
            obj_name: Object name
            obj_type: Object type (bowl, mug, etc.)
            gripper_pose: Current gripper pose
            point_cloud: Object point cloud (N, 3) in world frame
            rgb_image: RGB image (optional, for visualization)
            depth_image: Depth image in meters (used if point_cloud not provided)
            camera_intrinsics: Camera K matrix (3x3)
            camera_extrinsics: Camera pose in world frame (4x4)
            segmentation_mask: Object segmentation mask (optional)
            approach_strategy: Approach direction hint
            **kwargs: Additional arguments

        Returns:
            Tuple of (GraspPose, info_dict)
        """
        info = {
            "method": "contact_graspnet",
            "fallback": False,
            "model_loaded": self._model_loaded,
        }

        # Check if model is loaded
        if not self._model_loaded:
            info["fallback"] = True
            info["fallback_reason"] = self._load_error or "model_not_loaded"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        # Early check: hollow objects are better handled by heuristic rim-grasp
        # Do this BEFORE expensive point cloud generation and inference
        obj_name_lower = obj_name.lower()
        is_hollow = any(hollow in obj_name_lower for hollow in ['bowl', 'mug', 'cup', 'ramekin'])

        if is_hollow:
            info["fallback"] = True
            info["fallback_reason"] = f"hollow_object_fallback ({obj_name})"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        # Get or create point cloud
        if point_cloud is None:
            if depth_image is not None and camera_intrinsics is not None:
                point_cloud = self._depth_to_pointcloud(
                    depth_image, camera_intrinsics, camera_extrinsics, segmentation_mask
                )
                info["point_cloud_source"] = "depth_image"
                info["point_cloud_size"] = point_cloud.shape[0] if point_cloud is not None else 0
            else:
                info["fallback"] = True
                info["fallback_reason"] = "no_point_cloud_or_depth"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        if point_cloud.shape[0] < 10:
            info["fallback"] = True
            info["fallback_reason"] = f"insufficient_points: {point_cloud.shape[0]}"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        # Run Contact-GraspNet inference
        try:
            grasps = self._run_inference(point_cloud, segmentation_mask)

            if len(grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = "no_grasps_found"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Filter grasps by proximity to target object
            # CGN returns grasps for the entire scene, so we filter to target
            # Using tight 8cm threshold to ensure grasps are on the object itself
            obj_position = obj_pose[:3]
            max_grasp_distance = 0.08  # 8cm max from object center (tightened from 15cm)

            filtered_grasps = []
            for g in grasps:
                dist = np.linalg.norm(g["position"] - obj_position)
                if dist <= max_grasp_distance:
                    g["distance_to_object"] = dist
                    filtered_grasps.append(g)

            info["n_candidates_raw"] = len(grasps)
            info["n_candidates_filtered"] = len(filtered_grasps)

            if len(filtered_grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = f"no_grasps_near_object (closest: {min(np.linalg.norm(g['position'] - obj_position) for g in grasps):.3f}m)"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Re-sort by score (already sorted, but just in case)
            filtered_grasps.sort(key=lambda g: g["score"], reverse=True)
            grasps = filtered_grasps

            # Convert best grasp to GraspPose
            best_grasp = grasps[0]
            info["n_candidates"] = len(grasps)
            info["best_score"] = float(best_grasp["score"])
            info["best_grasp_distance"] = float(best_grasp["distance_to_object"])

            grasp_pose = GraspPose(
                position=best_grasp["position"],
                orientation=best_grasp["orientation"],
                gripper_width=best_grasp.get("width", 0.04),
                confidence=best_grasp["score"],
                approach_direction=best_grasp.get("approach", None),
                strategy="contact_graspnet",
                metadata={"grasp_index": 0},
            )

            return grasp_pose, info

        except Exception as e:
            info["fallback"] = True
            info["fallback_reason"] = f"inference_error: {e}"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

    def select_multiple_grasps(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        n_grasps: int = 5,
        **kwargs,
    ) -> List[Tuple[GraspPose, Dict[str, Any]]]:
        """Select multiple grasp candidates from Contact-GraspNet."""
        if not self._model_loaded:
            # Use heuristic fallback
            heuristic = HeuristicGraspSelector()
            return heuristic.select_multiple_grasps(obj_pose, obj_name, n_grasps, **kwargs)

        # Get point cloud from kwargs
        point_cloud = kwargs.get("point_cloud")
        if point_cloud is None:
            depth_image = kwargs.get("depth_image")
            camera_intrinsics = kwargs.get("camera_intrinsics")
            if depth_image is not None and camera_intrinsics is not None:
                point_cloud = self._depth_to_pointcloud(depth_image, camera_intrinsics)

        if point_cloud is None or point_cloud.shape[0] < 10:
            heuristic = HeuristicGraspSelector()
            return heuristic.select_multiple_grasps(obj_pose, obj_name, n_grasps, **kwargs)

        try:
            grasps = self._run_inference(point_cloud, kwargs.get("segmentation_mask"))
            results = []

            for i, g in enumerate(grasps[:n_grasps]):
                grasp_pose = GraspPose(
                    position=g["position"],
                    orientation=g["orientation"],
                    gripper_width=g.get("width", 0.04),
                    confidence=g["score"],
                    approach_direction=g.get("approach", None),
                    strategy="contact_graspnet",
                    metadata={"grasp_index": i},
                )
                info = {
                    "method": "contact_graspnet",
                    "fallback": False,
                    "grasp_index": i,
                    "score": float(g["score"]),
                }
                results.append((grasp_pose, info))

            return results if results else [(self._fallback_heuristic(
                obj_pose, obj_name, "top_down", {}
            ))]

        except Exception:
            heuristic = HeuristicGraspSelector()
            return heuristic.select_multiple_grasps(obj_pose, obj_name, n_grasps, **kwargs)

    def _run_inference(
        self,
        point_cloud: np.ndarray,
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Run Contact-GraspNet inference.

        Args:
            point_cloud: (N, 3) point cloud in world frame
            segmentation_mask: Optional (H, W) segmentation mask

        Returns:
            List of grasp dictionaries with keys:
            - position: (3,) grasp position
            - orientation: (3, 3) rotation matrix or (4,) quaternion
            - score: confidence score
            - width: gripper width
            - approach: approach direction
        """
        if self.grasp_estimator is None:
            return []

        # Filter point cloud by z range
        z_mask = (point_cloud[:, 2] >= self.z_range[0]) & (point_cloud[:, 2] <= self.z_range[1])
        filtered_pc = point_cloud[z_mask].astype(np.float32)

        if filtered_pc.shape[0] < 10:
            return []

        # Run inference using predict_scene_grasps
        # Returns: (pred_grasps_dict, scores_dict, contact_pts_dict, gripper_openings_dict)
        # Where dicts are keyed by segment id (-1 = all grasps, no segmentation)
        pred_grasps_dict, pred_scores_dict, contact_pts_dict, gripper_openings_dict = self.grasp_estimator.predict_scene_grasps(
            filtered_pc,
            forward_passes=self.forward_passes,
        )

        # Extract grasps from dict (key -1 = unsegmented/all grasps)
        if -1 not in pred_grasps_dict:
            return []

        pred_grasps = pred_grasps_dict[-1]  # (N, 4, 4) transform matrices
        pred_scores = pred_scores_dict[-1]  # (N,) scores
        gripper_openings = gripper_openings_dict[-1] if -1 in gripper_openings_dict else None  # (N,) widths

        # Convert to list of grasp dictionaries
        grasps = []
        for i in range(len(pred_scores)):
            if pred_scores[i] >= self.score_threshold:
                # pred_grasps[i] is a 4x4 transform matrix
                grasp_transform = pred_grasps[i]
                position = grasp_transform[:3, 3]
                rotation = grasp_transform[:3, :3]

                # Convert rotation matrix to quaternion [w, x, y, z]
                from scipy.spatial.transform import Rotation as R
                quat = R.from_matrix(rotation).as_quat()  # [x, y, z, w]
                orientation = np.array([quat[3], quat[0], quat[1], quat[2]])

                width = float(gripper_openings[i]) if gripper_openings is not None else 0.04
                grasps.append({
                    "position": position,
                    "orientation": orientation,
                    "score": float(pred_scores[i]),
                    "width": width,
                    "approach": -rotation[:, 2],  # Z-axis of grasp frame
                })

        # Sort by score descending
        grasps.sort(key=lambda g: g["score"], reverse=True)
        return grasps

    def _depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert depth image to point cloud in world frame.

        Args:
            depth_image: (H, W) depth image in meters
            camera_intrinsics: (3, 3) camera K matrix
            camera_extrinsics: (4, 4) camera pose in world frame (T_world_camera)
            segmentation_mask: Optional (H, W) mask for filtering

        Returns:
            (N, 3) point cloud in world frame
        """
        h, w = depth_image.shape[:2]
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        if depth_image.ndim == 3:
            z = depth_image[:, :, 0]
        else:
            z = depth_image

        # Unproject to camera frame
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = z.flatten() > 0.1

        # Apply segmentation mask if provided
        if segmentation_mask is not None:
            seg_mask = segmentation_mask.flatten() > 0
            valid = valid & seg_mask

        points_cam = points_cam[valid]

        # Transform to world frame if extrinsics provided
        if camera_extrinsics is not None and points_cam.shape[0] > 0:
            # Homogeneous coordinates
            points_homo = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
            # Transform: p_world = T_world_camera @ p_camera
            points_world = (camera_extrinsics @ points_homo.T).T[:, :3]
            return points_world

        return points_cam

    def _fallback_heuristic(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        approach_strategy: str,
        info: Dict[str, Any],
    ) -> Tuple[GraspPose, Dict[str, Any]]:
        """Fall back to heuristic grasp selection."""
        heuristic = HeuristicGraspSelector()
        grasp, heuristic_info = heuristic.select_grasp(
            obj_pose, obj_name, approach_strategy=approach_strategy
        )
        info.update(heuristic_info)
        return grasp, info


# Factory function for creating selectors
_SELECTOR_REGISTRY = {
    "heuristic": HeuristicGraspSelector,
    "giga": GIGAGraspSelector,
    "contact_graspnet": ContactGraspNetSelector,
}


def get_grasp_selector(
    selector_type: str = "heuristic",
    **kwargs,
) -> GraspSelector:
    """Factory function to create grasp selectors.

    Args:
        selector_type: Type of selector ("heuristic" or "giga")
        **kwargs: Arguments passed to selector constructor

    Returns:
        GraspSelector instance

    Raises:
        ValueError: If selector_type is unknown
    """
    if selector_type not in _SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector type: {selector_type}. "
            f"Available: {list(_SELECTOR_REGISTRY.keys())}"
        )

    return _SELECTOR_REGISTRY[selector_type](**kwargs)


def register_grasp_selector(name: str, selector_class: type):
    """Register a custom grasp selector.

    Args:
        name: Name for the selector
        selector_class: Class implementing GraspSelector interface
    """
    _SELECTOR_REGISTRY[name] = selector_class
