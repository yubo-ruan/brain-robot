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
    """6-DoF grasp affordance model (GIGA/VGN) integration.

    GIGA/VGN predicts dense grasp affordances from TSDF voxel grids.
    Takes depth image, creates TSDF, runs through 3D CNN to predict:
    - Grasp quality (per-voxel confidence)
    - Grasp orientation (quaternion)
    - Gripper width

    Paper: https://arxiv.org/abs/2104.01542
    GitHub: https://github.com/UT-Austin-RPL/GIGA

    Input: Depth image + camera intrinsics/extrinsics
    Output: 6-DoF grasp poses in world frame

    Note: Requires open3d for TSDF fusion. Model weights can be VGN or GIGA.
    """

    # TSDF volume parameters (tuned for LIBERO tabletop)
    TSDF_SIZE = 0.3  # 30cm cube
    TSDF_RESOLUTION = 40  # 40x40x40 voxels

    # Workspace bounds in world frame (LIBERO table)
    # Table is at ~0.8m height, objects are 0.8-1.2m
    WORKSPACE_CENTER = np.array([0.0, 0.15, 0.95])  # Center of grasp volume

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        tsdf_size: float = 0.3,
        resolution: int = 40,
        score_threshold: float = 0.5,
        grasp_height_offset: float = 0.04,
        **kwargs,
    ):
        """Initialize GIGA/VGN selector.

        Args:
            model_path: Path to VGN/GIGA model weights (.pt file)
            device: Device for inference ("cuda" or "cpu")
            tsdf_size: Size of TSDF volume in meters
            resolution: TSDF voxel resolution (40 = 40x40x40)
            score_threshold: Minimum grasp quality score
            grasp_height_offset: Ignored (for API compatibility)
        """
        import torch
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tsdf_size = tsdf_size
        self.resolution = resolution
        self.score_threshold = score_threshold

        self.model = None
        self._model_loaded = False

        # Precompute query positions for implicit function
        self._init_query_positions()

        if model_path:
            self._load_model(model_path)

    def _init_query_positions(self):
        """Initialize query positions for grasp prediction."""
        import torch
        x, y, z = torch.meshgrid(
            torch.linspace(-0.5, 0.5 - 1.0/self.resolution, self.resolution),
            torch.linspace(-0.5, 0.5 - 1.0/self.resolution, self.resolution),
            torch.linspace(-0.5, 0.5 - 1.0/self.resolution, self.resolution),
            indexing='ij'
        )
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0)
        self.query_positions = pos.view(1, self.resolution**3, 3).to(self.device)

    def _load_model(self, model_path: str):
        """Load VGN/GIGA model from checkpoint."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from pathlib import Path

        path = Path(model_path)
        if not path.exists():
            print(f"[GIGAGraspSelector] Model not found: {model_path}")
            return

        try:
            # Define VGN network architecture inline (avoids torch_scatter dependency)
            def conv(in_ch, out_ch, k):
                return nn.Conv3d(in_ch, out_ch, k, padding=k//2)

            def conv_stride(in_ch, out_ch, k):
                return nn.Conv3d(in_ch, out_ch, k, stride=2, padding=k//2)

            class Encoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = conv_stride(1, 16, 5)
                    self.conv2 = conv_stride(16, 32, 3)
                    self.conv3 = conv_stride(32, 64, 3)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = F.relu(self.conv3(x))
                    return x

            class Decoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = conv(64, 64, 3)
                    self.conv2 = conv(64, 32, 3)
                    self.conv3 = conv(32, 16, 5)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.interpolate(x, 10)
                    x = F.relu(self.conv2(x))
                    x = F.interpolate(x, 20)
                    x = F.relu(self.conv3(x))
                    x = F.interpolate(x, 40)
                    return x

            class VGNNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = Encoder()
                    self.decoder = Decoder()
                    self.conv_qual = conv(16, 1, 5)
                    self.conv_rot = conv(16, 4, 5)
                    self.conv_width = conv(16, 1, 5)

                def forward(self, x, pos=None):
                    x = self.encoder(x)
                    x = self.decoder(x)
                    qual = torch.sigmoid(self.conv_qual(x))
                    rot = F.normalize(self.conv_rot(x), dim=1)
                    width = self.conv_width(x)
                    return qual, rot, width

            self.model = VGNNet().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._model_loaded = True
            print(f"[GIGAGraspSelector] VGN model loaded from {model_path}")

        except Exception as e:
            print(f"[GIGAGraspSelector] Failed to load model: {e}")
            self._model_loaded = False

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
        """Select grasp using GIGA/VGN model.

        Args:
            obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
            obj_name: Object name
            depth_image: Depth image in meters
            camera_intrinsics: Camera K matrix (3x3)
            camera_extrinsics: Camera pose in world frame (4x4)
            segmentation_mask: Optional mask to focus on target object

        Returns:
            Tuple of (GraspPose, info_dict)
        """
        info = {"method": "giga", "fallback": False, "model_loaded": self._model_loaded}

        # Check if model is loaded
        if not self._model_loaded:
            info["fallback"] = True
            info["fallback_reason"] = "model_not_loaded"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        # Need depth and camera params for TSDF
        if depth_image is None or camera_intrinsics is None:
            info["fallback"] = True
            info["fallback_reason"] = "no_depth_or_intrinsics"
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

        try:
            # Create TSDF centered on object
            tsdf_vol = self._create_tsdf(
                depth_image, camera_intrinsics, camera_extrinsics,
                center=obj_pose[:3], mask=segmentation_mask
            )
            info["tsdf_created"] = True

            # Run VGN inference
            grasps, scores = self._run_inference(tsdf_vol, obj_pose[:3])
            info["n_candidates_raw"] = len(grasps)

            if len(grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = "no_grasps_found"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Filter by score
            valid_grasps = [(g, s) for g, s in zip(grasps, scores) if s >= self.score_threshold]
            info["n_candidates_filtered"] = len(valid_grasps)

            if len(valid_grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = f"no_grasps_above_threshold ({self.score_threshold})"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Sort by score and take best
            valid_grasps.sort(key=lambda x: x[1], reverse=True)
            best_grasp, best_score = valid_grasps[0]

            info["best_score"] = float(best_score)
            info["n_candidates"] = len(valid_grasps)

            # Default orientation (gripper down) for robosuite compatibility
            default_orientation = np.array([-0.02, 0.707, 0.707, -0.02])

            grasp_pose = GraspPose(
                position=best_grasp["position"],
                orientation=default_orientation,  # Use safe orientation
                gripper_width=best_grasp["width"],
                confidence=best_score,
                strategy="giga",
                metadata={"original_orientation": best_grasp["orientation"].tolist()},
            )

            return grasp_pose, info

        except Exception as e:
            info["fallback"] = True
            info["fallback_reason"] = f"inference_error: {e}"
            import traceback
            traceback.print_exc()
            return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

    def select_multiple_grasps(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        n_grasps: int = 5,
        **kwargs,
    ) -> List[Tuple[GraspPose, Dict[str, Any]]]:
        """Select multiple grasp candidates from GIGA."""
        primary, info = self.select_grasp(obj_pose, obj_name, **kwargs)

        if info.get("fallback"):
            heuristic = HeuristicGraspSelector()
            return heuristic.select_multiple_grasps(obj_pose, obj_name, n_grasps, **kwargs)

        return [(primary, info)]

    def _create_tsdf(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: Optional[np.ndarray],
        center: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create TSDF volume from depth image using point cloud projection.

        Uses a simplified TSDF construction that projects depth points into
        a voxel grid and computes signed distances. More stable than open3d's
        TSDF integration which can segfault.

        Args:
            depth_image: Depth in meters (H, W)
            camera_intrinsics: K matrix (3, 3)
            camera_extrinsics: T_world_camera (4, 4)
            center: World position to center TSDF on
            mask: Optional segmentation mask

        Returns:
            TSDF grid (1, resolution, resolution, resolution)
        """
        from scipy import ndimage

        # Handle depth shape
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            depth_masked = depth_image.copy()
            depth_masked[mask == 0] = 0
        else:
            depth_masked = depth_image

        h, w = depth_masked.shape
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        # Deproject depth to 3D points in camera frame
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_masked
        valid = z > 0.1  # Minimum valid depth

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack to get camera-frame points
        points_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)

        # Transform to world frame
        if camera_extrinsics is not None:
            T = camera_extrinsics
            R = T[:3, :3]
            t = T[:3, 3]
            points_world = points_cam @ R.T + t
        else:
            points_world = points_cam

        # Get valid points
        valid_points = points_world[valid]  # (N, 3)

        if len(valid_points) < 10:
            # Not enough points, return empty TSDF
            return np.ones((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)

        # Create voxel grid centered on object
        voxel_size = self.tsdf_size / self.resolution
        half_size = self.tsdf_size / 2

        # TSDF grid origin (corner of volume in world frame)
        origin = center - np.array([half_size, half_size, half_size])

        # Initialize TSDF with truncation distance
        trunc = 4 * voxel_size
        tsdf_grid = np.ones((self.resolution, self.resolution, self.resolution), dtype=np.float32) * trunc

        # Convert valid points to voxel coordinates
        voxel_coords = ((valid_points - origin) / voxel_size).astype(np.int32)

        # Clip to valid range
        valid_voxel = np.all(
            (voxel_coords >= 0) & (voxel_coords < self.resolution),
            axis=1
        )
        voxel_coords = voxel_coords[valid_voxel]
        valid_points = valid_points[valid_voxel]

        # Mark occupied voxels (surface at 0, inside negative, outside positive)
        # For each point, set the voxel it's in to 0 (on surface)
        for idx in range(len(voxel_coords)):
            i, j, k = voxel_coords[idx]
            tsdf_grid[i, j, k] = 0.0

        # Propagate signed distance using distance transform
        # Inside (negative): use distance from surface with negative sign
        # Outside (positive): use distance from surface with positive sign
        occupied = tsdf_grid == 0.0
        if occupied.any():
            # Distance transform gives distance to nearest occupied voxel
            dist_to_surface = ndimage.distance_transform_edt(~occupied) * voxel_size

            # Clip to truncation distance and normalize
            tsdf_grid = np.minimum(dist_to_surface, trunc) / trunc

            # Mark interior (heuristic: voxels "below" the surface in z)
            # This is approximate but works for tabletop scenes
            for idx in range(len(voxel_coords)):
                i, j, k = voxel_coords[idx]
                # Mark voxels below the surface point as inside (negative)
                for kk in range(k + 1, self.resolution):
                    if tsdf_grid[i, j, kk] > 0:
                        tsdf_grid[i, j, kk] = -tsdf_grid[i, j, kk]

        return tsdf_grid.reshape(1, self.resolution, self.resolution, self.resolution)

    def _run_inference(
        self,
        tsdf_vol: np.ndarray,
        center: np.ndarray,
    ) -> Tuple[List[Dict], List[float]]:
        """Run VGN inference on TSDF volume.

        Args:
            tsdf_vol: TSDF grid (1, 40, 40, 40)
            center: World position of TSDF center

        Returns:
            List of grasp dicts and scores
        """
        import torch
        from scipy import ndimage
        from scipy.spatial.transform import Rotation as R

        # Convert to tensor (ensure float32)
        tsdf_tensor = torch.from_numpy(tsdf_vol.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, 1, 40, 40, 40)

        # Forward pass
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = self.model(tsdf_tensor)

        # Move to CPU
        qual_vol = qual_vol.cpu().squeeze().numpy()  # (40, 40, 40)
        rot_vol = rot_vol.cpu().squeeze().numpy()  # (4, 40, 40, 40) -> need (40, 40, 40, 4)
        rot_vol = np.transpose(rot_vol, (1, 2, 3, 0))
        width_vol = width_vol.cpu().squeeze().numpy()  # (40, 40, 40)

        # Post-process quality volume
        # Smooth with Gaussian
        qual_vol = ndimage.gaussian_filter(qual_vol, sigma=1.0, mode="nearest")

        # Mask out invalid regions (far from surface)
        tsdf_np = tsdf_vol.squeeze()
        outside = tsdf_np > 0.5
        inside = np.logical_and(1e-3 < tsdf_np, tsdf_np < 0.5)
        valid = ndimage.binary_dilation(outside, iterations=2, mask=~inside)
        qual_vol[~valid] = 0.0

        # Apply score threshold
        qual_vol[qual_vol < 0.5] = 0.0

        # Non-maximum suppression
        max_vol = ndimage.maximum_filter(qual_vol, size=4)
        qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)

        # Extract grasps
        grasps = []
        scores = []
        voxel_size = self.tsdf_size / self.resolution

        for idx in np.argwhere(qual_vol > 0):
            i, j, k = idx
            score = qual_vol[i, j, k]

            # Voxel position in TSDF frame (centered at origin)
            pos_tsdf = (np.array([i, j, k]) + 0.5) * voxel_size - self.tsdf_size / 2

            # Transform to world frame
            pos_world = pos_tsdf + center

            # Orientation (quaternion)
            quat = rot_vol[i, j, k]  # [x, y, z, w] or [w, x, y, z]?
            # VGN uses scipy convention [x, y, z, w]
            orientation = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

            # Gripper width
            width = float(width_vol[i, j, k]) * self.tsdf_size

            grasps.append({
                "position": pos_world,
                "orientation": orientation,
                "width": np.clip(width, 0.033, 0.10),  # Clamp to valid range
            })
            scores.append(float(score))

        return grasps, scores

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

        # NOTE: Removed hollow object fallback to allow CGN to handle all objects.
        # CGN can predict grasps for hollow objects (bowls, mugs) too.
        # The previous fallback was preventing CGN from ever running on bowl tasks.

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
            # Using 15cm threshold - CGN grasps can be offset from object center
            obj_position = obj_pose[:3]
            max_grasp_distance = 0.15  # 15cm max from object center

            filtered_grasps = []
            for g in grasps:
                dist = np.linalg.norm(g["position"] - obj_position)
                if dist <= max_grasp_distance:
                    g["distance_to_object"] = dist
                    # Prefer grasps on the near side of the object (Y < object Y)
                    # Robot is at Y- so grasps with lower Y are more reachable
                    y_bias = obj_position[1] - g["position"][1]  # Positive if grasp is toward robot
                    g["reachability_score"] = y_bias  # Higher = better
                    filtered_grasps.append(g)

            info["n_candidates_raw"] = len(grasps)
            info["n_candidates_filtered"] = len(filtered_grasps)

            if len(filtered_grasps) == 0:
                info["fallback"] = True
                info["fallback_reason"] = f"no_grasps_near_object (closest: {min(np.linalg.norm(g['position'] - obj_position) for g in grasps):.3f}m)"
                return self._fallback_heuristic(obj_pose, obj_name, approach_strategy, info)

            # Filter for near-vertical approach directions (for top_down strategy)
            # The robosuite controller works best with grasps that approach from above
            # Approach direction should have negative Z component (pointing down)
            if approach_strategy == "top_down":
                vertical_grasps = []
                for g in filtered_grasps:
                    # Get approach direction from rotation matrix (Z-axis)
                    if "orientation" in g and g["orientation"] is not None:
                        rot = g["orientation"]
                        if rot.shape == (3, 3):
                            approach_dir = rot[:, 2]  # Z-axis of gripper
                        else:
                            # Quaternion - convert to rotation matrix
                            from scipy.spatial.transform import Rotation as R
                            quat = rot
                            if len(quat) == 4:
                                r = R.from_quat(quat)  # Assumes [x,y,z,w]
                                approach_dir = r.as_matrix()[:, 2]
                            else:
                                approach_dir = np.array([0, 0, -1])
                    else:
                        approach_dir = np.array([0, 0, -1])

                    # Check if approach is roughly downward (Z < -0.3)
                    # or roughly horizontal toward the robot (Y > 0.5 for front grasps)
                    is_downward = approach_dir[2] < -0.3
                    is_front_approach = approach_dir[1] > 0.5

                    if is_downward or is_front_approach:
                        g["approach_direction"] = approach_dir
                        vertical_grasps.append(g)

                info["n_vertical_filtered"] = len(vertical_grasps)

                if len(vertical_grasps) > 0:
                    filtered_grasps = vertical_grasps

            # Sort by combined score: grasp quality + reachability bias
            # Normalize reachability to [0, 1] range and add to score
            # This biases toward grasps on the near side of the object
            for g in filtered_grasps:
                reach = g.get("reachability_score", 0)
                # Clamp and normalize reachability to [0, 0.3] bonus
                reach_bonus = min(0.3, max(0, reach * 5))  # 5cm offset = 0.25 bonus
                g["combined_score"] = g["score"] + reach_bonus

            filtered_grasps.sort(key=lambda g: g["combined_score"], reverse=True)
            grasps = filtered_grasps

            # Convert best grasp to GraspPose
            best_grasp = grasps[0]
            info["n_candidates"] = len(grasps)
            info["best_score"] = float(best_grasp["score"])
            info["best_grasp_distance"] = float(best_grasp["distance_to_object"])

            # Use CGN position but apply heuristic orientation for robosuite compatibility
            # The robosuite OSC controller expects a specific gripper orientation
            # CGN orientations often don't work well with the controller
            default_orientation = np.array([-0.02, 0.707, 0.707, -0.02])  # [w,x,y,z] gripper-down

            # Get grasp position
            grasp_position = best_grasp["position"].copy()

            # For hollow objects, use heuristic rim-grasp position since CGN
            # often selects grasps that are too far from the object center
            # The heuristic rim-grasp works well for small tabletop objects
            obj_name_lower = obj_name.lower()
            is_hollow = any(h in obj_name_lower for h in ['bowl', 'mug', 'cup', 'ramekin'])
            if is_hollow and approach_strategy == "top_down":
                # Use heuristic position (rim offset toward robot)
                # Object radii for hollow objects
                OBJECT_RADII = {
                    'bowl': 0.045, 'mug': 0.035, 'cup': 0.035, 'ramekin': 0.030
                }
                # Find matching object type
                obj_radius = 0.04  # default
                for obj_type, radius in OBJECT_RADII.items():
                    if obj_type in obj_name_lower:
                        obj_radius = radius
                        break

                # Place grasp on rim, offset toward robot (Y-)
                grasp_position[0] = obj_pose[0]  # Keep X centered
                grasp_position[1] = obj_pose[1] - obj_radius  # Offset toward robot
                grasp_position[2] = obj_pose[2] + 0.04  # Rim height
                info["position_adjusted"] = True
                info["adjustment_reason"] = "hollow_object_rim_grasp"
                # Use "rim" strategy to trigger heuristic code path in grasp.py
                # This avoids orientation oscillation from learned grasp code
                grasp_strategy = "rim"
            else:
                grasp_strategy = "contact_graspnet"

            grasp_pose = GraspPose(
                position=grasp_position,
                orientation=default_orientation,  # Use heuristic orientation
                gripper_width=best_grasp.get("width", 0.04),
                confidence=best_grasp["score"],
                approach_direction=best_grasp.get("approach_direction", None),
                strategy=grasp_strategy,
                metadata={"grasp_index": 0, "original_orientation": best_grasp["orientation"].tolist() if hasattr(best_grasp["orientation"], "tolist") else best_grasp["orientation"]},
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
