"""Depth-based 3D position estimation for detected objects.

Combines 2D YOLO detections with depth images to estimate 3D object positions
in the world frame. Uses camera intrinsics and extrinsics for projection.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass
class CameraInfo:
    """Camera intrinsic and extrinsic parameters."""

    # Image dimensions
    width: int = 128
    height: int = 128

    # Intrinsic parameters
    fx: float = 154.51  # Focal length X (computed from fovy=45, h=128)
    fy: float = 154.51  # Focal length Y
    cx: float = 64.0    # Principal point X
    cy: float = 64.0    # Principal point Y

    # Extrinsic: camera pose in world frame [4x4 homogeneous transform]
    # This is T_world_camera: transforms points from camera to world frame
    extrinsic: Optional[np.ndarray] = None

    # Depth scale (for converting raw depth to meters)
    # MuJoCo depth is normalized [0, 1], needs conversion
    depth_near: float = 0.01   # Near clipping plane (meters)
    depth_far: float = 10.0    # Far clipping plane (meters)

    @classmethod
    def from_mujoco_camera(
        cls,
        sim,
        camera_name: str = "agentview",
        width: int = 128,
        height: int = 128,
    ) -> "CameraInfo":
        """Create CameraInfo from MuJoCo simulation camera.

        MuJoCo camera convention:
        - Camera looks down its local -Z axis
        - Local +X is right, +Y is up, +Z is backward (out of camera)
        - The xmat gives rotation from world to camera orientation

        Args:
            sim: MuJoCo simulation
            camera_name: Name of camera in MJCF
            width: Image width
            height: Image height

        Returns:
            CameraInfo with parameters extracted from simulation
        """
        try:
            cam_id = sim.model.camera_name2id(camera_name)
            fovy = sim.model.cam_fovy[cam_id]

            # Compute intrinsics from vertical FOV
            fy = height / (2 * np.tan(np.radians(fovy) / 2))
            fx = fy  # Assuming square pixels
            cx, cy = width / 2, height / 2

            # Get camera pose from simulation
            cam_xpos = sim.data.cam_xpos[cam_id].copy()
            cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()

            # MuJoCo xmat is the rotation matrix of the camera frame in world coords
            # For a point in camera frame P_cam, the world position is:
            #   P_world = cam_xmat @ P_cam + cam_xpos
            # So T_world_camera = [cam_xmat | cam_xpos]

            # But MuJoCo camera looks down -Z, and the image coordinate system is:
            # - u (right) = +X camera
            # - v (down) = -Y camera
            # - depth = +Z camera (distance along view ray)

            # We need to flip Y and Z to match standard vision convention
            # Vision: X-right, Y-down, Z-forward
            # MuJoCo camera: X-right, Y-up, Z-backward
            # Transform: vision = flip @ mujoco, where flip = diag(1, -1, -1)
            flip = np.diag([1.0, -1.0, -1.0])

            # Build T_world_camera
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = cam_xmat @ flip  # Apply flip to camera frame
            extrinsic[:3, 3] = cam_xpos

            return cls(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                extrinsic=extrinsic,
            )

        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not get camera params: {e}")
            return cls(width=width, height=height)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


def normalize_depth_to_meters(
    depth: np.ndarray,
    near: float = 0.01,
    far: float = 10.0,
) -> np.ndarray:
    """Convert normalized MuJoCo depth [0, 1] to meters.

    MuJoCo/robosuite renders depth in OpenGL convention where:
    - Depth is stored as normalized [0, 1] in OpenGL NDC
    - 0 = near plane, 1 = far plane (approximately, after remapping)

    The conversion uses the OpenGL depth buffer formula:
        z_ndc = 2 * depth - 1  (remap [0,1] to [-1,1])
        z_eye = 2 * near * far / (far + near - z_ndc * (far - near))

    Args:
        depth: Normalized depth image [0, 1]
        near: Near clipping plane (meters)
        far: Far clipping plane (meters)

    Returns:
        Depth in meters (distance from camera along view axis)
    """
    # Avoid edge cases
    depth = np.clip(depth, 0.001, 0.999)

    # OpenGL depth buffer to linear depth
    # z_ndc in [-1, 1], z_eye is linear depth
    z_ndc = 2.0 * depth - 1.0
    z_eye = 2.0 * near * far / (far + near - z_ndc * (far - near))

    return z_eye


def pixel_to_camera_frame(
    u: float,
    v: float,
    depth: float,
    camera_info: CameraInfo,
) -> np.ndarray:
    """Convert pixel coordinates + depth to 3D point in camera frame.

    Camera frame convention (OpenGL/MuJoCo):
    - X: right
    - Y: up
    - Z: backward (out of camera)

    Args:
        u: Pixel X coordinate
        v: Pixel Y coordinate
        depth: Depth in meters
        camera_info: Camera parameters

    Returns:
        3D point in camera frame [x, y, z]
    """
    # Unproject to camera frame
    x = (u - camera_info.cx) * depth / camera_info.fx
    y = (v - camera_info.cy) * depth / camera_info.fy
    z = depth

    return np.array([x, y, z])


def camera_to_world_frame(
    point_camera: np.ndarray,
    camera_info: CameraInfo,
) -> np.ndarray:
    """Transform point from camera frame to world frame.

    Args:
        point_camera: 3D point in camera frame [x, y, z]
        camera_info: Camera parameters with extrinsic

    Returns:
        3D point in world frame
    """
    if camera_info.extrinsic is None:
        raise ValueError("Camera extrinsic not available")

    # Homogeneous coordinates
    point_homo = np.array([*point_camera, 1.0])

    # Transform to world frame
    point_world = camera_info.extrinsic @ point_homo

    return point_world[:3]


def estimate_object_position_from_bbox(
    bbox: List[float],
    depth_image: np.ndarray,
    camera_info: CameraInfo,
    depth_percentile: float = 25.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate 3D object position from 2D bounding box and depth.

    Uses the median depth within the bounding box region, which is more
    robust than center pixel depth for objects with complex shapes.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_image: Depth image (H, W) or (H, W, 1)
        camera_info: Camera parameters
        depth_percentile: Percentile of depth values to use (lower = closer)

    Returns:
        Tuple of (position_world, info_dict)
    """
    info = {}

    # Handle depth shape
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    # Get bbox region
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(camera_info.width, x2)
    y2 = min(camera_info.height, y2)

    # Extract depth region
    depth_region = depth_image[y1:y2, x1:x2]
    info["bbox_pixels"] = (x2 - x1) * (y2 - y1)

    if depth_region.size == 0:
        return np.zeros(3), {"error": "empty_bbox"}

    # Convert normalized depth to meters
    depth_meters = normalize_depth_to_meters(
        depth_region,
        camera_info.depth_near,
        camera_info.depth_far,
    )

    # Use percentile depth (lower = closer to camera = object surface)
    valid_depths = depth_meters[depth_meters > 0.01]
    if len(valid_depths) == 0:
        return np.zeros(3), {"error": "no_valid_depth"}

    object_depth = np.percentile(valid_depths, depth_percentile)
    info["depth_meters"] = float(object_depth)
    info["depth_std"] = float(np.std(valid_depths))

    # Use bbox center as pixel coordinates
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    info["center_uv"] = (u, v)

    # Project to camera frame
    point_camera = pixel_to_camera_frame(u, v, object_depth, camera_info)
    info["point_camera"] = point_camera.tolist()

    # Transform to world frame
    try:
        point_world = camera_to_world_frame(point_camera, camera_info)
        info["point_world"] = point_world.tolist()
        return point_world, info
    except ValueError as e:
        info["error"] = str(e)
        return point_camera, info


def extract_object_pointcloud(
    bbox: List[float],
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    camera_info: CameraInfo,
    downsample: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud for object within bounding box.

    Useful for GIGA grasp prediction which needs object point clouds.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_image: Depth image (H, W) or (H, W, 1)
        rgb_image: RGB image (H, W, 3)
        camera_info: Camera parameters
        downsample: Downsampling factor for efficiency

    Returns:
        Tuple of (points_world, colors) where:
        - points_world: (N, 3) point cloud in world frame
        - colors: (N, 3) RGB colors normalized [0, 1]
    """
    # Handle depth shape
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    # Get bbox region with downsampling
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(camera_info.width, x2)
    y2 = min(camera_info.height, y2)

    # Generate pixel grid
    us = np.arange(x1, x2, downsample)
    vs = np.arange(y1, y2, downsample)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    # Get depths
    depths_norm = depth_image[vv, uu]
    depths = normalize_depth_to_meters(
        depths_norm,
        camera_info.depth_near,
        camera_info.depth_far,
    )

    # Filter invalid depths
    valid = (depths > 0.01) & (depths < 5.0)
    uu = uu[valid]
    vv = vv[valid]
    depths = depths[valid]

    if len(depths) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Project to camera frame
    x_cam = (uu - camera_info.cx) * depths / camera_info.fx
    y_cam = (vv - camera_info.cy) * depths / camera_info.fy
    z_cam = depths

    points_camera = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Transform to world frame
    if camera_info.extrinsic is not None:
        points_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
        points_world = (camera_info.extrinsic @ points_homo.T).T[:, :3]
    else:
        points_world = points_camera

    # Get colors
    colors = rgb_image[vv, uu].astype(np.float32) / 255.0

    return points_world, colors


class DepthPositionEstimator:
    """Estimates 3D object positions from YOLO detections + depth.

    Integrates with YOLO detector to provide full 3D object localization.
    Updates Detection objects in-place with estimated positions.
    """

    def __init__(
        self,
        camera_name: str = "agentview",
        depth_percentile: float = 25.0,
    ):
        """Initialize estimator.

        Args:
            camera_name: Name of camera to use
            depth_percentile: Percentile of depth values to use
        """
        self.camera_name = camera_name
        self.depth_percentile = depth_percentile
        self.camera_info: Optional[CameraInfo] = None

    def update_camera_info(self, sim) -> None:
        """Update camera info from simulation.

        Should be called when environment resets or camera moves.

        Args:
            sim: MuJoCo simulation
        """
        self.camera_info = CameraInfo.from_mujoco_camera(
            sim, self.camera_name
        )

    def estimate_positions(
        self,
        detections: List,
        depth_image: np.ndarray,
        sim=None,
    ) -> List:
        """Estimate 3D positions for all detections.

        Args:
            detections: List of Detection objects (from YOLO)
            depth_image: Depth image
            sim: MuJoCo simulation (for updating camera if needed)

        Returns:
            Updated detections with position estimates
        """
        # Update camera info if needed
        if sim is not None and self.camera_info is None:
            self.update_camera_info(sim)

        if self.camera_info is None:
            print("Warning: No camera info available")
            return detections

        for det in detections:
            if det.bbox is None:
                continue

            position, info = estimate_object_position_from_bbox(
                det.bbox,
                depth_image,
                self.camera_info,
                self.depth_percentile,
            )

            # Update detection
            det.position = position
            if hasattr(det, 'metadata'):
                det.metadata = det.metadata or {}
                det.metadata['depth_info'] = info

        return detections

    def get_pointcloud(
        self,
        bbox: List[float],
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract point cloud for a bounding box region.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_image: Depth image
            rgb_image: RGB image

        Returns:
            Tuple of (points, colors)
        """
        if self.camera_info is None:
            raise ValueError("Camera info not initialized")

        return extract_object_pointcloud(
            bbox, depth_image, rgb_image, self.camera_info
        )
