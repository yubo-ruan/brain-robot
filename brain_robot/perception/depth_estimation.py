"""Depth-based 3D position estimation for detected objects.

Combines 2D YOLO detections with depth images to estimate 3D object positions
in the world frame. Uses camera intrinsics and extrinsics from robosuite.

Key insight: Uses robosuite's camera utilities for correct coordinate transforms
and depth conversion. The depth image from MuJoCo is normalized [0,1] and must
be converted using the model's extent and clip planes.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Import robosuite camera utilities for correct transforms
try:
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
        get_real_depth_map,
    )
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False


@dataclass
class CameraInfo:
    """Camera intrinsic and extrinsic parameters.

    Uses robosuite conventions for camera coordinate system:
    - Extrinsic is T_world_camera (camera pose in world frame)
    - Includes axis correction so Z points forward (OpenCV convention)
    """

    # Image dimensions
    width: int = 128
    height: int = 128

    # Intrinsic parameters (3x3 matrix K)
    intrinsic: Optional[np.ndarray] = None

    # For convenience, also store individual values
    fx: float = 154.51  # Focal length X
    fy: float = 154.51  # Focal length Y
    cx: float = 64.0    # Principal point X
    cy: float = 64.0    # Principal point Y

    # Extrinsic: camera pose in world frame [4x4 homogeneous transform]
    # This is T_world_camera with robosuite's axis correction applied
    extrinsic: Optional[np.ndarray] = None

    # Camera name used to create this CameraInfo
    camera_name: str = "agentview"

    # Reference to sim for depth conversion
    _sim: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def from_mujoco_camera(
        cls,
        sim,
        camera_name: str = "agentview",
        width: int = 128,
        height: int = 128,
    ) -> "CameraInfo":
        """Create CameraInfo from MuJoCo simulation camera.

        Uses robosuite's camera utilities which handle the MuJoCo->OpenCV
        coordinate transform correctly.

        Args:
            sim: MuJoCo simulation
            camera_name: Name of camera in MJCF
            width: Image width
            height: Image height

        Returns:
            CameraInfo with parameters extracted from simulation
        """
        if not HAS_ROBOSUITE:
            print("Warning: robosuite not available, using default camera params")
            return cls(width=width, height=height)

        try:
            # Use robosuite's camera utilities for correct transforms
            K = get_camera_intrinsic_matrix(sim, camera_name, height, width)
            T_cam_world = get_camera_extrinsic_matrix(sim, camera_name)

            # Extract individual intrinsic values
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            return cls(
                width=width,
                height=height,
                intrinsic=K,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                extrinsic=T_cam_world,
                camera_name=camera_name,
                _sim=sim,
            )

        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not get camera params: {e}")
            return cls(width=width, height=height)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        if self.intrinsic is not None:
            return self.intrinsic
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def convert_depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw MuJoCo depth to meters using correct formula.

        Args:
            depth_raw: Raw depth image from MuJoCo [0, 1]

        Returns:
            Depth in meters
        """
        if self._sim is not None and HAS_ROBOSUITE:
            return get_real_depth_map(self._sim, depth_raw)
        else:
            # Fallback to default conversion (less accurate)
            return normalize_depth_to_meters(depth_raw)


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
    depth_percentile: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate 3D object position from 2D bounding box and depth.

    Uses a low percentile of depth values (close to minimum) to get the
    object surface rather than background. The minimum depth in the bbox
    region corresponds to the closest surface, which is the object.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_image: Depth image (H, W) or (H, W, 1), raw from MuJoCo [0,1]
        camera_info: Camera parameters (with _sim for depth conversion)
        depth_percentile: Percentile of depth values to use (lower = closer surface)

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

    # Extract depth region (raw values)
    depth_region_raw = depth_image[y1:y2, x1:x2]
    info["bbox_pixels"] = (x2 - x1) * (y2 - y1)

    if depth_region_raw.size == 0:
        return np.zeros(3), {"error": "empty_bbox"}

    # Convert entire depth image to meters using robosuite's correct formula
    if camera_info._sim is not None and HAS_ROBOSUITE:
        depth_image_meters = get_real_depth_map(camera_info._sim, depth_image)
    else:
        depth_image_meters = normalize_depth_to_meters(depth_image)

    # Extract the region in meters
    depth_region_meters = depth_image_meters[y1:y2, x1:x2]

    # Use low percentile depth (closest surface = object, not background)
    # Minimum is noisy, so use 5th percentile for robustness
    valid_depths = depth_region_meters[depth_region_meters > 0.1]  # Filter invalid
    if len(valid_depths) == 0:
        return np.zeros(3), {"error": "no_valid_depth"}

    object_depth = np.percentile(valid_depths, depth_percentile)
    info["depth_meters"] = float(object_depth)
    info["depth_min"] = float(valid_depths.min())
    info["depth_max"] = float(valid_depths.max())
    info["depth_std"] = float(np.std(valid_depths))

    # Use bbox center as pixel coordinates
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    info["center_uv"] = (u, v)

    # Unproject to camera frame using intrinsics
    K = camera_info.get_intrinsic_matrix()
    x_cam = (u - K[0, 2]) * object_depth / K[0, 0]
    y_cam = (v - K[1, 2]) * object_depth / K[1, 1]
    z_cam = object_depth
    point_camera = np.array([x_cam, y_cam, z_cam])
    info["point_camera"] = point_camera.tolist()

    # Transform to world frame
    try:
        point_world = camera_to_world_frame(point_camera, camera_info)
        info["point_world"] = point_world.tolist()
        return point_world, info
    except ValueError as e:
        info["error"] = str(e)
        return point_camera, info


def extract_object_pointcloud_oracle(
    sim,
    camera_name: str,
    bbox: List[float],
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    downsample: int = 2,
    depth_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud using robosuite's camera utilities (ORACLE).

    Uses robosuite's camera intrinsics and extrinsics for accurate
    coordinate transforms. Filters points to keep only the object
    surface (closest points in bbox).

    Args:
        sim: MuJoCo simulation (for camera parameters)
        camera_name: Camera name (e.g., "agentview")
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_image: Depth image (H, W) or (H, W, 1), raw from MuJoCo [0,1]
        rgb_image: RGB image (H, W, 3)
        downsample: Downsampling factor for efficiency
        depth_threshold: Keep points within this distance of min depth (meters)

    Returns:
        Tuple of (points_world, colors) where:
        - points_world: (N, 3) point cloud in world frame
        - colors: (N, 3) RGB colors normalized [0, 1]
    """
    if not HAS_ROBOSUITE:
        return np.zeros((0, 3)), np.zeros((0, 3))

    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    # Handle depth shape
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    height, width = depth_image.shape

    # Convert depth to meters using robosuite's correct formula
    depth_meters = get_real_depth_map(sim, depth_image)
    if depth_meters.ndim == 3:
        depth_meters = depth_meters[:, :, 0]

    # Get bbox region
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Generate pixel grid with downsampling
    us = np.arange(x1, x2, downsample)  # columns (x)
    vs = np.arange(y1, y2, downsample)  # rows (y)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten().astype(np.float64)
    vv = vv.flatten().astype(np.float64)

    # Get depths at sampled pixels
    depths = depth_meters[vv.astype(int), uu.astype(int)]

    # Filter to keep only object surface (closest points)
    valid_mask = depths > 0.1
    if valid_mask.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    min_depth = depths[valid_mask].min()
    object_mask = valid_mask & (depths < min_depth + depth_threshold)

    uu_valid = uu[object_mask]
    vv_valid = vv[object_mask]

    if len(uu_valid) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Get camera intrinsic and extrinsic matrices
    K = get_camera_intrinsic_matrix(sim, camera_name, height, width)  # 3x3
    R = get_camera_extrinsic_matrix(sim, camera_name)  # 4x4 camera pose in world
    K_inv = np.linalg.inv(K)

    # Get depths for valid points
    z_vals = depth_meters[vv_valid.astype(int), uu_valid.astype(int)]

    # Unproject pixels to camera frame: p_cam = K^-1 @ [u, v, 1] * z
    n_points = len(uu_valid)
    pixels_homo = np.stack([uu_valid, vv_valid, np.ones(n_points)], axis=0)  # (3, N)
    p_cam = (K_inv @ pixels_homo) * z_vals  # (3, N)

    # Transform to world frame: p_world = R @ [p_cam; 1]
    p_cam_homo = np.vstack([p_cam, np.ones((1, n_points))])  # (4, N)
    p_world = R @ p_cam_homo  # (4, N)
    points_world = p_world[:3, :].T  # (N, 3)

    # Get colors
    colors = rgb_image[vv_valid.astype(int), uu_valid.astype(int)].astype(np.float32) / 255.0

    return points_world, colors


def extract_object_pointcloud(
    bbox: List[float],
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    camera_info: CameraInfo,
    downsample: int = 2,
    depth_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud for object within bounding box.

    If sim reference is available in camera_info, uses the oracle method
    with robosuite's tested transforms. Otherwise falls back to manual
    computation.

    Args:
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        depth_image: Depth image (H, W) or (H, W, 1), raw from MuJoCo [0,1]
        rgb_image: RGB image (H, W, 3)
        camera_info: Camera parameters (with _sim for oracle mode)
        downsample: Downsampling factor for efficiency
        depth_threshold: Keep points within this distance of min depth (meters)

    Returns:
        Tuple of (points_world, colors) where:
        - points_world: (N, 3) point cloud in world frame
        - colors: (N, 3) RGB colors normalized [0, 1]
    """
    # Use oracle method if sim is available (recommended)
    if camera_info._sim is not None and HAS_ROBOSUITE:
        return extract_object_pointcloud_oracle(
            sim=camera_info._sim,
            camera_name=camera_info.camera_name,
            bbox=bbox,
            depth_image=depth_image,
            rgb_image=rgb_image,
            downsample=downsample,
            depth_threshold=depth_threshold,
        )

    # Fallback: manual computation (less accurate)
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]

    depth_meters = normalize_depth_to_meters(depth_image)

    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(camera_info.width, x2)
    y2 = min(camera_info.height, y2)

    us = np.arange(x1, x2, downsample)
    vs = np.arange(y1, y2, downsample)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    depths = depth_meters[vv, uu]

    # Filter to object surface
    valid_mask = depths > 0.1
    if valid_mask.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    min_depth = depths[valid_mask].min()
    object_mask = valid_mask & (depths < min_depth + depth_threshold)

    uu = uu[object_mask]
    vv = vv[object_mask]
    depths = depths[object_mask]

    if len(depths) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    K = camera_info.get_intrinsic_matrix()
    x_cam = (uu - K[0, 2]) * depths / K[0, 0]
    y_cam = (vv - K[1, 2]) * depths / K[1, 1]
    z_cam = depths

    points_camera = np.stack([x_cam, y_cam, z_cam], axis=1)

    if camera_info.extrinsic is not None:
        points_homo = np.hstack([points_camera, np.ones((len(points_camera), 1))])
        points_world = (camera_info.extrinsic @ points_homo.T).T[:, :3]
    else:
        points_world = points_camera

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
