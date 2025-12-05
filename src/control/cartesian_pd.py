"""Cartesian PD controller for end-effector pose control.

Implements proportional-derivative control in Cartesian space
for position and orientation tracking.
"""

import numpy as np
from typing import Tuple, Optional

from ..config import SkillConfig


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Compute orientation error as axis-angle representation.
    
    Args:
        q_target: Target quaternion [w, x, y, z].
        q_current: Current quaternion [w, x, y, z].
        
    Returns:
        Orientation error as 3D vector (axis * angle).
    """
    # Error quaternion: q_error = q_target * q_current^-1
    q_error = quat_multiply(q_target, quat_conjugate(q_current))
    
    # Ensure positive w (shorter rotation path)
    if q_error[0] < 0:
        q_error = -q_error
    
    # Convert to axis-angle
    # For small angles: error ≈ 2 * [x, y, z]
    return 2.0 * q_error[1:4]


def pd_control(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    kp_pos: float = 5.0,
    kp_ori: float = 2.0,
    gripper: float = 0.0,
    max_action: float = 1.0,
) -> np.ndarray:
    """Compute PD control action for pose tracking.
    
    Args:
        current_pose: Current 7D pose [x, y, z, qw, qx, qy, qz].
        target_pose: Target 7D pose [x, y, z, qw, qx, qy, qz].
        kp_pos: Position proportional gain.
        kp_ori: Orientation proportional gain.
        gripper: Gripper action (-1 = close, 1 = open).
        max_action: Maximum action magnitude for clipping.
        
    Returns:
        Action vector [dx, dy, dz, drx, dry, drz, gripper].
    """
    # Position error
    pos_error = target_pose[:3] - current_pose[:3]
    
    # Orientation error
    ori_error = quat_error(target_pose[3:7], current_pose[3:7])
    
    # Compute action
    pos_action = kp_pos * pos_error
    ori_action = kp_ori * ori_error
    
    # Combine and clip
    action = np.concatenate([pos_action, ori_action, [gripper]])
    action[:6] = np.clip(action[:6], -max_action, max_action)
    action[6] = np.clip(action[6], -1.0, 1.0)
    
    return action


class CartesianPDController:
    """Cartesian PD controller with configurable gains.
    
    Provides pose tracking with position and orientation control.
    """
    
    def __init__(
        self,
        kp_pos: float = 5.0,
        kp_ori: float = 2.0,
        max_action: float = 1.0,
    ):
        """Initialize controller.
        
        Args:
            kp_pos: Position proportional gain.
            kp_ori: Orientation proportional gain.
            max_action: Maximum action magnitude.
        """
        self.kp_pos = kp_pos
        self.kp_ori = kp_ori
        self.max_action = max_action
        
        # Target state
        self._target_pose: Optional[np.ndarray] = None
        self._gripper_target: float = 0.0
    
    @classmethod
    def from_config(cls, config: SkillConfig) -> "CartesianPDController":
        """Create controller from configuration."""
        return cls(
            kp_pos=config.kp_pos,
            kp_ori=config.kp_ori,
            max_action=config.max_action_magnitude,
        )
    
    def set_target(self, target_pose: np.ndarray, gripper: float = 0.0):
        """Set target pose for controller.
        
        Args:
            target_pose: 7D target pose [x, y, z, qw, qx, qy, qz].
            gripper: Gripper target (-1 = close, 1 = open).
        """
        self._target_pose = target_pose.copy()
        self._gripper_target = gripper
    
    def compute_action(self, current_pose: np.ndarray) -> np.ndarray:
        """Compute control action.
        
        Args:
            current_pose: Current 7D pose.
            
        Returns:
            Action vector [dx, dy, dz, drx, dry, drz, gripper].
        """
        if self._target_pose is None:
            return np.zeros(7)
        
        return pd_control(
            current_pose=current_pose,
            target_pose=self._target_pose,
            kp_pos=self.kp_pos,
            kp_ori=self.kp_ori,
            gripper=self._gripper_target,
            max_action=self.max_action,
        )
    
    def position_error(self, current_pose: np.ndarray) -> float:
        """Get position error magnitude."""
        if self._target_pose is None:
            return 0.0
        return float(np.linalg.norm(self._target_pose[:3] - current_pose[:3]))
    
    def orientation_error(self, current_pose: np.ndarray) -> float:
        """Get orientation error magnitude."""
        if self._target_pose is None:
            return 0.0
        ori_err = quat_error(self._target_pose[3:7], current_pose[3:7])
        return float(np.linalg.norm(ori_err))
    
    def is_at_target(
        self,
        current_pose: np.ndarray,
        pos_threshold: float = 0.03,
        ori_threshold: float = 0.1,
    ) -> bool:
        """Check if current pose is at target within thresholds.
        
        Args:
            current_pose: Current 7D pose.
            pos_threshold: Position error threshold (meters).
            ori_threshold: Orientation error threshold (radians).
            
        Returns:
            True if within both thresholds.
        """
        return (
            self.position_error(current_pose) < pos_threshold and
            self.orientation_error(current_pose) < ori_threshold
        )


def compute_pregrasp_pose(
    object_pose: np.ndarray,
    pregrasp_height: float = 0.10,
    gripper_orientation: Optional[np.ndarray] = None,
    approach_direction: np.ndarray = None,
) -> np.ndarray:
    """Compute pre-grasp pose above object.

    Args:
        object_pose: 7D object pose.
        pregrasp_height: Height above object (meters).
        gripper_orientation: Optional gripper quaternion to use. If None, uses default downward.
        approach_direction: Direction to approach from (default: top-down).

    Returns:
        7D pre-grasp pose.
    """
    pregrasp = np.zeros(7)

    # Set position above object
    pregrasp[:3] = object_pose[:3].copy()
    if approach_direction is None:
        # Top-down approach
        pregrasp[2] += pregrasp_height
    else:
        # Approach from specified direction
        approach_direction = approach_direction / np.linalg.norm(approach_direction)
        pregrasp[:3] += approach_direction * pregrasp_height

    # Set orientation - use provided gripper orientation or default downward
    if gripper_orientation is not None:
        pregrasp[3:7] = gripper_orientation
    else:
        # Default: gripper pointing down (typical robosuite default)
        # This is approximately [-0.02, 0.707, 0.707, -0.02] from observations
        pregrasp[3:7] = np.array([0.0, 0.707107, 0.707107, 0.0])

    return pregrasp
