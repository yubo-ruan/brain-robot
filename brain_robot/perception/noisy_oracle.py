"""Noisy oracle perception for robustness testing.

Adds configurable Gaussian noise to ground truth perception.
Used to test skill robustness before full learned perception.
"""

import numpy as np
from typing import Optional

from .interface import PerceptionInterface, PerceptionResult
from .oracle import OraclePerception
from ..config import PerceptionConfig


class NoisyOraclePerception(PerceptionInterface):
    """Oracle perception with added Gaussian noise.
    
    Wraps OraclePerception and adds configurable noise to positions
    and orientations. Useful for testing skill robustness.
    """
    
    def __init__(
        self,
        pos_noise_std: float = 0.01,  # 1cm position noise
        ori_noise_std: float = 0.05,  # ~3 degree orientation noise
        seed: Optional[int] = None,
    ):
        """Initialize noisy oracle.
        
        Args:
            pos_noise_std: Standard deviation for position noise (meters).
            ori_noise_std: Standard deviation for orientation noise (radians).
            seed: Optional random seed for reproducibility.
        """
        self.pos_noise_std = pos_noise_std
        self.ori_noise_std = ori_noise_std
        self.oracle = OraclePerception()
        self.rng = np.random.RandomState(seed)
    
    @classmethod
    def from_config(cls, config: PerceptionConfig, seed: Optional[int] = None):
        """Create from configuration."""
        return cls(
            pos_noise_std=config.noise_pos_std,
            ori_noise_std=config.noise_ori_std,
            seed=seed,
        )
    
    def perceive(self, env) -> PerceptionResult:
        """Extract perception with added noise.
        
        Args:
            env: The robosuite/LIBERO environment.
            
        Returns:
            PerceptionResult with noisy poses.
        """
        # Get ground truth
        result = self.oracle.perceive(env)
        
        # Add noise to object poses
        noisy_objects = {}
        for name, pose in result.objects.items():
            noisy_objects[name] = self._add_pose_noise(pose)
        result.objects = noisy_objects
        
        # Add noise to gripper pose
        if result.gripper_pose is not None:
            result.gripper_pose = self._add_pose_noise(result.gripper_pose)
        
        return result
    
    def _add_pose_noise(self, pose: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to a 7D pose.
        
        Args:
            pose: [x, y, z, qw, qx, qy, qz]
            
        Returns:
            Noisy pose with same format.
        """
        noisy_pose = pose.copy()
        
        # Position noise
        noisy_pose[:3] += self.rng.normal(0, self.pos_noise_std, 3)
        
        # Orientation noise (add small rotation)
        if self.ori_noise_std > 0:
            noise_axis = self.rng.normal(0, 1, 3)
            noise_axis = noise_axis / (np.linalg.norm(noise_axis) + 1e-8)
            noise_angle = self.rng.normal(0, self.ori_noise_std)
            noise_quat = self._axis_angle_to_quat(noise_axis, noise_angle)
            noisy_pose[3:7] = self._quat_multiply(noisy_pose[3:7], noise_quat)
            # Normalize quaternion
            noisy_pose[3:7] /= np.linalg.norm(noisy_pose[3:7])
        
        return noisy_pose
    
    def _axis_angle_to_quat(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion [w, x, y, z]."""
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def reset(self):
        """Reset internal state."""
        self.oracle.reset()
    
    def set_seed(self, seed: int):
        """Set random seed for noise generation."""
        self.rng = np.random.RandomState(seed)
