"""Perception interface for src.

Abstract base class defining the perception API.
All perception implementations (oracle, learned) must follow this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class PerceptionResult:
    """Result from perception system.

    All poses are in world frame (robosuite default).
    Pose format: [x, y, z, qw, qx, qy, qz] - position + quaternion.
    """

    # Object poses: {object_name: 7D pose}
    objects: Dict[str, np.ndarray] = field(default_factory=dict)

    # Object names (for convenience)
    object_names: List[str] = field(default_factory=list)

    # Spatial relations (computed from poses)
    on: Dict[str, str] = field(default_factory=dict)  # obj → surface it's on
    inside: Dict[str, str] = field(default_factory=dict)  # obj → container it's in

    # Gripper state
    gripper_pose: Optional[np.ndarray] = None  # 7D pose
    gripper_width: float = 0.0  # Current gripper opening

    # Robot proprioception
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None

    # Metadata
    timestamp: float = 0.0
    frame: str = "world"  # Coordinate frame
    
    def get_object_position(self, name: str) -> Optional[np.ndarray]:
        """Get position (xyz) of named object."""
        if name in self.objects:
            return self.objects[name][:3].copy()
        return None
    
    def get_object_orientation(self, name: str) -> Optional[np.ndarray]:
        """Get orientation (quaternion) of named object."""
        if name in self.objects:
            return self.objects[name][3:7].copy()
        return None
    
    def get_gripper_position(self) -> Optional[np.ndarray]:
        """Get gripper position (xyz)."""
        if self.gripper_pose is not None:
            return self.gripper_pose[:3].copy()
        return None


class PerceptionInterface(ABC):
    """Abstract base class for perception systems.
    
    All perception implementations must implement the perceive() method.
    This ensures consistent API across oracle, noisy, and learned perception.
    """
    
    @abstractmethod
    def perceive(self, env) -> PerceptionResult:
        """Extract perception from environment.
        
        Args:
            env: The robosuite/LIBERO environment.
            
        Returns:
            PerceptionResult containing objects, gripper, and robot state.
        """
        pass
    
    def reset(self):
        """Reset any internal state. Override if needed."""
        pass
