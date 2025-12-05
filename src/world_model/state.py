"""Symbolic world state representation.

WorldState tracks symbolic relations (holding, on, inside) and object states.
This enables multi-step reasoning and skill precondition checking.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
import numpy as np

from ..perception.interface import PerceptionResult


@dataclass
class ObjectState:
    """Per-object state tracking.
    
    Tracks both geometric (pose) and semantic (role) information.
    """
    
    name: str
    pose: np.ndarray  # 7D pose [x, y, z, qw, qx, qy, qz]
    last_seen: float  # Timestamp when last observed
    confidence: float = 1.0  # For learned perception (1.0 for oracle)
    
    # Semantic properties (can be set by Qwen grounding)
    object_type: Optional[str] = None  # e.g., "bowl", "plate"
    color: Optional[str] = None
    material: Optional[str] = None
    
    @property
    def position(self) -> np.ndarray:
        """Get position component."""
        return self.pose[:3].copy()
    
    @property
    def orientation(self) -> np.ndarray:
        """Get orientation quaternion."""
        return self.pose[3:7].copy()
    
    def to_dict(self) -> dict:
        """Serialize for JSON/Qwen prompt."""
        return {
            "name": self.name,
            "position": self.pose[:3].tolist(),
            "orientation": self.pose[3:7].tolist(),
            "type": self.object_type,
            "color": self.color,
        }


@dataclass
class WorldState:
    """Symbolic world state with explicit uncertainty handling.
    
    Tracks:
    - holding: Currently held object (or None)
    - on: Object → surface relations
    - inside: Object → container relations
    - open_state: Container open/closed states
    - objects: Full ObjectState for each detected object
    """
    
    # Core symbolic relations
    holding: Optional[str] = None  # Currently held object
    on: Dict[str, str] = field(default_factory=dict)  # obj → surface
    inside: Dict[str, str] = field(default_factory=dict)  # obj → container
    open_state: Dict[str, bool] = field(default_factory=dict)  # container → bool
    
    # Object tracking
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    
    # Gripper state
    gripper_pose: Optional[np.ndarray] = None
    gripper_width: float = 0.0
    
    # Task grounding (set by Qwen semantic grounding)
    task_source: Optional[str] = None  # Object to manipulate
    task_target: Optional[str] = None  # Target location/container
    
    # Uncertainty tracking
    perception_stale: bool = False
    last_perception_time: float = 0.0
    stale_threshold: float = 1.0  # seconds
    
    def update_from_perception(self, perception: PerceptionResult):
        """Update world state from perception result.

        Reconciles symbolic state with new perception data.
        Handles disappeared objects and updates timestamps.
        """
        current_time = perception.timestamp or time.time()

        # Track which objects we see
        perceived_objects = set(perception.object_names)
        known_objects = set(self.objects.keys())

        # Handle disappeared objects
        disappeared = known_objects - perceived_objects
        for obj in disappeared:
            # Mark relations as uncertain (or remove them)
            if obj in self.on:
                del self.on[obj]
            if obj in self.inside:
                del self.inside[obj]
            # Keep object in memory but mark as stale
            if obj in self.objects:
                self.objects[obj].confidence = 0.5

        # Update existing objects and add new ones
        for name in perception.object_names:
            if name in perception.objects:
                pose = perception.objects[name]
                if name in self.objects:
                    # Update existing
                    self.objects[name].pose = pose
                    self.objects[name].last_seen = current_time
                    self.objects[name].confidence = 1.0
                else:
                    # Add new
                    self.objects[name] = ObjectState(
                        name=name,
                        pose=pose,
                        last_seen=current_time,
                    )

        # Update spatial relations from perception
        # Only update for objects we're not holding (held objects have no spatial relation)
        if perception.on:
            for obj, surface in perception.on.items():
                if obj != self.holding:
                    self.on[obj] = surface
        if perception.inside:
            for obj, container in perception.inside.items():
                if obj != self.holding:
                    self.inside[obj] = container
                    # Inside overrides on
                    if obj in self.on:
                        del self.on[obj]

        # Update gripper state
        self.gripper_pose = perception.gripper_pose
        self.gripper_width = perception.gripper_width

        # Update timestamps
        self.last_perception_time = current_time
        self.perception_stale = False
    
    def mark_stale_if_needed(self, current_time: Optional[float] = None):
        """Check if perception data is stale."""
        if current_time is None:
            current_time = time.time()
        
        if current_time - self.last_perception_time > self.stale_threshold:
            self.perception_stale = True
    
    def get_object_pose(self, name: str) -> Optional[np.ndarray]:
        """Get pose of named object."""
        if name in self.objects:
            return self.objects[name].pose.copy()
        return None
    
    def get_object_position(self, name: str) -> Optional[np.ndarray]:
        """Get position of named object."""
        if name in self.objects:
            return self.objects[name].position
        return None
    
    def get_gripper_position(self) -> Optional[np.ndarray]:
        """Get current gripper position."""
        if self.gripper_pose is not None:
            return self.gripper_pose[:3].copy()
        return None
    
    def is_holding(self, obj: Optional[str] = None) -> bool:
        """Check if holding anything (or specific object)."""
        if obj is None:
            return self.holding is not None
        return self.holding == obj
    
    def is_gripper_empty(self) -> bool:
        """Check if gripper is not holding anything."""
        return self.holding is None
    
    def set_holding(self, obj: Optional[str]):
        """Update holding state.
        
        When picking up an object, removes it from on/inside relations.
        """
        if obj is not None:
            # Remove from surface relations
            if obj in self.on:
                del self.on[obj]
            if obj in self.inside:
                del self.inside[obj]
        self.holding = obj
    
    def set_on(self, obj: str, surface: str):
        """Set object on surface relation."""
        self.on[obj] = surface
        # Remove from inside if was there
        if obj in self.inside:
            del self.inside[obj]
    
    def set_inside(self, obj: str, container: str):
        """Set object inside container relation."""
        self.inside[obj] = container
        # Remove from on if was there
        if obj in self.on:
            del self.on[obj]
    
    def to_dict(self) -> dict:
        """Serialize for JSON/Qwen prompt.
        
        Returns human-readable symbolic state.
        """
        return {
            "holding": self.holding,
            "on": dict(self.on),
            "inside": dict(self.inside),
            "open": dict(self.open_state),
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "gripper_position": self.gripper_pose[:3].tolist() if self.gripper_pose is not None else None,
        }
    
    def copy(self) -> "WorldState":
        """Create a copy of the world state."""
        new_state = WorldState(
            holding=self.holding,
            on=dict(self.on),
            inside=dict(self.inside),
            open_state=dict(self.open_state),
            gripper_pose=self.gripper_pose.copy() if self.gripper_pose is not None else None,
            gripper_width=self.gripper_width,
            task_source=self.task_source,
            task_target=self.task_target,
            perception_stale=self.perception_stale,
            last_perception_time=self.last_perception_time,
        )
        # Deep copy objects
        for name, obj in self.objects.items():
            new_state.objects[name] = ObjectState(
                name=obj.name,
                pose=obj.pose.copy(),
                last_seen=obj.last_seen,
                confidence=obj.confidence,
                object_type=obj.object_type,
                color=obj.color,
                material=obj.material,
            )
        return new_state
