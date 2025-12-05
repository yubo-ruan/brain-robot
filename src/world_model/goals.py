"""Task goal representation.

Explicit goal specification for tasks, enabling precondition/postcondition checking.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any


class GoalType(Enum):
    """Types of task goals."""
    
    PLACE_ON = auto()      # Place object on surface
    PLACE_IN = auto()      # Place object inside container
    PICK_UP = auto()       # Pick up and hold object
    OPEN = auto()          # Open container/drawer
    CLOSE = auto()         # Close container/drawer
    MOVE_TO = auto()       # Move object to region
    CUSTOM = auto()        # Custom goal with checker function


@dataclass
class TaskGoal:
    """Explicit representation of a task goal.
    
    Enables checking if a goal is satisfied given current world state.
    """
    
    goal_type: GoalType
    
    # Goal parameters
    source_object: Optional[str] = None  # Object to manipulate
    target_location: Optional[str] = None  # Target surface/container/region
    
    # Additional parameters for complex goals
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable description
    description: str = ""
    
    def is_satisfied(self, world_state) -> bool:
        """Check if goal is satisfied by current world state.
        
        Args:
            world_state: WorldState instance.
            
        Returns:
            True if goal conditions are met.
        """
        if self.goal_type == GoalType.PLACE_ON:
            return (
                world_state.on.get(self.source_object) == self.target_location
                and not world_state.is_holding(self.source_object)
            )
        
        elif self.goal_type == GoalType.PLACE_IN:
            return (
                world_state.inside.get(self.source_object) == self.target_location
                and not world_state.is_holding(self.source_object)
            )
        
        elif self.goal_type == GoalType.PICK_UP:
            return world_state.is_holding(self.source_object)
        
        elif self.goal_type == GoalType.OPEN:
            return world_state.open_state.get(self.target_location, False)
        
        elif self.goal_type == GoalType.CLOSE:
            return not world_state.open_state.get(self.target_location, True)
        
        elif self.goal_type == GoalType.MOVE_TO:
            # Check if object is near target region
            # This requires geometric checking
            obj_pos = world_state.get_object_position(self.source_object)
            target_pos = self.params.get("target_position")
            threshold = self.params.get("threshold", 0.1)
            
            if obj_pos is None or target_pos is None:
                return False
            
            import numpy as np
            distance = np.linalg.norm(obj_pos - np.array(target_pos))
            return distance < threshold
        
        elif self.goal_type == GoalType.CUSTOM:
            # Custom checker function
            checker = self.params.get("checker")
            if checker is not None:
                return checker(world_state)
            return False
        
        return False
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "type": self.goal_type.name,
            "source": self.source_object,
            "target": self.target_location,
            "description": self.description,
        }
    
    @classmethod
    def place_on(cls, obj: str, surface: str, description: str = "") -> "TaskGoal":
        """Create a PLACE_ON goal."""
        return cls(
            goal_type=GoalType.PLACE_ON,
            source_object=obj,
            target_location=surface,
            description=description or f"Place {obj} on {surface}",
        )
    
    @classmethod
    def place_in(cls, obj: str, container: str, description: str = "") -> "TaskGoal":
        """Create a PLACE_IN goal."""
        return cls(
            goal_type=GoalType.PLACE_IN,
            source_object=obj,
            target_location=container,
            description=description or f"Place {obj} in {container}",
        )
    
    @classmethod
    def pick_up(cls, obj: str, description: str = "") -> "TaskGoal":
        """Create a PICK_UP goal."""
        return cls(
            goal_type=GoalType.PICK_UP,
            source_object=obj,
            description=description or f"Pick up {obj}",
        )
    
    @classmethod
    def open_container(cls, container: str, description: str = "") -> "TaskGoal":
        """Create an OPEN goal."""
        return cls(
            goal_type=GoalType.OPEN,
            target_location=container,
            description=description or f"Open {container}",
        )


def parse_task_to_goal(task_description: str, object_mapping: Dict[str, str]) -> Optional[TaskGoal]:
    """Parse natural language task to structured goal.
    
    This is a simple heuristic parser. Phase 2 will use Qwen for this.
    
    Args:
        task_description: Natural language task.
        object_mapping: Mapping from role (e.g., "bowl") to object_id.
        
    Returns:
        TaskGoal or None if parsing fails.
    """
    task_lower = task_description.lower()
    
    # Detect "place X in Y" pattern
    if "place" in task_lower or "put" in task_lower:
        if " in " in task_lower or " into " in task_lower:
            # PLACE_IN goal
            # This is simplified - real parsing would be more sophisticated
            return TaskGoal(
                goal_type=GoalType.PLACE_IN,
                source_object=object_mapping.get("source"),
                target_location=object_mapping.get("target"),
                description=task_description,
            )
        elif " on " in task_lower:
            # PLACE_ON goal
            return TaskGoal(
                goal_type=GoalType.PLACE_ON,
                source_object=object_mapping.get("source"),
                target_location=object_mapping.get("target"),
                description=task_description,
            )
    
    # Detect "pick up X" pattern
    if "pick up" in task_lower or "pick" in task_lower or "grab" in task_lower:
        return TaskGoal(
            goal_type=GoalType.PICK_UP,
            source_object=object_mapping.get("source"),
            description=task_description,
        )
    
    # Detect "open X" pattern
    if "open" in task_lower:
        return TaskGoal(
            goal_type=GoalType.OPEN,
            target_location=object_mapping.get("target"),
            description=task_description,
        )
    
    return None
