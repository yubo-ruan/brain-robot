"""World model module for brain_robot.

Provides symbolic world state representation and goal specification.
"""

from .state import WorldState, ObjectState
from .goals import TaskGoal, GoalType

__all__ = [
    "WorldState",
    "ObjectState",
    "TaskGoal",
    "GoalType",
]
