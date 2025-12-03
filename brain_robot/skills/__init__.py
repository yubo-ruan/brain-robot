"""Skills module for brain_robot.

Provides composable motor primitives for robot manipulation.
"""

from .base import Skill, SkillResult
from .approach import ApproachSkill
from .grasp import GraspSkill
from .move import MoveSkill
from .place import PlaceSkill
from .drawer import OpenDrawerSkill, CloseDrawerSkill
from .stove import TurnOnStoveSkill, TurnOffStoveSkill
from .grasp_selection import (
    GraspSelector,
    GraspPose,
    HeuristicGraspSelector,
    GIGAGraspSelector,
    get_grasp_selector,
    register_grasp_selector,
)

__all__ = [
    # Base classes
    "Skill",
    "SkillResult",
    # Manipulation skills
    "ApproachSkill",
    "GraspSkill",
    "MoveSkill",
    "PlaceSkill",
    "OpenDrawerSkill",
    "CloseDrawerSkill",
    "TurnOnStoveSkill",
    "TurnOffStoveSkill",
    # Grasp selection
    "GraspSelector",
    "GraspPose",
    "HeuristicGraspSelector",
    "GIGAGraspSelector",
    "get_grasp_selector",
    "register_grasp_selector",
]


# Skill registry for easy lookup by name
SKILL_REGISTRY = {
    "ApproachObject": ApproachSkill,
    "GraspObject": GraspSkill,
    "MoveObjectToRegion": MoveSkill,
    "PlaceObject": PlaceSkill,
    "OpenDrawer": OpenDrawerSkill,
    "CloseDrawer": CloseDrawerSkill,
    "TurnOnStove": TurnOnStoveSkill,
    "TurnOffStove": TurnOffStoveSkill,
}


def get_skill(name: str, **kwargs):
    """Get skill instance by name.
    
    Args:
        name: Skill name (e.g., "ApproachObject").
        **kwargs: Arguments to pass to skill constructor.
        
    Returns:
        Skill instance.
        
    Raises:
        KeyError: If skill name not found.
    """
    if name not in SKILL_REGISTRY:
        raise KeyError(f"Unknown skill: {name}. Available: {list(SKILL_REGISTRY.keys())}")
    return SKILL_REGISTRY[name](**kwargs)
