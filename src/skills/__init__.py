"""Skills module for src.

Provides composable motor primitives for robot manipulation.
"""

from .base import Skill, SkillResult
from .approach import ApproachSkill
from .grasp import GraspSkill
from .move import MoveSkill
from .place import PlaceSkill

__all__ = [
    "Skill",
    "SkillResult",
    "ApproachSkill",
    "GraspSkill",
    "MoveSkill",
    "PlaceSkill",
]


# Skill registry for easy lookup by name
SKILL_REGISTRY = {
    "ApproachObject": ApproachSkill,
    "GraspObject": GraspSkill,
    "MoveObjectToRegion": MoveSkill,
    "PlaceObject": PlaceSkill,
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
