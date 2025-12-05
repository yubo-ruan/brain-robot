"""Skill schema definitions for Qwen planning.

Defines available skills with their arguments, preconditions, and postconditions.
This schema is included in the Qwen prompt to guide plan generation.
"""

from typing import Dict, Any, Optional
from ..skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill
from ..config import SkillConfig


# Skill schema for Qwen prompt
SKILL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "ApproachObject": {
        "description": "Move gripper to pre-grasp pose above the specified object",
        "args": {
            "obj": "object_id - The ID of the object to approach"
        },
        "preconditions": [
            "gripper is not holding anything (holding == null)",
            "object exists in the scene"
        ],
        "postconditions": [
            "gripper is positioned above the object at pre-grasp height"
        ],
        "example": {"skill": "ApproachObject", "args": {"obj": "akita_black_bowl_1_main"}}
    },
    "GraspObject": {
        "description": "Lower gripper, close on object, and lift slightly",
        "args": {
            "obj": "object_id - The ID of the object to grasp"
        },
        "preconditions": [
            "gripper is not holding anything (holding == null)",
            "gripper is positioned above the object (after ApproachObject)"
        ],
        "postconditions": [
            "gripper is holding the object (holding == obj)"
        ],
        "example": {"skill": "GraspObject", "args": {"obj": "akita_black_bowl_1_main"}}
    },
    "MoveObjectToRegion": {
        "description": "Move the held object to above the target region/object",
        "args": {
            "obj": "object_id - The ID of the object being moved (must be held)",
            "region": "target_id - The ID of the target location (object or region)"
        },
        "preconditions": [
            "gripper is holding the specified object (holding == obj)"
        ],
        "postconditions": [
            "held object is positioned above the target region"
        ],
        "example": {"skill": "MoveObjectToRegion", "args": {"obj": "akita_black_bowl_1_main", "region": "plate_1_main"}}
    },
    "PlaceObject": {
        "description": "Lower the held object and release it at the target location",
        "args": {
            "obj": "object_id - The ID of the object to place",
            "region": "target_id - The ID of the target location"
        },
        "preconditions": [
            "gripper is holding the specified object (holding == obj)",
            "gripper is above the target region (after MoveObjectToRegion)"
        ],
        "postconditions": [
            "gripper is not holding anything (holding == null)",
            "object is at the target location"
        ],
        "example": {"skill": "PlaceObject", "args": {"obj": "akita_black_bowl_1_main", "region": "plate_1_main"}}
    }
}


def get_skill_schema_for_prompt() -> str:
    """Format skill schema as string for inclusion in Qwen prompt."""
    lines = []
    for skill_name, schema in SKILL_SCHEMA.items():
        lines.append(f"## {skill_name}")
        lines.append(f"Description: {schema['description']}")
        lines.append("Arguments:")
        for arg_name, arg_desc in schema['args'].items():
            lines.append(f"  - {arg_name}: {arg_desc}")
        lines.append("Preconditions:")
        for pre in schema['preconditions']:
            lines.append(f"  - {pre}")
        lines.append("Postconditions:")
        for post in schema['postconditions']:
            lines.append(f"  - {post}")
        lines.append(f"Example: {schema['example']}")
        lines.append("")
    return "\n".join(lines)


def get_skill_by_name(skill_name: str, config: Optional[SkillConfig] = None):
    """Get skill instance by name.

    Args:
        skill_name: Name of the skill (e.g., "ApproachObject")
        config: Optional skill configuration

    Returns:
        Skill instance or None if not found
    """
    config = config or SkillConfig()

    skill_map = {
        "ApproachObject": ApproachSkill,
        "GraspObject": GraspSkill,
        "MoveObjectToRegion": MoveSkill,
        "PlaceObject": PlaceSkill,
    }

    skill_class = skill_map.get(skill_name)
    if skill_class is None:
        return None

    return skill_class(config=config)


def get_required_args(skill_name: str) -> list:
    """Get list of required argument names for a skill."""
    schema = SKILL_SCHEMA.get(skill_name)
    if schema is None:
        return []
    return list(schema['args'].keys())


def skill_exists(skill_name: str) -> bool:
    """Check if a skill name is valid."""
    return skill_name in SKILL_SCHEMA
