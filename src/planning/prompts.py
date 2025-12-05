"""Prompt templates for Qwen skill planning.

Prompts are designed to make Qwen output valid JSON skill sequences.
"""

from typing import Dict, List, Any
from .skill_schema import get_skill_schema_for_prompt
from ..grounding.enriched_object import parse_object_description, enrich_objects


SYSTEM_PROMPT = """You are a robot task planner. Given a task description and the current world state, output a sequence of skills to accomplish the task.

You must output ONLY a valid JSON array of skill calls. No explanations, no markdown, just the JSON array.

{skill_schema}

CRITICAL RULES:
1. Output ONLY a JSON array, nothing else
2. Use exact skill names from the list above
3. Use exact object IDs from the world state
4. Follow preconditions - you cannot grasp while holding, cannot move without holding
5. A typical pick-and-place sequence is: ApproachObject → GraspObject → MoveObjectToRegion → PlaceObject
6. Maximum 10 skills per plan"""


USER_PROMPT_TEMPLATE = """Task: {task_description}

World State:
{world_state_json}

Output the skill sequence as a JSON array:"""


FEW_SHOT_EXAMPLE = """Example:
Task: "pick up the black bowl and place it on the plate"
World State:
{
  "objects": [
    {"id": "akita_black_bowl_1_main", "description": "black bowl", "position": "(-0.04, 0.19, 0.97)"},
    {"id": "plate_1_main", "description": "plate", "position": "(0.07, 0.20, 0.97)"}
  ],
  "holding": null,
  "on": {"akita_black_bowl_1_main": "table"},
  "inside": {}
}

Output:
[
  {"skill": "ApproachObject", "args": {"obj": "akita_black_bowl_1_main"}},
  {"skill": "GraspObject", "args": {"obj": "akita_black_bowl_1_main"}},
  {"skill": "MoveObjectToRegion", "args": {"obj": "akita_black_bowl_1_main", "region": "plate_1_main"}},
  {"skill": "PlaceObject", "args": {"obj": "akita_black_bowl_1_main", "region": "plate_1_main"}}
]"""


def build_system_prompt(include_schema: bool = True) -> str:
    """Build the system prompt for Qwen."""
    if include_schema:
        schema = get_skill_schema_for_prompt()
        return SYSTEM_PROMPT.format(skill_schema=schema)
    return SYSTEM_PROMPT.format(skill_schema="")


def build_user_prompt(
    task_description: str,
    world_state_dict: Dict[str, Any],
    include_few_shot: bool = True,
) -> str:
    """Build the user prompt with task and world state.

    Args:
        task_description: Natural language task
        world_state_dict: World state as dictionary (with object descriptions)
        include_few_shot: Whether to include few-shot example

    Returns:
        Formatted user prompt
    """
    import json

    # Format world state nicely
    world_state_json = json.dumps(world_state_dict, indent=2)

    prompt = USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        world_state_json=world_state_json,
    )

    if include_few_shot:
        prompt = FEW_SHOT_EXAMPLE + "\n\nNow solve this task:\n" + prompt

    return prompt


def prepare_world_state_for_qwen(world_state) -> Dict[str, Any]:
    """Convert WorldState to Qwen-friendly format with descriptions.

    This is the key projection layer - converts internal object IDs to
    human-readable descriptions that Qwen can understand.

    Args:
        world_state: WorldState object

    Returns:
        Dictionary suitable for Qwen prompt
    """
    objects = []

    for obj_id, obj_state in world_state.objects.items():
        # Parse object name for description
        desc = parse_object_description(obj_id)
        pos = obj_state.pose[:3] if obj_state.pose is not None else None

        obj_entry = {
            "id": obj_id,
            "description": desc,
        }
        if pos is not None:
            obj_entry["position"] = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

        objects.append(obj_entry)

    return {
        "objects": objects,
        "holding": world_state.holding,
        "on": dict(world_state.on) if world_state.on else {},
        "inside": dict(world_state.inside) if world_state.inside else {},
    }


# parse_object_description is now imported from grounding module
