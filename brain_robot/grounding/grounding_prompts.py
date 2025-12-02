"""Prompt templates for Qwen semantic grounding."""

import json
from typing import List, Dict, Any

from .enriched_object import EnrichedObject


GROUNDING_SYSTEM_PROMPT = """You identify objects for robot manipulation tasks.

Given a task description and a list of detected objects, determine:
1. source_object: The object to pick up or manipulate
2. target_location: Where to place it (another object ID or "table")

Rules:
- Use EXACT object IDs from the provided list
- Pay attention to spatial context (e.g., "bowl on the cookie box" means the bowl that is on the cookie box)
- If multiple objects match, prefer the one with matching spatial context
- Output ONLY valid JSON, no explanations"""


GROUNDING_USER_TEMPLATE = """Task: {task_description}

Detected Objects:
{objects_json}

Which object should the robot manipulate, and where should it place it?

Output format:
{{
  "source_object": "<exact object ID from list>",
  "target_location": "<exact object ID from list or 'table'>",
  "confidence": "high" | "medium" | "low"
}}"""


GROUNDING_FEW_SHOT = """Example 1:
Task: "pick up the black bowl and place it on the plate"
Objects:
[
  {"id": "akita_black_bowl_1_main", "description": "black bowl", "spatial": "on table"},
  {"id": "plate_1_main", "description": "plate", "spatial": "on table"}
]
Output:
{"source_object": "akita_black_bowl_1_main", "target_location": "plate_1_main", "confidence": "high"}

Example 2:
Task: "pick up the black bowl on the cookie box and place it on the plate"
Objects:
[
  {"id": "akita_black_bowl_1_main", "description": "black bowl", "spatial": "on cookie box"},
  {"id": "akita_black_bowl_2_main", "description": "black bowl", "spatial": "on wooden cabinet"},
  {"id": "plate_1_main", "description": "plate", "spatial": "on table"},
  {"id": "cookies_1_main", "description": "cookie box", "spatial": "on table"}
]
Output:
{"source_object": "akita_black_bowl_1_main", "target_location": "plate_1_main", "confidence": "high"}

Example 3:
Task: "put the mug in the drawer"
Objects:
[
  {"id": "mug_1_main", "description": "mug", "spatial": "on table"},
  {"id": "wooden_drawer_1_main", "description": "wooden drawer", "spatial": "on cabinet"}
]
Output:
{"source_object": "mug_1_main", "target_location": "wooden_drawer_1_main", "confidence": "high"}
"""


# Valid region names (non-object targets)
VALID_REGIONS = {
    "table": {"type": "surface", "description": "main table surface"},
}


def is_valid_region(target: str) -> bool:
    """Check if target is a valid non-object region."""
    return target.lower() in VALID_REGIONS


def build_grounding_prompt(
    task_description: str,
    objects: List[EnrichedObject],
    include_few_shot: bool = True,
) -> str:
    """Build the full grounding prompt for Qwen.

    Args:
        task_description: Natural language task
        objects: List of enriched objects in scene
        include_few_shot: Whether to include few-shot examples

    Returns:
        Complete prompt string
    """
    # Convert objects to JSON-friendly format
    objects_list = [obj.to_dict() for obj in objects]
    objects_json = json.dumps(objects_list, indent=2)

    user_prompt = GROUNDING_USER_TEMPLATE.format(
        task_description=task_description,
        objects_json=objects_json,
    )

    if include_few_shot:
        full_prompt = (
            GROUNDING_SYSTEM_PROMPT + "\n\n" +
            GROUNDING_FEW_SHOT + "\n\n" +
            "Now solve this task:\n" + user_prompt
        )
    else:
        full_prompt = GROUNDING_SYSTEM_PROMPT + "\n\n" + user_prompt

    return full_prompt


def parse_grounding_output(raw_output: str) -> tuple:
    """Parse Qwen's grounding output.

    Returns:
        (result_dict, error_message) tuple
    """
    import re

    # Try direct JSON parse
    try:
        result = json.loads(raw_output.strip())
        return result, None
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
    if match:
        try:
            result = json.loads(match.group(1))
            return result, None
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    match = re.search(r'\{[\s\S]*\}', raw_output)
    if match:
        try:
            result = json.loads(match.group(0))
            return result, None
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse JSON from: {raw_output[:200]}..."


def validate_grounding(
    result: Dict[str, Any],
    valid_object_ids: set,
) -> tuple:
    """Validate grounding result.

    Args:
        result: Parsed grounding result
        valid_object_ids: Set of valid object IDs in scene

    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(result, dict):
        return False, "Result must be a dictionary"

    source = result.get("source_object")
    target = result.get("target_location")

    if not source:
        return False, "Missing source_object"
    if not target:
        return False, "Missing target_location"

    # Check source exists
    if source not in valid_object_ids:
        return False, f"Unknown source object: '{source}'"

    # Check target exists (object or valid region)
    if target not in valid_object_ids and not is_valid_region(target):
        return False, f"Unknown target: '{target}'"

    return True, "OK"
