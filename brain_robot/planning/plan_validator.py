"""Plan validation and parsing utilities.

Validates Qwen output and catches common errors before execution.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional

from .skill_schema import SKILL_SCHEMA, get_required_args, skill_exists


def parse_qwen_output(raw_output: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Parse Qwen output, attempting to extract JSON plan.

    Handles common formatting issues:
    - Direct JSON array
    - JSON in markdown code blocks
    - JSON embedded in text

    Args:
        raw_output: Raw string output from Qwen

    Returns:
        (plan, error) tuple. If successful, plan is a list of skill dicts.
        If failed, error contains the error message.
    """
    if not raw_output or not raw_output.strip():
        return None, "Empty output from Qwen"

    raw_output = raw_output.strip()

    # Try 1: Direct JSON parse
    try:
        plan = json.loads(raw_output)
        if isinstance(plan, list):
            return plan, None
        elif isinstance(plan, dict) and "plan" in plan:
            # Handle wrapped format {"plan": [...]}
            return plan["plan"], None
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code block
    code_block_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, raw_output)
        if match:
            try:
                plan = json.loads(match.group(1))
                if isinstance(plan, list):
                    return plan, None
                elif isinstance(plan, dict) and "plan" in plan:
                    return plan["plan"], None
            except json.JSONDecodeError:
                pass

    # Try 3: Find JSON array in text
    array_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw_output)
    if array_match:
        try:
            plan = json.loads(array_match.group(0))
            if isinstance(plan, list):
                return plan, None
        except json.JSONDecodeError:
            pass

    # Try 4: Find single JSON object (single-step plan)
    obj_match = re.search(r'\{[^{}]*"skill"[^{}]*\}', raw_output)
    if obj_match:
        try:
            step = json.loads(obj_match.group(0))
            if isinstance(step, dict) and "skill" in step:
                return [step], None
        except json.JSONDecodeError:
            pass

    return None, f"Could not parse JSON from output: {raw_output[:200]}..."


def validate_plan(
    plan: List[Dict[str, Any]],
    max_plan_length: int = 10,
) -> Tuple[bool, str]:
    """Validate a parsed plan before execution.

    Checks:
    - Plan is a list
    - Plan length is reasonable
    - Each step has valid skill name
    - Each step has required arguments
    - No obvious loops (same skill+args 3+ times consecutively)

    Args:
        plan: List of skill call dictionaries
        max_plan_length: Maximum allowed plan length

    Returns:
        (valid, message) tuple
    """
    # Check type
    if not isinstance(plan, list):
        return False, f"Plan must be a list, got {type(plan).__name__}"

    # Check empty
    if len(plan) == 0:
        return False, "Plan is empty"

    # Check length
    if len(plan) > max_plan_length:
        return False, f"Plan too long: {len(plan)} steps (max {max_plan_length})"

    # Track for loop detection
    seen_steps = []

    for i, step in enumerate(plan):
        # Check step is dict
        if not isinstance(step, dict):
            return False, f"Step {i}: Expected dict, got {type(step).__name__}"

        # Check skill field exists
        skill_name = step.get("skill")
        if skill_name is None:
            return False, f"Step {i}: Missing 'skill' field"

        # Check skill exists
        if not skill_exists(skill_name):
            valid_skills = list(SKILL_SCHEMA.keys())
            return False, f"Step {i}: Unknown skill '{skill_name}'. Valid skills: {valid_skills}"

        # Check args field exists
        args = step.get("args", {})
        if not isinstance(args, dict):
            return False, f"Step {i}: 'args' must be a dict, got {type(args).__name__}"

        # Check required args
        required_args = get_required_args(skill_name)
        for arg_name in required_args:
            if arg_name not in args:
                return False, f"Step {i}: Skill '{skill_name}' missing required arg '{arg_name}'"
            if args[arg_name] is None or args[arg_name] == "":
                return False, f"Step {i}: Arg '{arg_name}' cannot be empty"

        # Loop detection: same skill + args 3 times in a row
        step_signature = (skill_name, tuple(sorted(args.items())))
        seen_steps.append(step_signature)

        if len(seen_steps) >= 3:
            last_three = seen_steps[-3:]
            if last_three[0] == last_three[1] == last_three[2]:
                return False, f"Step {i}: Detected loop - '{skill_name}' repeated 3 times consecutively"

    return True, "Plan valid"


def validate_plan_semantics(
    plan: List[Dict[str, Any]],
    world_state_dict: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate plan semantics against world state.

    Additional checks:
    - Objects referenced in plan exist in world state
    - Basic precondition sanity (e.g., can't grasp if already holding)

    Args:
        plan: Validated plan (passed validate_plan)
        world_state_dict: World state as dictionary

    Returns:
        (valid, message) tuple
    """
    objects = world_state_dict.get("objects", [])
    object_ids = set()

    # Extract object IDs from world state
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict) and "id" in obj:
                object_ids.add(obj["id"])
    elif isinstance(objects, dict):
        object_ids = set(objects.keys())

    # Track simulated state
    holding = world_state_dict.get("holding")

    for i, step in enumerate(plan):
        skill_name = step["skill"]
        args = step.get("args", {})

        # Check object references exist
        obj_arg = args.get("obj")
        if obj_arg and obj_arg not in object_ids:
            # Allow if it looks like a valid object ID pattern
            if not any(keyword in obj_arg.lower() for keyword in ["bowl", "plate", "mug", "ramekin", "drawer", "cabinet"]):
                return False, f"Step {i}: Object '{obj_arg}' not found in world state"

        region_arg = args.get("region")
        if region_arg and region_arg not in object_ids:
            if not any(keyword in region_arg.lower() for keyword in ["bowl", "plate", "mug", "ramekin", "drawer", "cabinet", "table"]):
                return False, f"Step {i}: Region '{region_arg}' not found in world state"

        # Simulate state transitions for basic sanity
        if skill_name == "ApproachObject":
            if holding is not None:
                return False, f"Step {i}: Cannot ApproachObject while holding '{holding}'"

        elif skill_name == "GraspObject":
            if holding is not None:
                return False, f"Step {i}: Cannot GraspObject while already holding '{holding}'"
            holding = obj_arg  # Simulate grasp

        elif skill_name == "MoveObjectToRegion":
            if holding is None:
                return False, f"Step {i}: Cannot MoveObjectToRegion - not holding anything"
            if holding != obj_arg:
                return False, f"Step {i}: Cannot move '{obj_arg}' - currently holding '{holding}'"

        elif skill_name == "PlaceObject":
            if holding is None:
                return False, f"Step {i}: Cannot PlaceObject - not holding anything"
            if holding != obj_arg:
                return False, f"Step {i}: Cannot place '{obj_arg}' - currently holding '{holding}'"
            holding = None  # Simulate release

    return True, "Plan semantics valid"
