"""Offline planner tests.

Tests plan validation and parsing without needing Qwen model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.planning.plan_validator import (
    parse_qwen_output,
    validate_plan,
    validate_plan_semantics,
)
from src.planning.prompts import (
    prepare_world_state_for_qwen,
    parse_object_description,
    build_system_prompt,
    build_user_prompt,
)
from src.world_model.state import WorldState, ObjectState
import numpy as np


def test_parse_qwen_output_direct_json():
    """Test parsing direct JSON array."""
    print("\n" + "=" * 60)
    print("TEST: Parse direct JSON array")
    print("=" * 60)

    raw = '[{"skill": "ApproachObject", "args": {"obj": "bowl_1"}}]'
    plan, error = parse_qwen_output(raw)

    assert error is None, f"Unexpected error: {error}"
    assert len(plan) == 1
    assert plan[0]["skill"] == "ApproachObject"
    print("✓ Direct JSON parsing works")
    return True


def test_parse_qwen_output_markdown():
    """Test parsing JSON in markdown code block."""
    print("\n" + "=" * 60)
    print("TEST: Parse JSON in markdown")
    print("=" * 60)

    raw = '''Here is the plan:
```json
[
  {"skill": "ApproachObject", "args": {"obj": "bowl_1"}},
  {"skill": "GraspObject", "args": {"obj": "bowl_1"}}
]
```
'''
    plan, error = parse_qwen_output(raw)

    assert error is None, f"Unexpected error: {error}"
    assert len(plan) == 2
    assert plan[0]["skill"] == "ApproachObject"
    assert plan[1]["skill"] == "GraspObject"
    print("✓ Markdown JSON parsing works")
    return True


def test_parse_qwen_output_wrapped():
    """Test parsing wrapped format {"plan": [...]}."""
    print("\n" + "=" * 60)
    print("TEST: Parse wrapped JSON")
    print("=" * 60)

    raw = '{"plan": [{"skill": "ApproachObject", "args": {"obj": "bowl_1"}}]}'
    plan, error = parse_qwen_output(raw)

    assert error is None, f"Unexpected error: {error}"
    assert len(plan) == 1
    print("✓ Wrapped JSON parsing works")
    return True


def test_validate_plan_valid():
    """Test validation of valid plan."""
    print("\n" + "=" * 60)
    print("TEST: Validate valid plan")
    print("=" * 60)

    plan = [
        {"skill": "ApproachObject", "args": {"obj": "bowl_1"}},
        {"skill": "GraspObject", "args": {"obj": "bowl_1"}},
        {"skill": "MoveObjectToRegion", "args": {"obj": "bowl_1", "region": "plate_1"}},
        {"skill": "PlaceObject", "args": {"obj": "bowl_1", "region": "plate_1"}},
    ]

    valid, error = validate_plan(plan)
    assert valid, f"Plan should be valid: {error}"
    print("✓ Valid plan accepted")
    return True


def test_validate_plan_unknown_skill():
    """Test validation catches unknown skill."""
    print("\n" + "=" * 60)
    print("TEST: Validate catches unknown skill")
    print("=" * 60)

    plan = [{"skill": "UnknownSkill", "args": {"obj": "bowl_1"}}]

    valid, error = validate_plan(plan)
    assert not valid, "Should reject unknown skill"
    assert "Unknown skill" in error
    print(f"✓ Correctly rejected: {error}")
    return True


def test_validate_plan_missing_arg():
    """Test validation catches missing argument."""
    print("\n" + "=" * 60)
    print("TEST: Validate catches missing argument")
    print("=" * 60)

    plan = [{"skill": "ApproachObject", "args": {}}]  # Missing 'obj'

    valid, error = validate_plan(plan)
    assert not valid, "Should reject missing arg"
    assert "missing required arg" in error.lower()
    print(f"✓ Correctly rejected: {error}")
    return True


def test_validate_plan_loop_detection():
    """Test validation catches loops."""
    print("\n" + "=" * 60)
    print("TEST: Validate detects loops")
    print("=" * 60)

    plan = [
        {"skill": "ApproachObject", "args": {"obj": "bowl_1"}},
        {"skill": "ApproachObject", "args": {"obj": "bowl_1"}},
        {"skill": "ApproachObject", "args": {"obj": "bowl_1"}},
    ]

    valid, error = validate_plan(plan)
    assert not valid, "Should detect loop"
    assert "loop" in error.lower()
    print(f"✓ Correctly detected loop: {error}")
    return True


def test_validate_plan_too_long():
    """Test validation rejects too-long plans."""
    print("\n" + "=" * 60)
    print("TEST: Validate rejects long plans")
    print("=" * 60)

    plan = [{"skill": "ApproachObject", "args": {"obj": f"bowl_{i}"}} for i in range(15)]

    valid, error = validate_plan(plan, max_plan_length=10)
    assert not valid, "Should reject long plan"
    assert "too long" in error.lower()
    print(f"✓ Correctly rejected: {error}")
    return True


def test_validate_semantics_grasp_while_holding():
    """Test semantic validation catches grasp while holding."""
    print("\n" + "=" * 60)
    print("TEST: Semantic validation - grasp while holding")
    print("=" * 60)

    plan = [
        {"skill": "GraspObject", "args": {"obj": "bowl_1"}},
        {"skill": "GraspObject", "args": {"obj": "bowl_2"}},  # Can't grasp while holding
    ]

    world_state = {
        "objects": [
            {"id": "bowl_1", "description": "bowl"},
            {"id": "bowl_2", "description": "bowl"},
        ],
        "holding": None,
    }

    valid, error = validate_plan_semantics(plan, world_state)
    assert not valid, "Should reject grasp while holding"
    assert "holding" in error.lower()
    print(f"✓ Correctly rejected: {error}")
    return True


def test_validate_semantics_move_without_holding():
    """Test semantic validation catches move without holding."""
    print("\n" + "=" * 60)
    print("TEST: Semantic validation - move without holding")
    print("=" * 60)

    plan = [
        {"skill": "MoveObjectToRegion", "args": {"obj": "bowl_1", "region": "plate_1"}},
    ]

    world_state = {
        "objects": [
            {"id": "bowl_1", "description": "bowl"},
            {"id": "plate_1", "description": "plate"},
        ],
        "holding": None,
    }

    valid, error = validate_plan_semantics(plan, world_state)
    assert not valid, "Should reject move without holding"
    assert "not holding" in error.lower()
    print(f"✓ Correctly rejected: {error}")
    return True


def test_parse_object_description():
    """Test object ID to description parsing."""
    print("\n" + "=" * 60)
    print("TEST: Parse object descriptions")
    print("=" * 60)

    test_cases = [
        ("akita_black_bowl_1_main", "black bowl"),
        ("plate_1_main", "plate"),
        ("glazed_rim_porcelain_ramekin_1_main", "porcelain ramekin"),
        ("wooden_cabinet_1_base", "wooden cabinet"),
    ]

    all_passed = True
    for obj_id, expected in test_cases:
        result = parse_object_description(obj_id)
        if result == expected:
            print(f"✓ {obj_id} → {result}")
        else:
            print(f"✗ {obj_id} → {result} (expected: {expected})")
            all_passed = False

    return all_passed


def test_prepare_world_state():
    """Test world state preparation for Qwen."""
    print("\n" + "=" * 60)
    print("TEST: Prepare world state for Qwen")
    print("=" * 60)

    import time
    world_state = WorldState()
    world_state.objects["akita_black_bowl_1_main"] = ObjectState(
        name="akita_black_bowl_1_main",
        pose=np.array([0.1, 0.2, 0.9, 0, 0, 0, 1]),
        last_seen=time.time(),
    )
    world_state.objects["plate_1_main"] = ObjectState(
        name="plate_1_main",
        pose=np.array([0.3, 0.2, 0.9, 0, 0, 0, 1]),
        last_seen=time.time(),
    )
    world_state.holding = None
    world_state.on = {"akita_black_bowl_1_main": "table"}

    result = prepare_world_state_for_qwen(world_state)

    assert "objects" in result
    assert len(result["objects"]) == 2
    assert result["holding"] is None

    # Check descriptions are included
    obj_ids = [o["id"] for o in result["objects"]]
    assert "akita_black_bowl_1_main" in obj_ids

    bowl_obj = next(o for o in result["objects"] if "bowl" in o["id"])
    assert "description" in bowl_obj
    assert "bowl" in bowl_obj["description"].lower()

    print(f"✓ World state prepared:")
    import json
    print(json.dumps(result, indent=2))
    return True


def test_build_prompts():
    """Test prompt building."""
    print("\n" + "=" * 60)
    print("TEST: Build prompts")
    print("=" * 60)

    system_prompt = build_system_prompt(include_schema=True)
    assert "ApproachObject" in system_prompt
    assert "GraspObject" in system_prompt
    assert "preconditions" in system_prompt.lower()
    print(f"✓ System prompt built ({len(system_prompt)} chars)")

    world_state = {
        "objects": [{"id": "bowl_1", "description": "black bowl"}],
        "holding": None,
    }

    user_prompt = build_user_prompt(
        task_description="pick up the bowl",
        world_state_dict=world_state,
        include_few_shot=True,
    )

    assert "pick up the bowl" in user_prompt
    assert "bowl_1" in user_prompt
    print(f"✓ User prompt built ({len(user_prompt)} chars)")

    return True


if __name__ == "__main__":
    results = []

    # Parsing tests
    results.append(("Parse direct JSON", test_parse_qwen_output_direct_json()))
    results.append(("Parse markdown JSON", test_parse_qwen_output_markdown()))
    results.append(("Parse wrapped JSON", test_parse_qwen_output_wrapped()))

    # Validation tests
    results.append(("Validate valid plan", test_validate_plan_valid()))
    results.append(("Validate unknown skill", test_validate_plan_unknown_skill()))
    results.append(("Validate missing arg", test_validate_plan_missing_arg()))
    results.append(("Validate loop detection", test_validate_plan_loop_detection()))
    results.append(("Validate plan length", test_validate_plan_too_long()))

    # Semantic validation tests
    results.append(("Semantic: grasp while holding", test_validate_semantics_grasp_while_holding()))
    results.append(("Semantic: move without holding", test_validate_semantics_move_without_holding()))

    # Prompt preparation tests
    results.append(("Parse object descriptions", test_parse_object_description()))
    results.append(("Prepare world state", test_prepare_world_state()))
    results.append(("Build prompts", test_build_prompts()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")
