"""Offline tests for semantic grounding.

Tests grounding without running the simulator.
Run with: python -m src.tests.test_grounding_offline
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.grounding.enriched_object import (
    EnrichedObject,
    parse_object_description,
    get_spatial_descriptions,
)
from src.grounding.grounding_prompts import (
    parse_grounding_output,
    validate_grounding,
    build_grounding_prompt,
)
from src.grounding.semantic_grounder import QwenSemanticGrounder


# ============================================================
# Test Cases
# ============================================================

# Standard LIBERO tasks
LIBERO_TEST_CASES = [
    {
        "name": "task_0_simple",
        "task": "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
            EnrichedObject(id="glazed_rim_porcelain_ramekin_1_main", description="porcelain ramekin", type="ramekin", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "task_1_next_to",
        "task": "pick up the black bowl next to the ramekin and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
            EnrichedObject(id="glazed_rim_porcelain_ramekin_1_main", description="porcelain ramekin", type="ramekin", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "task_2_table_center",
        "task": "pick up the black bowl from table center and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "task_3_on_cookie_box",
        "task": "pick up the black bowl on the cookie box and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on cookie box"),
            EnrichedObject(id="akita_black_bowl_2_main", description="black bowl", type="bowl", color="black", spatial_text="on wooden cabinet"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
            EnrichedObject(id="cookies_1_main", description="cookie box", type="cookies", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",  # The one ON the cookie box
        "expected_target": "plate_1_main",
    },
    {
        "name": "task_4_in_drawer",
        "task": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="inside drawer"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
            EnrichedObject(id="wooden_cabinet_1_top", description="wooden cabinet", type="cabinet", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "task_5_on_ramekin",
        "task": "pick up the black bowl on the ramekin and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on ramekin"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
            EnrichedObject(id="glazed_rim_porcelain_ramekin_1_main", description="porcelain ramekin", type="ramekin", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
]

# Paraphrase tests
PARAPHRASE_TEST_CASES = [
    {
        "name": "paraphrase_move_onto",
        "task": "move the black bowl onto the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "paraphrase_put_on",
        "task": "put the black bowl on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
    {
        "name": "paraphrase_grab_place",
        "task": "grab the black bowl and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",
        "expected_target": "plate_1_main",
    },
]

# Edge case tests (should fail gracefully or be marked ambiguous)
EDGE_CASE_TESTS = [
    {
        "name": "edge_no_matching_object",
        "task": "pick up the red mug and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": None,  # Should fail validation or pick wrong object
        "expected_target": "plate_1_main",
        "expect_failure": True,
    },
    {
        "name": "edge_multiple_identical_no_spatial",
        "task": "pick up the black bowl and place it on the plate",
        "objects": [
            EnrichedObject(id="akita_black_bowl_1_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="akita_black_bowl_2_main", description="black bowl", type="bowl", color="black", spatial_text="on table"),
            EnrichedObject(id="plate_1_main", description="plate", type="plate", spatial_text="on table"),
        ],
        "expected_source": "akita_black_bowl_1_main",  # Either is acceptable
        "expected_target": "plate_1_main",
        "expect_ambiguous": True,
    },
]


# ============================================================
# Unit Tests (no model required)
# ============================================================

def test_parse_object_description():
    """Test object description parsing."""
    print("\n" + "=" * 60)
    print("TEST: parse_object_description")
    print("=" * 60)

    test_cases = [
        ("akita_black_bowl_1_main", "black bowl"),
        ("plate_1_main", "plate"),
        ("cookies_1_main", "cookie box"),  # LIBERO name mapping
        ("glazed_rim_porcelain_ramekin_1_main", "porcelain ramekin"),
        ("wooden_cabinet_1_base", "wooden cabinet"),
        ("flat_stove_1_burner", "stove burner"),
    ]

    passed = 0
    for obj_id, expected in test_cases:
        result = parse_object_description(obj_id)
        if result == expected:
            print(f"  PASS: {obj_id} → {result}")
            passed += 1
        else:
            print(f"  FAIL: {obj_id} → {result} (expected: {expected})")

    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_spatial_descriptions():
    """Test spatial description generation."""
    print("\n" + "=" * 60)
    print("TEST: get_spatial_descriptions")
    print("=" * 60)

    # Test with "on" relations
    on = {"akita_black_bowl_1_main": "cookies_1_main"}
    inside = {}

    spatial_id, spatial_text = get_spatial_descriptions(
        "akita_black_bowl_1_main", on, inside
    )

    print(f"  Object on cookie box:")
    print(f"    spatial_id: {spatial_id}")
    print(f"    spatial_text: {spatial_text}")

    assert spatial_id == "on cookies_1_main", f"Expected 'on cookies_1_main', got '{spatial_id}'"
    assert spatial_text == "on cookie box", f"Expected 'on cookie box', got '{spatial_text}'"

    # Test with "inside" relations
    inside = {"mug_1_main": "wooden_drawer_1"}
    on = {}

    spatial_id, spatial_text = get_spatial_descriptions(
        "mug_1_main", on, inside
    )

    print(f"  Object inside drawer:")
    print(f"    spatial_id: {spatial_id}")
    print(f"    spatial_text: {spatial_text}")

    assert spatial_id == "inside wooden_drawer_1"
    assert "drawer" in spatial_text.lower()

    print("\n  PASS: All spatial description tests")
    return True


def test_ambiguity_detection():
    """Test heuristic ambiguity detection."""
    print("\n" + "=" * 60)
    print("TEST: Ambiguity detection")
    print("=" * 60)

    from src.grounding.semantic_grounder import QwenSemanticGrounder

    grounder = QwenSemanticGrounder()

    # Case 1: Disambiguated by spatial context
    task1 = "pick up the black bowl on the cookie box"
    objects1 = [
        EnrichedObject(id="bowl_1", description="black bowl", type="bowl", color="black", spatial_text="on cookie box"),
        EnrichedObject(id="bowl_2", description="black bowl", type="bowl", color="black", spatial_text="on cabinet"),
    ]

    ambig1, alts1 = grounder._detect_ambiguity(task1, objects1, "bowl_1")
    print(f"  Task: '{task1}'")
    print(f"    ambiguous: {ambig1}, alternatives: {alts1}")
    assert not ambig1, "Should NOT be ambiguous - spatial context matches"

    # Case 2: Ambiguous - no spatial context in task
    task2 = "pick up the black bowl"
    objects2 = [
        EnrichedObject(id="bowl_1", description="black bowl", type="bowl", color="black", spatial_text="on table"),
        EnrichedObject(id="bowl_2", description="black bowl", type="bowl", color="black", spatial_text="on table"),
    ]

    ambig2, alts2 = grounder._detect_ambiguity(task2, objects2, "bowl_1")
    print(f"  Task: '{task2}'")
    print(f"    ambiguous: {ambig2}, alternatives: {alts2}")
    assert ambig2, "Should be ambiguous - no spatial context"
    assert "bowl_2" in alts2

    # Case 3: Not ambiguous - different types
    task3 = "pick up the black bowl"
    objects3 = [
        EnrichedObject(id="bowl_1", description="black bowl", type="bowl", color="black", spatial_text="on table"),
        EnrichedObject(id="plate_1", description="plate", type="plate", spatial_text="on table"),
    ]

    ambig3, alts3 = grounder._detect_ambiguity(task3, objects3, "bowl_1")
    print(f"  Task: '{task3}'")
    print(f"    ambiguous: {ambig3}, alternatives: {alts3}")
    assert not ambig3, "Should NOT be ambiguous - different types"

    print("\n  PASS: All ambiguity detection tests")
    return True


def test_grounding_output_parsing():
    """Test JSON output parsing."""
    print("\n" + "=" * 60)
    print("TEST: Grounding output parsing")
    print("=" * 60)

    test_cases = [
        # Direct JSON
        (
            '{"source_object": "bowl_1", "target_location": "plate_1", "confidence": "high"}',
            {"source_object": "bowl_1", "target_location": "plate_1", "confidence": "high"},
        ),
        # With markdown
        (
            '```json\n{"source_object": "bowl_1", "target_location": "plate_1"}\n```',
            {"source_object": "bowl_1", "target_location": "plate_1"},
        ),
        # With explanation before
        (
            'The source is the bowl.\n{"source_object": "bowl_1", "target_location": "plate_1"}',
            {"source_object": "bowl_1", "target_location": "plate_1"},
        ),
    ]

    passed = 0
    for raw, expected in test_cases:
        result, error = parse_grounding_output(raw)
        if result == expected:
            print(f"  PASS: Parsed correctly")
            passed += 1
        else:
            print(f"  FAIL: Got {result}, expected {expected}")
            if error:
                print(f"    Error: {error}")

    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_grounding_validation():
    """Test grounding validation."""
    print("\n" + "=" * 60)
    print("TEST: Grounding validation")
    print("=" * 60)

    valid_ids = {"bowl_1", "plate_1", "cabinet_1"}

    # Valid grounding
    result1 = {"source_object": "bowl_1", "target_location": "plate_1"}
    valid1, msg1 = validate_grounding(result1, valid_ids)
    print(f"  Valid grounding: valid={valid1}, msg={msg1}")
    assert valid1

    # Invalid source
    result2 = {"source_object": "fake_obj", "target_location": "plate_1"}
    valid2, msg2 = validate_grounding(result2, valid_ids)
    print(f"  Invalid source: valid={valid2}, msg={msg2}")
    assert not valid2

    # Invalid target
    result3 = {"source_object": "bowl_1", "target_location": "fake_target"}
    valid3, msg3 = validate_grounding(result3, valid_ids)
    print(f"  Invalid target: valid={valid3}, msg={msg3}")
    assert not valid3

    # Valid region target
    result4 = {"source_object": "bowl_1", "target_location": "table"}
    valid4, msg4 = validate_grounding(result4, valid_ids)
    print(f"  Table target: valid={valid4}, msg={msg4}")
    assert valid4

    print("\n  PASS: All validation tests")
    return True


# ============================================================
# Integration Tests (requires Qwen model)
# ============================================================

def test_grounding_with_model():
    """Test full grounding with Qwen model."""
    print("\n" + "=" * 60)
    print("TEST: Full grounding with Qwen (requires model)")
    print("=" * 60)

    grounder = QwenSemanticGrounder()

    all_tests = LIBERO_TEST_CASES + PARAPHRASE_TEST_CASES
    passed = 0
    total = len(all_tests)

    for tc in all_tests:
        result = grounder.ground(tc["task"], tc["objects"])

        source_ok = result.source_object == tc["expected_source"]
        target_ok = result.target_location == tc["expected_target"]

        if source_ok and target_ok:
            print(f"  PASS: {tc['name']}")
            passed += 1
        else:
            print(f"  FAIL: {tc['name']}")
            print(f"    Task: {tc['task']}")
            print(f"    Expected: {tc['expected_source']} → {tc['expected_target']}")
            print(f"    Got: {result.source_object} → {result.target_location}")
            if result.ambiguous:
                print(f"    (ambiguous, alternatives: {result.alternative_sources})")

    print(f"\nResult: {passed}/{total} passed ({100*passed/total:.1f}%)")
    return passed / total >= 0.9  # 90% threshold


def test_edge_cases_with_model():
    """Test edge cases with Qwen model."""
    print("\n" + "=" * 60)
    print("TEST: Edge cases with Qwen (requires model)")
    print("=" * 60)

    grounder = QwenSemanticGrounder()

    for tc in EDGE_CASE_TESTS:
        result = grounder.ground(tc["task"], tc["objects"])

        print(f"\n  Case: {tc['name']}")
        print(f"    Task: {tc['task']}")
        print(f"    Result: {result.source_object} → {result.target_location}")
        print(f"    Valid: {result.valid}")
        print(f"    Ambiguous: {result.ambiguous}")

        if tc.get("expect_failure"):
            if not result.valid or result.source_object != tc.get("expected_source"):
                print(f"    ✓ Expected failure/wrong result")
            else:
                print(f"    ! Unexpectedly succeeded")

        if tc.get("expect_ambiguous"):
            if result.ambiguous:
                print(f"    ✓ Correctly marked ambiguous")
            else:
                print(f"    ! Should have been marked ambiguous")


# ============================================================
# Main
# ============================================================

def run_unit_tests():
    """Run all unit tests (no model required)."""
    print("\n" + "=" * 60)
    print("UNIT TESTS (no model required)")
    print("=" * 60)

    results = []
    results.append(("parse_object_description", test_parse_object_description()))
    results.append(("spatial_descriptions", test_spatial_descriptions()))
    results.append(("ambiguity_detection", test_ambiguity_detection()))
    results.append(("output_parsing", test_grounding_output_parsing()))
    results.append(("validation", test_grounding_validation()))

    print("\n" + "=" * 60)
    print("UNIT TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    return all_passed


def run_model_tests():
    """Run tests that require Qwen model."""
    print("\n" + "=" * 60)
    print("MODEL TESTS (requires Qwen)")
    print("=" * 60)

    grounding_ok = test_grounding_with_model()
    test_edge_cases_with_model()

    return grounding_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounding Offline Tests")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run only unit tests (no model required)")
    args = parser.parse_args()

    unit_ok = run_unit_tests()

    if not args.unit_only:
        model_ok = run_model_tests()
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        if unit_ok and model_ok:
            print("All tests PASSED")
        else:
            print("Some tests FAILED")
    else:
        print("\n" + "=" * 60)
        print("FINAL RESULT (unit tests only)")
        print("=" * 60)
        if unit_ok:
            print("All unit tests PASSED")
        else:
            print("Some unit tests FAILED")
