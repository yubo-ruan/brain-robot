"""Unified evaluation script supporting Phase 1, 2, 3, and 4 modes.

Modes:
- hardcoded: Phase 1 baseline with hardcoded skill sequence
- qwen: Phase 2 with Qwen skill planning
- qwen_grounded: Phase 3 with Qwen semantic grounding + skill planning

Perception:
- oracle: Ground truth from simulator (default)
- learned: YOLO detection + depth-based pose estimation (Phase 4)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from brain_robot.config import RunConfig, SkillConfig, PerceptionConfig, LoggingConfig
from brain_robot.perception.oracle import OraclePerception
from brain_robot.perception.learned import LearnedPerception
from brain_robot.world_model.state import WorldState
from brain_robot.skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill
from brain_robot.logging.episode_logger import EpisodeLogger, RunSummary
from brain_robot.utils.seeds import set_global_seed, get_episode_seed
from brain_robot.utils.git_info import get_git_info


def make_libero_env(task_suite: str, task_id: int):
    """Create LIBERO environment."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(task_suite)()
    task = benchmark.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
        "camera_depths": True,  # Required for CGN grasp selector
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def check_libero_success(env) -> bool:
    """Check LIBERO's built-in success condition.

    This is the ground truth for whether the task goal was actually achieved,
    independent of whether our skills thought they succeeded.
    """
    if hasattr(env, 'env') and hasattr(env.env, '_check_success'):
        return bool(env.env._check_success())
    return False


def classify_task_type(task_description: str) -> str:
    """Classify task into supported types.

    Returns one of:
        - "pick_and_place": Pick X and place on/in Y
        - "open": Open drawer/cabinet
        - "close": Close drawer/cabinet
        - "turn_on": Turn on stove/appliance
        - "turn_off": Turn off stove/appliance
        - "push": Push object to location
        - "unsupported": Task type not recognized
    """
    task_lower = task_description.lower()

    # Check for open/close tasks
    if 'open' in task_lower and ('drawer' in task_lower or 'cabinet' in task_lower or 'door' in task_lower):
        return "open"
    if 'close' in task_lower and ('drawer' in task_lower or 'cabinet' in task_lower or 'door' in task_lower):
        return "close"

    # Check for turn on/off tasks
    if 'turn on' in task_lower or 'switch on' in task_lower:
        return "turn_on"
    if 'turn off' in task_lower or 'switch off' in task_lower:
        return "turn_off"

    # Check for push tasks
    if 'push' in task_lower:
        return "push"

    # Check for pick and place (most common)
    pick_keywords = ['pick up', 'pick the', 'grab', 'take', 'put the', 'place the', 'move the']
    place_keywords = ['place it', 'put it', 'on the', 'in the', 'into the', 'onto the']

    has_pick = any(kw in task_lower for kw in pick_keywords)
    has_place = any(kw in task_lower for kw in place_keywords)

    if has_pick or has_place:
        return "pick_and_place"

    return "unsupported"


def extract_expected_classes(task_description: str) -> tuple:
    """Extract expected source and target class from task description.

    Returns:
        (expected_source_class, expected_target_class) - strings like "bowl", "plate"
    """
    task_lower = task_description.lower()

    # Known object classes (including libero_object grocery items)
    OBJECT_CLASSES = [
        # Kitchen objects
        'bowl', 'plate', 'mug', 'ramekin', 'drawer', 'cabinet', 'cookie_box', 'can', 'bottle', 'stove',
        # Grocery items (libero_object)
        'alphabet_soup', 'cream_cheese', 'salad_dressing', 'bbq_sauce', 'ketchup',
        'tomato_sauce', 'butter', 'milk', 'chocolate_pudding', 'orange_juice', 'basket',
    ]

    expected_source = None
    expected_target = None

    # Find source class from task
    source_keywords = ['pick up the', 'pick the', 'grab the', 'take the']
    for kw in source_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[1]
            words = rest.split()[:5]  # Look at first 5 words
            rest_text = ' '.join(words)
            for cls in OBJECT_CLASSES:
                # Handle underscores in class names (e.g., chocolate_pudding -> chocolate pudding)
                cls_spaced = cls.replace('_', ' ')
                if cls_spaced in rest_text or cls in rest_text:
                    expected_source = cls
                    break
            break

    # Find target class from task
    target_keywords = ['place it on the', 'place it in the', 'put it on the', 'put it in the',
                       'place on the', 'place in the', 'on the', 'in the', 'into the']
    for kw in target_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[-1]
            words = rest.split()[:5]
            rest_text = ' '.join(words)
            for cls in OBJECT_CLASSES:
                # Handle underscores in class names
                cls_spaced = cls.replace('_', ' ')
                if cls_spaced in rest_text or cls in rest_text:
                    expected_target = cls
                    break
            break

    return expected_source, expected_target


def get_object_class(obj_id: str) -> str:
    """Extract class name from object ID.

    Handles both oracle IDs (akita_black_bowl_1_main) and learned IDs (bowl_0_learned).
    """
    obj_lower = obj_id.lower()

    # Check for known classes in order (more specific first)
    OBJECT_CLASSES = [
        # Kitchen objects
        'cookie_box', 'ramekin', 'cabinet', 'drawer', 'bottle', 'stove', 'bowl', 'plate', 'mug', 'can',
        # Grocery items (libero_object)
        'alphabet_soup', 'cream_cheese', 'salad_dressing', 'bbq_sauce', 'ketchup',
        'tomato_sauce', 'butter', 'milk', 'chocolate_pudding', 'orange_juice', 'basket',
    ]
    for cls in OBJECT_CLASSES:
        if cls in obj_lower:
            return cls

    return "unknown"


def parse_task_for_grounding(task_description: str, object_names: list) -> tuple:
    """Simple task parsing to find source and target objects (Phase 1 heuristic).

    Supports multiple task formats:
    - libero_spatial: "pick up the X and place it on the Y"
    - libero_goal: "put the X on the Y"
    """
    task_lower = task_description.lower()

    object_types = {}
    for obj_id in object_names:
        obj_lower = obj_id.lower()
        if 'bowl' in obj_lower:
            object_types.setdefault('bowl', []).append(obj_id)
        if 'plate' in obj_lower and 'burner' not in obj_lower:
            object_types.setdefault('plate', []).append(obj_id)
        if 'ramekin' in obj_lower:
            object_types.setdefault('ramekin', []).append(obj_id)
        if 'mug' in obj_lower:
            object_types.setdefault('mug', []).append(obj_id)
        if 'drawer' in obj_lower:
            object_types.setdefault('drawer', []).append(obj_id)
        if 'cabinet' in obj_lower:
            # For cabinets: prefer cabinet_top for "on top of" tasks, then main/base
            if 'cabinet_top' in obj_lower:
                object_types.setdefault('cabinet_top', []).append(obj_id)
                object_types.setdefault('cabinet', []).append(obj_id)
            elif '_main' in obj_lower or '_base' in obj_lower:
                if 'cabinet_middle' not in obj_lower and 'cabinet_bottom' not in obj_lower:
                    object_types.setdefault('cabinet', []).insert(0, obj_id)
        if 'stove' in obj_lower:
            # For stoves, prefer the main body but accept burner/plate as fallback
            if 'burner' not in obj_lower and 'button' not in obj_lower:
                object_types.setdefault('stove', []).insert(0, obj_id)  # Main stove first
            elif 'burner_plate' in obj_lower or 'burner' in obj_lower:
                object_types.setdefault('stove', []).append(obj_id)  # Burner as fallback
        if 'bottle' in obj_lower:
            object_types.setdefault('bottle', []).append(obj_id)
        # Grocery items (libero_object tasks)
        if 'basket' in obj_lower:
            object_types.setdefault('basket', []).append(obj_id)
        if 'alphabet_soup' in obj_lower or 'alphabet soup' in obj_lower:
            object_types.setdefault('alphabet soup', []).append(obj_id)
            object_types.setdefault('alphabet_soup', []).append(obj_id)
        if 'salad_dressing' in obj_lower or 'salad dressing' in obj_lower:
            object_types.setdefault('salad dressing', []).append(obj_id)
            object_types.setdefault('salad_dressing', []).append(obj_id)
        if 'bbq_sauce' in obj_lower or 'bbq sauce' in obj_lower:
            object_types.setdefault('bbq sauce', []).append(obj_id)
            object_types.setdefault('bbq_sauce', []).append(obj_id)
        if 'ketchup' in obj_lower:
            object_types.setdefault('ketchup', []).append(obj_id)
        if 'tomato_sauce' in obj_lower or 'tomato sauce' in obj_lower:
            object_types.setdefault('tomato sauce', []).append(obj_id)
            object_types.setdefault('tomato_sauce', []).append(obj_id)
        if 'cream_cheese' in obj_lower or 'cream cheese' in obj_lower:
            object_types.setdefault('cream cheese', []).append(obj_id)
            object_types.setdefault('cream_cheese', []).append(obj_id)
        if 'butter' in obj_lower:
            object_types.setdefault('butter', []).append(obj_id)
        if 'milk' in obj_lower:
            object_types.setdefault('milk', []).append(obj_id)
        if 'chocolate_pudding' in obj_lower or 'chocolate pudding' in obj_lower:
            object_types.setdefault('chocolate pudding', []).append(obj_id)
            object_types.setdefault('chocolate_pudding', []).append(obj_id)
        if 'orange_juice' in obj_lower or 'orange juice' in obj_lower:
            object_types.setdefault('orange juice', []).append(obj_id)
            object_types.setdefault('orange_juice', []).append(obj_id)
        if 'wine_bottle' in obj_lower or 'wine bottle' in obj_lower:
            object_types.setdefault('wine bottle', []).append(obj_id)
            object_types.setdefault('wine_bottle', []).append(obj_id)

    source_obj = None
    target_obj = None

    # Pattern 1: "pick up the X" / "grab the X" / etc.
    source_keywords = ['pick up the', 'pick the', 'grab the', 'take the']
    for kw in source_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[1]
            # Extract first few words (up to "and" or end)
            rest_words = rest.split()
            rest_prefix = ' '.join(rest_words[:5])  # Take first 5 words for matching
            for obj_type, obj_list in object_types.items():
                # Check if obj_type (e.g., "alphabet soup") is in the prefix
                if obj_type in rest_prefix:
                    source_obj = obj_list[0]
                    break
            if source_obj:
                break

    # Pattern 2: "put the X on/in the Y" (libero_goal style) - source comes before target
    if not source_obj:
        import re
        # Handle "put the X on the Y", "put the X in the Y", "put the X on top of the Y"
        # Note: source can be multi-word like "wine bottle"
        put_match = re.search(r'put the (.+?) (on top of|on|in|into) the', task_lower)
        if put_match:
            source_phrase = put_match.group(1).strip()
            # Try matching source_phrase to object types
            for obj_type, obj_list in object_types.items():
                if obj_type in source_phrase or source_phrase.replace(' ', '_') in obj_type:
                    source_obj = obj_list[0]
                    break
            # Also try without spaces (wine bottle -> wine_bottle -> bottle)
            if not source_obj:
                for obj_type, obj_list in object_types.items():
                    source_words = source_phrase.split()
                    if any(word == obj_type for word in source_words):
                        source_obj = obj_list[0]
                        break
            # Also extract target from the same pattern
            target_part = task_lower.split(put_match.group(0))[-1].strip()
            preposition = put_match.group(2)  # "on top of", "on", "in", etc.

            # For "on top of", prefer cabinet_top over cabinet
            if preposition == 'on top of':
                for obj_type, obj_list in object_types.items():
                    target_base = target_part.split()[0]
                    if obj_type == f'{target_base}_top':
                        target_obj = obj_list[0]
                        break

            # Fallback to regular matching
            if not target_obj:
                target_prefix = ' '.join(target_part.split()[:5])
                for obj_type, obj_list in object_types.items():
                    if obj_type in target_prefix:
                        target_obj = obj_list[0]
                        break

    # Pattern 3: "place it on the Y" / "on the Y"
    if not target_obj:
        target_keywords = ['place it on the', 'place it in the', 'put it on the', 'put it in the',
                           'place on the', 'place in the', 'on the', 'in the', 'into the']
        for kw in target_keywords:
            if kw in task_lower:
                rest = task_lower.split(kw)[-1]
                rest_prefix = ' '.join(rest.split()[:5])
                for obj_type, obj_list in object_types.items():
                    if obj_type in rest_prefix:
                        target_obj = obj_list[0]
                        break
                if target_obj:
                    break

    # Fallback: if source found but no target, pick different object type
    if source_obj and not target_obj:
        source_type = None
        for obj_type, obj_list in object_types.items():
            if source_obj in obj_list:
                source_type = obj_type
                break
        for obj_type, obj_list in object_types.items():
            if obj_type != source_type:
                target_obj = obj_list[0]
                break

    # NOTE: Removed dangerous fallback that picked first available object
    # when expected class wasn't found. This caused "lucky" successes where
    # YOLO misclassified the bowl but fallback still picked it.
    # Now we return None to fail honestly rather than guess.

    if not source_obj:
        print(f"  ⚠ WARNING: No source object found matching task description")
    if not target_obj:
        print(f"  ⚠ WARNING: No target object found matching task description")

    return source_obj, target_obj


@dataclass
class EpisodeResult:
    """Result of a single episode with both physical and semantic success."""
    physical_success: bool  # Our skills reported success
    libero_success: bool  # LIBERO's ground truth success (task goal achieved)
    semantic_source_correct: bool
    semantic_target_correct: bool
    chosen_source_id: str = ""
    chosen_target_id: str = ""
    chosen_source_class: str = ""
    chosen_target_class: str = ""
    expected_source_class: str = ""
    expected_target_class: str = ""
    # Failure taxonomy fields
    failed_skill: str = ""  # Which skill failed
    failure_reason: str = ""  # Why it failed
    # Task type classification
    task_type: str = "pick_and_place"  # From classify_task_type()

    @property
    def semantic_success(self) -> bool:
        """Both source and target must be semantically correct."""
        return self.semantic_source_correct and self.semantic_target_correct


def extract_failure_reason(result_info: dict) -> str:
    """Extract failure reason from skill result info."""
    if result_info.get("timeout"):
        return "timeout"
    if result_info.get("stuck"):
        return "stuck"
    if "precondition" in result_info.get("error_msg", "").lower():
        return "precondition"
    if "closed on air" in result_info.get("error_msg", "").lower():
        return "closed_on_air"
    if "not holding" in result_info.get("error_msg", "").lower():
        return "not_holding"
    if "not found" in result_info.get("error_msg", "").lower():
        return "object_not_found"
    if result_info.get("error_msg"):
        return result_info["error_msg"][:50]
    return "unknown"


def print_failure_taxonomy(semantic_results: list):
    """Print failure taxonomy summary."""
    if not semantic_results:
        return

    skill_failures = {}
    reason_counts = {}
    task_type_counts = {}
    unsupported_tasks = 0

    for r in semantic_results:
        # Count task types
        task_type = r.get("task_type", "pick_and_place")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        if not r["physical_success"]:
            reason = r.get("failure_reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            # Count unsupported task types
            if reason.startswith("unsupported_task_type:"):
                unsupported_tasks += 1
            elif r.get("failed_skill"):
                skill = r["failed_skill"]
                skill_failures[skill] = skill_failures.get(skill, 0) + 1

    print("\n" + "-" * 60)
    print("FAILURE TAXONOMY")
    print("-" * 60)

    # Task type breakdown
    print(f"\nBy Task Type:")
    for task_type, count in sorted(task_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {task_type}: {count}")

    if unsupported_tasks > 0:
        print(f"\n⚠ Unsupported task types: {unsupported_tasks} episodes skipped")

    if skill_failures:
        total_skill_failures = sum(skill_failures.values())
        print(f"\nBy Skill ({total_skill_failures} skill failures):")
        for skill in ["ApproachObject", "GraspObject", "MoveObjectToRegion", "PlaceObject"]:
            if skill in skill_failures:
                count = skill_failures[skill]
                print(f"  {skill}: {count} ({100*count/total_skill_failures:.0f}%)")

    if reason_counts:
        total_failures = sum(reason_counts.values())
        print(f"\nBy Reason ({total_failures} total failures):")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({100*count/total_failures:.0f}%)")


def run_episode_hardcoded(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    config: SkillConfig,
    logger: EpisodeLogger,
    grasp_selector_type: str = "heuristic",
) -> EpisodeResult:
    """Run episode with hardcoded skill sequence (Phase 1 baseline)."""
    # Classify task type first
    task_type = classify_task_type(task_description)
    print(f"  Task type: {task_type}")

    # Handle unsupported task types gracefully
    if task_type not in ["pick_and_place"]:
        print(f"  ⚠ Task type '{task_type}' not supported by current skill pipeline")
        return EpisodeResult(
            physical_success=False,
            libero_success=False,
            semantic_source_correct=False,
            semantic_target_correct=False,
            failure_reason=f"unsupported_task_type:{task_type}",
            task_type=task_type,
        )

    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    logger.log_world_state(world_state)

    # Extract expected classes from task
    expected_source_class, expected_target_class = extract_expected_classes(task_description)

    if len(perc_result.object_names) < 2:
        return EpisodeResult(
            physical_success=False,
            libero_success=check_libero_success(env),
            semantic_source_correct=False,
            semantic_target_correct=False,
            expected_source_class=expected_source_class or "",
            expected_target_class=expected_target_class or "",
            failure_reason="insufficient_objects",
            task_type=task_type,
        )

    source_obj, target_obj = parse_task_for_grounding(task_description, perc_result.object_names)

    # Fail early if grounding failed (no matching object found)
    if source_obj is None or target_obj is None:
        print(f"  ✗ Grounding failed: source={source_obj}, target={target_obj}")
        return EpisodeResult(
            physical_success=False,
            libero_success=check_libero_success(env),
            semantic_source_correct=False,
            semantic_target_correct=False,
            chosen_source_id=source_obj or "",
            chosen_target_id=target_obj or "",
            chosen_source_class="none",
            chosen_target_class="none",
            expected_source_class=expected_source_class or "",
            expected_target_class=expected_target_class or "",
            failure_reason="grounding_failed",
            task_type=task_type,
        )

    # Check semantic correctness
    chosen_source_class = get_object_class(source_obj) if source_obj else "unknown"
    chosen_target_class = get_object_class(target_obj) if target_obj else "unknown"
    semantic_source_correct = (chosen_source_class == expected_source_class) if expected_source_class else True
    semantic_target_correct = (chosen_target_class == expected_target_class) if expected_target_class else True

    print(f"  Source: {source_obj} (class={chosen_source_class}, expected={expected_source_class})")
    print(f"  Target: {target_obj} (class={chosen_target_class}, expected={expected_target_class})")
    print(f"  Grasp selector: {grasp_selector_type}")

    skills = [
        (ApproachSkill(config=config), {"obj": source_obj}),
        (GraspSkill(config=config, grasp_selector_type=grasp_selector_type), {"obj": source_obj}),
        (MoveSkill(config=config), {"obj": source_obj, "region": target_obj}),
        (PlaceSkill(config=config), {"obj": source_obj, "region": target_obj}),
    ]

    step_count = 0

    for skill, args in skills:
        with logger.get_timer().measure("perception"):
            perc_result = perception.perceive(env)
            world_state.update_from_perception(perc_result)

        skill_timer_name = f"skill_{skill.name}"
        with logger.get_timer().measure(skill_timer_name):
            result = skill.run(env, world_state, args)

        logger.log_skill(skill.name, args, result)
        logger.log_world_state(world_state)

        step_count += result.info.get("steps_taken", 0)

        if not result.success:
            failure_reason = extract_failure_reason(result.info)
            print(f"  {skill.name} failed: {result.info.get('error_msg', 'Unknown')}")
            return EpisodeResult(
                physical_success=False,
                libero_success=check_libero_success(env),
                semantic_source_correct=semantic_source_correct,
                semantic_target_correct=semantic_target_correct,
                chosen_source_id=source_obj or "",
                chosen_target_id=target_obj or "",
                chosen_source_class=chosen_source_class,
                chosen_target_class=chosen_target_class,
                expected_source_class=expected_source_class or "",
                expected_target_class=expected_target_class or "",
                failed_skill=skill.name,
                failure_reason=failure_reason,
                task_type=task_type,
            )

        print(f"  {skill.name}: OK ({result.info.get('steps_taken', 0)} steps)")

    print(f"  Total steps: {step_count}")

    # Check LIBERO's ground truth success
    libero_success = check_libero_success(env)
    print(f"  LIBERO success: {libero_success}")

    return EpisodeResult(
        physical_success=True,
        libero_success=libero_success,
        semantic_source_correct=semantic_source_correct,
        semantic_target_correct=semantic_target_correct,
        chosen_source_id=source_obj or "",
        chosen_target_id=target_obj or "",
        chosen_source_class=chosen_source_class,
        chosen_target_class=chosen_target_class,
        expected_source_class=expected_source_class or "",
        expected_target_class=expected_target_class or "",
        task_type=task_type,
    )


def run_episode_qwen(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    config: SkillConfig,
    logger: EpisodeLogger,
    planner,
    metrics,
    task_id: str,
) -> bool:
    """Run episode with Qwen skill planning (Phase 2)."""
    from brain_robot.planning.prompts import prepare_world_state_for_qwen

    # Get initial perception
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    logger.log_world_state(world_state)

    if len(perc_result.object_names) < 2:
        return False

    # Plan with Qwen
    print("  Planning with Qwen...")
    plan_result = planner.plan(
        task_description=task_description,
        world_state=world_state,
        metrics=metrics,
        task_id=task_id,
    )

    if not plan_result.success:
        print(f"  Planning failed: {plan_result.error}")
        return False

    print(f"  Plan generated: {len(plan_result.plan)} steps")
    for i, step in enumerate(plan_result.plan):
        print(f"    {i+1}. {step['skill']}({step['args']})")

    # Log Qwen interaction
    if plan_result.prompt and plan_result.raw_output:
        logger.log_qwen(plan_result.prompt, plan_result.raw_output)

    # Execute plan
    success, exec_info = planner.execute_plan(
        plan=plan_result.plan,
        env=env,
        world_state=world_state,
        config=config,
        metrics=metrics,
        task_id=task_id,
        logger=logger,
        perception=perception,
    )

    if success:
        metrics.record_goal_reached(task_id)
        print(f"  Total steps: {exec_info.get('steps_taken', 0)}")
    else:
        metrics.record_goal_not_reached(task_id, plan_result.plan, exec_info.get("error", "Unknown"))
        print(f"  Execution failed at step {exec_info.get('failed_step', '?')}: {exec_info.get('error', 'Unknown')}")

    return success


def run_episode_qwen_grounded(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    config: SkillConfig,
    logger: EpisodeLogger,
    grounder,
    grounding_metrics,
    task_id: str,
) -> bool:
    """Run episode with Qwen semantic grounding + hardcoded skill sequence (Phase 3).

    This mode uses Qwen to ground the task language to object IDs,
    then executes a deterministic skill sequence.
    """
    from brain_robot.grounding.enriched_object import enrich_objects

    # Get initial perception
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    logger.log_world_state(world_state)

    if len(perc_result.object_names) < 2:
        return False

    # Enrich objects with human-readable descriptions
    enriched = enrich_objects(world_state)

    # Ground task to object IDs using Qwen
    print("  Grounding with Qwen...")
    grounding_result = grounder.ground(
        task_description=task_description,
        objects=enriched,
        metrics=grounding_metrics,
        task_id=task_id,
    )

    if not grounding_result.valid:
        print(f"  Grounding failed: {grounding_result.error}")
        return False

    source_obj = grounding_result.source_object
    target_obj = grounding_result.target_location

    print(f"  Grounded: source={source_obj}, target={target_obj}")
    print(f"  Confidence: {grounding_result.confidence}")
    if grounding_result.ambiguous:
        print(f"  WARNING: Ambiguous grounding! Alternatives: {grounding_result.alternative_sources}")

    # Log grounding interaction
    if grounding_result.prompt and grounding_result.raw_output:
        logger.log_qwen(grounding_result.prompt, grounding_result.raw_output)

    # Execute deterministic skill sequence with grounded objects
    skills = [
        (ApproachSkill(config=config), {"obj": source_obj}),
        (GraspSkill(config=config), {"obj": source_obj}),
        (MoveSkill(config=config), {"obj": source_obj, "region": target_obj}),
        (PlaceSkill(config=config), {"obj": source_obj, "region": target_obj}),
    ]

    step_count = 0

    for skill, args in skills:
        # Update perception before each skill
        with logger.get_timer().measure("perception"):
            perc_result = perception.perceive(env)
            world_state.update_from_perception(perc_result)

        skill_timer_name = f"skill_{skill.name}"
        with logger.get_timer().measure(skill_timer_name):
            result = skill.run(env, world_state, args)

        logger.log_skill(skill.name, args, result)
        logger.log_world_state(world_state)

        step_count += result.info.get("steps_taken", 0)

        if not result.success:
            print(f"  {skill.name} failed: {result.info.get('error_msg', 'Unknown')}")
            return False

        print(f"  {skill.name}: OK ({result.info.get('steps_taken', 0)} steps)")

    print(f"  Total steps: {step_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation (Phase 1, 2 & 3)")
    parser.add_argument("--mode", type=str, choices=["hardcoded", "qwen", "qwen_grounded"], default="hardcoded",
                        help="Planning mode: hardcoded (Phase 1), qwen (Phase 2), or qwen_grounded (Phase 3)")
    parser.add_argument("--perception", type=str, choices=["oracle", "learned"], default="oracle",
                        help="Perception mode: oracle (ground truth) or learned (YOLO + depth)")
    parser.add_argument("--bootstrap", type=str, choices=["oracle", "cold"], default="oracle",
                        help="Bootstrap mode for learned perception: oracle (use GT at t=0) or cold (detection only)")
    parser.add_argument("--model-path", type=str, default="models/yolo_libero.pt",
                        help="Path to YOLO model (only used with --perception learned --detector yolo)")
    parser.add_argument("--detector", type=str, choices=["yolo", "gdino", "gsam"], default="yolo",
                        help="Detector type: yolo (trained), gdino (open-vocab), gsam (open-vocab + masks)")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/evaluation")
    parser.add_argument("--grasp-selector", type=str, choices=["heuristic", "contact_graspnet", "hybrid"], default="heuristic",
                        help="Grasp selector: heuristic (rule-based), contact_graspnet (learned 6-DoF), or hybrid (heuristic + CGN refinement)")
    args = parser.parse_args()

    # Validate model path for learned perception with YOLO
    if args.perception == "learned" and args.detector == "yolo" and not Path(args.model_path).exists():
        print(f"Error: YOLO model not found at {args.model_path}")
        print("Train a model first with: python scripts/train_yolo_detector.py")
        return 1

    # Create config
    config = RunConfig(
        seed=args.seed,
        task_suite=args.task_suite,
        task_id=args.task_id,
        n_episodes=args.n_episodes,
        skill=SkillConfig(),
        perception=PerceptionConfig(use_oracle=(args.perception == "oracle")),
        logging=LoggingConfig(output_dir=args.output_dir),
    )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perception_suffix = f"_{args.perception}" if args.perception != "oracle" else ""
    detector_suffix = f"_{args.detector}" if args.perception == "learned" and args.detector != "yolo" else ""
    bootstrap_suffix = f"_{args.bootstrap}" if args.perception == "learned" and args.bootstrap != "oracle" else ""
    grasp_suffix = f"_{args.grasp_selector}" if args.grasp_selector != "heuristic" else ""
    output_dir = Path(args.output_dir) / f"{args.mode}{perception_suffix}{detector_suffix}{bootstrap_suffix}{grasp_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "mode": args.mode,
            "perception": args.perception,
            "detector": args.detector if args.perception == "learned" else None,
            "bootstrap": args.bootstrap if args.perception == "learned" else None,
            "model_path": args.model_path if args.perception == "learned" and args.detector == "yolo" else None,
            "grasp_selector": args.grasp_selector,
            **config.to_dict()
        }, f, indent=2)

    phase_names = {
        "hardcoded": "Phase 1 (Hardcoded)",
        "qwen": "Phase 2 (Qwen Planning)",
        "qwen_grounded": "Phase 3 (Qwen Grounding)",
    }
    phase_name = phase_names.get(args.mode, args.mode)
    detector_names = {"yolo": "YOLO", "gdino": "Grounding-DINO", "gsam": "Grounded-SAM"}
    if args.perception == "learned":
        perception_name = f"Learned ({detector_names.get(args.detector, args.detector)}+Depth)"
    else:
        perception_name = "Oracle"

    print("=" * 60)
    print(f"Evaluation: {phase_name}")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Perception: {perception_name}")
    print(f"Grasp Selector: {args.grasp_selector}")
    if args.perception == "learned":
        print(f"Detector: {args.detector}")
        if args.detector == "yolo":
            print(f"Model: {args.model_path}")
        print(f"Bootstrap: {args.bootstrap}")
    print(f"Task Suite: {args.task_suite}")
    print(f"Task ID: {args.task_id}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Git: {get_git_info()}")
    print("=" * 60)

    # Create environment
    env, task_description = make_libero_env(args.task_suite, args.task_id)
    print(f"Task: {task_description}")

    # Setup perception
    oracle_perception = OraclePerception()  # Always need oracle for bootstrap/reference

    if args.perception == "learned":
        # Extract target objects from task description for open-vocab detectors
        target_objects = None
        if args.detector in ["gdino", "gsam"]:
            # Extract source/target from task for open-vocab detection
            expected_source, expected_target = extract_expected_classes(task_description)
            target_objects = []
            if expected_source:
                target_objects.append(expected_source)
            if expected_target:
                target_objects.append(expected_target)
            # Add basket as common target for libero_object tasks
            if "basket" not in target_objects:
                target_objects.append("basket")
            print(f"Target objects for open-vocab detector: {target_objects}")

        learned_perception = LearnedPerception(
            model_path=args.model_path,
            confidence_threshold=0.5 if args.detector == "yolo" else 0.35,
            image_size=(256, 256),
            detector_type=args.detector,
            target_objects=target_objects,
        )
        detector_name = {"yolo": "YOLO", "gdino": "Grounding-DINO", "gsam": "Grounded-SAM"}
        print(f"Warming up {detector_name.get(args.detector, args.detector)} detector...")
        learned_perception.detector.warmup()
        perception = learned_perception
    else:
        perception = oracle_perception

    logger = EpisodeLogger(str(output_dir), config=config)

    # Setup planner/grounder and metrics based on mode
    planner = None
    metrics = None
    grounder = None
    grounding_metrics = None

    if args.mode == "qwen":
        from brain_robot.planning import QwenSkillPlanner, PlannerMetrics
        planner = QwenSkillPlanner(temperature=0.1, max_retries=2)
        metrics = PlannerMetrics()
    elif args.mode == "qwen_grounded":
        from brain_robot.grounding import QwenSemanticGrounder, GroundingMetrics
        grounder = QwenSemanticGrounder(temperature=0.1, max_retries=2)
        grounding_metrics = GroundingMetrics()

    # Run episodes
    successes = 0
    semantic_results = []  # Track semantic correctness per episode
    task_id_str = f"{args.task_suite}_{args.task_id}"

    for episode_idx in range(args.n_episodes):
        episode_seed = get_episode_seed(args.seed, episode_idx)
        set_global_seed(episode_seed, env)

        print(f"\nEpisode {episode_idx + 1}/{args.n_episodes} (seed={episode_seed})")

        env.reset()
        world_state = WorldState()

        # Bootstrap learned perception at episode start
        if args.perception == "learned":
            learned_perception.reset()
            if args.bootstrap == "oracle":
                oracle_result = oracle_perception.perceive(env)
                learned_perception.bootstrap_from_oracle(oracle_result, env)
            else:  # cold start
                learned_perception.bootstrap_from_detections(env)

        logger.start_episode(
            task=task_description,
            task_id=args.task_id,
            episode_idx=episode_idx,
            seed=episode_seed,
        )

        # Track semantic success separately
        episode_result = None

        try:
            if args.mode == "hardcoded":
                episode_result = run_episode_hardcoded(
                    env=env,
                    task_description=task_description,
                    world_state=world_state,
                    perception=perception,
                    config=config.skill,
                    logger=logger,
                    grasp_selector_type=args.grasp_selector,
                )
                success = episode_result.physical_success
            elif args.mode == "qwen":
                success = run_episode_qwen(
                    env=env,
                    task_description=task_description,
                    world_state=world_state,
                    perception=perception,
                    config=config.skill,
                    logger=logger,
                    planner=planner,
                    metrics=metrics,
                    task_id=task_id_str,
                )
            elif args.mode == "qwen_grounded":
                success = run_episode_qwen_grounded(
                    env=env,
                    task_description=task_description,
                    world_state=world_state,
                    perception=perception,
                    config=config.skill,
                    logger=logger,
                    grounder=grounder,
                    grounding_metrics=grounding_metrics,
                    task_id=task_id_str,
                )
        except Exception as e:
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            success = False

        logger.end_episode(
            success=success,
            failure_reason=None if success else "Execution failed",
        )

        if success:
            successes += 1
            print(f"  Result: SUCCESS")
        else:
            print(f"  Result: FAILURE")

        # Track semantic metrics for hardcoded mode
        if episode_result is not None:
            semantic_results.append({
                "episode": episode_idx,
                "physical_success": episode_result.physical_success,
                "libero_success": episode_result.libero_success,
                "semantic_source_correct": episode_result.semantic_source_correct,
                "semantic_target_correct": episode_result.semantic_target_correct,
                "semantic_success": episode_result.semantic_success,
                "chosen_source_id": episode_result.chosen_source_id,
                "chosen_target_id": episode_result.chosen_target_id,
                "chosen_source_class": episode_result.chosen_source_class,
                "chosen_target_class": episode_result.chosen_target_class,
                "expected_source_class": episode_result.expected_source_class,
                "expected_target_class": episode_result.expected_target_class,
                "failed_skill": episode_result.failed_skill,
                "failure_reason": episode_result.failure_reason,
                "task_type": episode_result.task_type,
            })

            if not episode_result.semantic_source_correct:
                print(f"  ⚠ SEMANTIC ERROR: source is {episode_result.chosen_source_class}, expected {episode_result.expected_source_class}")
            if not episode_result.semantic_target_correct:
                print(f"  ⚠ SEMANTIC ERROR: target is {episode_result.chosen_target_class}, expected {episode_result.expected_target_class}")

    # Final summary
    success_rate = successes / args.n_episodes

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Total Episodes: {args.n_episodes}")
    print(f"Successful: {successes}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Target: {config.success_rate_threshold:.1%}")

    if success_rate >= config.success_rate_threshold:
        print("\n✓ SUCCESS CRITERIA MET")
    else:
        print(f"\n✗ Below target ({success_rate:.1%} < {config.success_rate_threshold:.1%})")

    # Print semantic metrics if available
    if semantic_results:
        n_semantic = len(semantic_results)
        n_physical_success = sum(1 for r in semantic_results if r["physical_success"])
        n_libero_success = sum(1 for r in semantic_results if r.get("libero_success", False))
        n_semantic_source = sum(1 for r in semantic_results if r["semantic_source_correct"])
        n_semantic_target = sum(1 for r in semantic_results if r["semantic_target_correct"])
        n_semantic_both = sum(1 for r in semantic_results if r["semantic_success"])

        # Count "lucky" successes (physical success but semantic failure)
        n_lucky = sum(1 for r in semantic_results
                      if r["physical_success"] and not r["semantic_success"])
        # Count grounding failures (no object found)
        n_grounding_fail = sum(1 for r in semantic_results
                               if r.get("chosen_source_class") == "none" or r.get("chosen_target_class") == "none")
        # Count "fake" successes (skills said OK but LIBERO says not achieved)
        n_fake = sum(1 for r in semantic_results
                     if r["physical_success"] and not r.get("libero_success", False))

        print("\n" + "-" * 60)
        print("SUCCESS METRICS")
        print("-" * 60)
        print(f"Skill Success Rate:        {100*n_physical_success/n_semantic:.1f}% ({n_physical_success}/{n_semantic}) [skills reported OK]")
        print(f"LIBERO Success Rate:       {100*n_libero_success/n_semantic:.1f}% ({n_libero_success}/{n_semantic}) [actual task goal achieved]")
        if n_fake > 0:
            print(f"Fake Success Rate:         {100*n_fake/n_semantic:.1f}% ({n_fake}/{n_semantic}) [skills OK but goal NOT achieved]")

        print("\n" + "-" * 60)
        print("SEMANTIC CORRECTNESS")
        print("-" * 60)
        print(f"Semantic Source Correct:   {100*n_semantic_source/n_semantic:.1f}% ({n_semantic_source}/{n_semantic})")
        print(f"Semantic Target Correct:   {100*n_semantic_target/n_semantic:.1f}% ({n_semantic_target}/{n_semantic})")
        print(f"Semantic Both Correct:     {100*n_semantic_both/n_semantic:.1f}% ({n_semantic_both}/{n_semantic})")
        print(f"Lucky Success Rate:        {100*n_lucky/n_semantic:.1f}% ({n_lucky}/{n_semantic}) [physical ∧ ¬semantic]")
        if n_grounding_fail > 0:
            print(f"Grounding Failures:        {100*n_grounding_fail/n_semantic:.1f}% ({n_grounding_fail}/{n_semantic}) [no matching object]")

        # Print failure taxonomy
        print_failure_taxonomy(semantic_results)

        # Check for dangerous cases
        if n_lucky > 0:
            print(f"\n⚠ WARNING: {n_lucky} episodes succeeded physically but with WRONG objects!")
            print("   This means the robot did the manipulation correctly but on the wrong object.")
        if n_fake > 0:
            print(f"\n⚠ WARNING: {n_fake} episodes had skill success but LIBERO goal NOT achieved!")
            print("   This means our skills thought they succeeded but the task was NOT completed.")

    # Print planner metrics for Qwen mode
    if metrics:
        metrics.print_summary()
        metrics.save(str(output_dir / "planner_metrics.json"))

    # Print grounding metrics for qwen_grounded mode
    if grounding_metrics:
        print("\n--- Grounding Metrics ---")
        gm_summary = grounding_metrics.summary()
        print(f"Total attempts: {gm_summary['total_attempts']}")
        print(f"Parse rate: {gm_summary['parse_rate']:.1%}")
        print(f"Validation rate: {gm_summary['validation_rate']:.1%}")
        print(f"Ambiguous rate: {gm_summary['ambiguous_rate']:.1%}")

        import json as json_mod
        with open(output_dir / "grounding_metrics.json", "w") as f:
            json_mod.dump(gm_summary, f, indent=2)

    # Save final summary
    summary = {
        "mode": args.mode,
        "perception": args.perception,
        "bootstrap": args.bootstrap if args.perception == "learned" else None,
        "model_path": args.model_path if args.perception == "learned" else None,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": task_description,
        "n_episodes": args.n_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "target_rate": config.success_rate_threshold,
        "passed": success_rate >= config.success_rate_threshold,
        "config": config.to_dict(),
        "git_info": get_git_info(),
    }

    # Add semantic metrics
    if semantic_results:
        n_semantic = len(semantic_results)
        summary["semantic_metrics"] = {
            "physical_success_rate": sum(1 for r in semantic_results if r["physical_success"]) / n_semantic,
            "semantic_source_correct_rate": sum(1 for r in semantic_results if r["semantic_source_correct"]) / n_semantic,
            "semantic_target_correct_rate": sum(1 for r in semantic_results if r["semantic_target_correct"]) / n_semantic,
            "semantic_both_correct_rate": sum(1 for r in semantic_results if r["semantic_success"]) / n_semantic,
        }
        summary["semantic_results"] = semantic_results

    if metrics:
        summary["planner_metrics"] = metrics.summary()

    if grounding_metrics:
        summary["grounding_metrics"] = grounding_metrics.summary()

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    env.close()

    return 0 if success_rate >= config.success_rate_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
