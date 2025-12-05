"""Unified evaluation script supporting Phase 1, 2, and 3 modes.

Modes:
- hardcoded: Phase 1 baseline with hardcoded skill sequence
- qwen: Phase 2 with Qwen skill planning
- qwen_grounded: Phase 3 with Qwen semantic grounding + skill planning
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import RunConfig, SkillConfig, PerceptionConfig, LoggingConfig
from src.perception.oracle import OraclePerception
from src.world_model.state import WorldState
from src.skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill
from src.logging.episode_logger import EpisodeLogger, RunSummary
from src.utils.seeds import set_global_seed, get_episode_seed
from src.utils.git_info import get_git_info


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
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def parse_task_for_grounding(task_description: str, object_names: list) -> tuple:
    """Simple task parsing to find source and target objects (Phase 1 heuristic)."""
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
            object_types.setdefault('cabinet', []).append(obj_id)

    source_obj = None
    target_obj = None

    source_keywords = ['pick up the', 'pick the', 'grab the', 'take the']
    for kw in source_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[1]
            for obj_type, obj_list in object_types.items():
                if obj_type in rest.split()[0:3]:
                    source_obj = obj_list[0]
                    break
            if source_obj:
                break

    target_keywords = ['place it on the', 'place it in the', 'put it on the', 'put it in the',
                       'place on the', 'place in the', 'on the', 'in the', 'into the']
    for kw in target_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[-1]
            for obj_type, obj_list in object_types.items():
                if obj_type in rest.split()[0:3]:
                    target_obj = obj_list[0]
                    break
            if target_obj:
                break

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

    if not source_obj and len(object_names) >= 1:
        source_obj = object_names[0]
    if not target_obj and len(object_names) >= 2:
        target_obj = object_names[1]

    return source_obj, target_obj


def run_episode_hardcoded(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    config: SkillConfig,
    logger: EpisodeLogger,
) -> bool:
    """Run episode with hardcoded skill sequence (Phase 1 baseline)."""
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    logger.log_world_state(world_state)

    if len(perc_result.object_names) < 2:
        return False

    source_obj, target_obj = parse_task_for_grounding(task_description, perc_result.object_names)

    print(f"  Source: {source_obj}")
    print(f"  Target: {target_obj}")

    skills = [
        (ApproachSkill(config=config), {"obj": source_obj}),
        (GraspSkill(config=config), {"obj": source_obj}),
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
            print(f"  {skill.name} failed: {result.info.get('error_msg', 'Unknown')}")
            return False

        print(f"  {skill.name}: OK ({result.info.get('steps_taken', 0)} steps)")

    print(f"  Total steps: {step_count}")
    return True


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
    from src.planning.prompts import prepare_world_state_for_qwen

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
    from src.grounding.enriched_object import enrich_objects

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
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/evaluation")
    args = parser.parse_args()

    # Create config
    config = RunConfig(
        seed=args.seed,
        task_suite=args.task_suite,
        task_id=args.task_id,
        n_episodes=args.n_episodes,
        skill=SkillConfig(),
        perception=PerceptionConfig(use_oracle=True),
        logging=LoggingConfig(output_dir=args.output_dir),
    )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"mode": args.mode, **config.to_dict()}, f, indent=2)

    phase_names = {
        "hardcoded": "Phase 1 (Hardcoded)",
        "qwen": "Phase 2 (Qwen Planning)",
        "qwen_grounded": "Phase 3 (Qwen Grounding)",
    }
    phase_name = phase_names.get(args.mode, args.mode)

    print("=" * 60)
    print(f"Evaluation: {phase_name}")
    print("=" * 60)
    print(f"Mode: {args.mode}")
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

    # Setup components
    perception = OraclePerception()
    logger = EpisodeLogger(str(output_dir), config=config)

    # Setup planner/grounder and metrics based on mode
    planner = None
    metrics = None
    grounder = None
    grounding_metrics = None

    if args.mode == "qwen":
        from src.planning import QwenSkillPlanner, PlannerMetrics
        planner = QwenSkillPlanner(temperature=0.1, max_retries=2)
        metrics = PlannerMetrics()
    elif args.mode == "qwen_grounded":
        from src.grounding import QwenSemanticGrounder, GroundingMetrics
        grounder = QwenSemanticGrounder(temperature=0.1, max_retries=2)
        grounding_metrics = GroundingMetrics()

    # Run episodes
    successes = 0
    task_id_str = f"{args.task_suite}_{args.task_id}"

    for episode_idx in range(args.n_episodes):
        episode_seed = get_episode_seed(args.seed, episode_idx)
        set_global_seed(episode_seed, env)

        print(f"\nEpisode {episode_idx + 1}/{args.n_episodes} (seed={episode_seed})")

        env.reset()
        world_state = WorldState()

        logger.start_episode(
            task=task_description,
            task_id=args.task_id,
            episode_idx=episode_idx,
            seed=episode_seed,
        )

        try:
            if args.mode == "hardcoded":
                success = run_episode_hardcoded(
                    env=env,
                    task_description=task_description,
                    world_state=world_state,
                    perception=perception,
                    config=config.skill,
                    logger=logger,
                )
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
