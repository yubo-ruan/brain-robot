"""Phase 1 evaluation script.

Runs pick-and-place with oracle perception and hardcoded skill sequence.
Goal: Achieve >80% success rate over 20 episodes.
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
    """Simple task parsing to find source and target objects.

    This is a minimal Phase 1 heuristic. Phase 3 will use Qwen for this.

    Args:
        task_description: Natural language task (e.g., "pick up the black bowl and place it on the plate")
        object_names: List of object IDs from perception

    Returns:
        (source_obj, target_obj) tuple
    """
    task_lower = task_description.lower()

    # Build a mapping from object types to object IDs
    object_types = {}
    for obj_id in object_names:
        obj_lower = obj_id.lower()
        # Extract type keywords
        if 'bowl' in obj_lower:
            object_types.setdefault('bowl', []).append(obj_id)
        if 'plate' in obj_lower and 'burner' not in obj_lower:  # Exclude stove burner plate
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

    # Parse task for source: "pick up the X" pattern
    source_keywords = ['pick up the', 'pick the', 'grab the', 'take the']
    for kw in source_keywords:
        if kw in task_lower:
            # Find what comes after
            rest = task_lower.split(kw)[1]
            for obj_type, obj_list in object_types.items():
                if obj_type in rest.split()[0:3]:  # Check first 3 words
                    source_obj = obj_list[0]  # Take first matching object
                    break
            if source_obj:
                break

    # Parse task for target: "place it on/in the X" pattern
    target_keywords = ['place it on the', 'place it in the', 'put it on the', 'put it in the',
                       'place on the', 'place in the', 'on the', 'in the', 'into the']
    for kw in target_keywords:
        if kw in task_lower:
            rest = task_lower.split(kw)[-1]  # Take last occurrence
            for obj_type, obj_list in object_types.items():
                if obj_type in rest.split()[0:3]:
                    target_obj = obj_list[0]
                    break
            if target_obj:
                break

    # Fallback: if no target found but we have source, look for different object type
    if source_obj and not target_obj:
        source_type = None
        for obj_type, obj_list in object_types.items():
            if source_obj in obj_list:
                source_type = obj_type
                break
        # Pick first object of different type
        for obj_type, obj_list in object_types.items():
            if obj_type != source_type:
                target_obj = obj_list[0]
                break

    # Final fallback: first two objects
    if not source_obj and len(object_names) >= 1:
        source_obj = object_names[0]
    if not target_obj and len(object_names) >= 2:
        target_obj = object_names[1]

    return source_obj, target_obj


def run_episode(
    env,
    task_description: str,
    world_state: WorldState,
    perception: OraclePerception,
    config: SkillConfig,
    logger: EpisodeLogger,
) -> bool:
    """Run a single episode with hardcoded skill sequence.

    Returns:
        True if episode succeeded.
    """
    # Get initial perception
    perc_result = perception.perceive(env)
    world_state.update_from_perception(perc_result)
    logger.log_world_state(world_state)

    # Determine source and target from task description
    if len(perc_result.object_names) < 2:
        return False

    source_obj, target_obj = parse_task_for_grounding(task_description, perc_result.object_names)
    
    print(f"  Source: {source_obj}")
    print(f"  Target: {target_obj}")
    
    # Define skill sequence
    skills = [
        (ApproachSkill(config=config), {"obj": source_obj}),
        (GraspSkill(config=config), {"obj": source_obj}),
        (MoveSkill(config=config), {"obj": source_obj, "region": target_obj}),
        (PlaceSkill(config=config), {"obj": source_obj, "region": target_obj}),
    ]
    
    step_count = 0
    
    for skill, args in skills:
        # Update perception before skill
        with logger.get_timer().measure("perception"):
            perc_result = perception.perceive(env)
            world_state.update_from_perception(perc_result)
        
        # Execute skill
        skill_timer_name = f"skill_{skill.name}"
        with logger.get_timer().measure(skill_timer_name):
            result = skill.run(env, world_state, args)
        
        # Log skill execution
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
    parser = argparse.ArgumentParser(description="Phase 1 Evaluation")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/phase1")
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
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print("=" * 60)
    print("Phase 1 Evaluation")
    print("=" * 60)
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
    run_summary = RunSummary(str(output_dir), config=config)
    
    # Run episodes
    successes = 0
    
    for episode_idx in range(args.n_episodes):
        episode_seed = get_episode_seed(args.seed, episode_idx)
        set_global_seed(episode_seed, env)
        
        print(f"\nEpisode {episode_idx + 1}/{args.n_episodes} (seed={episode_seed})")
        
        # Reset environment
        env.reset()
        
        # Create fresh world state
        world_state = WorldState()
        
        # Start logging
        logger.start_episode(
            task=task_description,
            task_id=args.task_id,
            episode_idx=episode_idx,
            seed=episode_seed,
        )
        
        # Run episode
        try:
            success = run_episode(
                env=env,
                task_description=task_description,
                world_state=world_state,
                perception=perception,
                config=config.skill,
                logger=logger,
            )
        except Exception as e:
            print(f"  Exception: {e}")
            success = False
        
        # End logging
        logger.end_episode(
            success=success,
            failure_reason=None if success else "Skill execution failed",
        )
        
        if success:
            successes += 1
            print(f"  Result: SUCCESS")
        else:
            print(f"  Result: FAILURE")
        
        # Update run summary
        if logger.current_episode is None:
            # Episode was ended, get from saved file
            pass
    
    # Final summary
    success_rate = successes / args.n_episodes
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {args.n_episodes}")
    print(f"Successful: {successes}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Target: {config.success_rate_threshold:.1%}")
    
    if success_rate >= config.success_rate_threshold:
        print("\n✓ PHASE 1 SUCCESS CRITERIA MET")
    else:
        print(f"\n✗ Below target ({success_rate:.1%} < {config.success_rate_threshold:.1%})")
    
    # Save final summary
    summary = {
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
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    env.close()
    
    return 0 if success_rate >= config.success_rate_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
