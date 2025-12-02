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

from brain_robot.config import RunConfig, SkillConfig, PerceptionConfig, LoggingConfig
from brain_robot.perception.oracle import OraclePerception
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
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name
    
    return env, task.language


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
    
    # Determine source and target from objects
    # For Phase 1, we use simple heuristics
    if len(perc_result.object_names) < 2:
        return False
    
    source_obj = perc_result.object_names[0]
    target_obj = perc_result.object_names[1]
    
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
