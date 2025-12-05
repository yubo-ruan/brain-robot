"""Per-skill unit tests.

Tests each skill under oracle perception conditions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.perception.oracle import OraclePerception
from src.world_model.state import WorldState
from src.skills.approach import ApproachSkill
from src.skills.grasp import GraspSkill
from src.skills.move import MoveSkill
from src.skills.place import PlaceSkill
from src.utils.seeds import set_global_seed
from src.config import SkillConfig


def make_libero_env(task_suite: str = "libero_spatial", task_id: int = 0):
    """Create LIBERO environment for testing."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
        
        benchmark = get_benchmark(task_suite)()
        task = benchmark.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        
        env = OffScreenRenderEnv(**env_args)
        return env
        
    except ImportError as e:
        print(f"LIBERO not available: {e}")
        return None


def test_approach_success():
    """Test ApproachSkill reaches target under oracle perception."""
    print("\n" + "=" * 60)
    print("TEST: ApproachSkill Success")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        env.reset()
        
        # Setup perception and world state
        perception = OraclePerception()
        perc_result = perception.perceive(env)
        
        world_state = WorldState()
        world_state.update_from_perception(perc_result)
        
        # Pick first object
        if not perc_result.object_names:
            print("SKIP: No objects detected")
            return False
        
        obj_name = perc_result.object_names[0]
        print(f"Target object: {obj_name}")
        print(f"Object position: {perc_result.objects[obj_name][:3]}")
        
        # Run approach skill with default config (tuned gains)
        config = SkillConfig()
        skill = ApproachSkill(config=config)
        
        result = skill.run(env, world_state, {"obj": obj_name})
        
        print(f"\nResult: success={result.success}")
        print(f"Steps taken: {result.info.get('steps_taken', 'N/A')}")
        print(f"Final error: {result.info.get('final_error', 'N/A')}")
        
        if result.success:
            print("✓ Approach skill succeeded")
            
            # Verify actual position
            final_perc = perception.perceive(env)
            obj_pos = perc_result.objects[obj_name][:3]
            gripper_pos = final_perc.gripper_pose[:3]
            expected_z = obj_pos[2] + config.approach_pregrasp_height
            
            xy_dist = np.linalg.norm(gripper_pos[:2] - obj_pos[:2])
            z_error = abs(gripper_pos[2] - expected_z)
            
            print(f"XY distance to object: {xy_dist:.3f}m")
            print(f"Z error from pregrasp: {z_error:.3f}m")
            
            return xy_dist < 0.05 and z_error < 0.05
        else:
            print(f"✗ Approach failed: {result.info.get('error_msg', 'Unknown')}")
            return False
            
    finally:
        env.close()


def test_approach_timeout():
    """Test ApproachSkill times out gracefully with very short timeout."""
    print("\n" + "=" * 60)
    print("TEST: ApproachSkill Timeout")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        env.reset()
        
        perception = OraclePerception()
        perc_result = perception.perceive(env)
        
        world_state = WorldState()
        world_state.update_from_perception(perc_result)
        
        if not perc_result.object_names:
            print("SKIP: No objects detected")
            return False
        
        obj_name = perc_result.object_names[0]
        
        # Very short timeout
        skill = ApproachSkill(max_steps=5)
        result = skill.run(env, world_state, {"obj": obj_name})
        
        print(f"Result: success={result.success}")
        print(f"Timeout: {result.info.get('timeout', False)}")
        
        # Should fail due to timeout
        if not result.success and result.info.get('timeout', False):
            print("✓ Skill correctly timed out")
            return True
        else:
            print("✗ Expected timeout failure")
            return False
            
    finally:
        env.close()


def test_approach_precondition_fail():
    """Test ApproachSkill fails precondition for nonexistent object."""
    print("\n" + "=" * 60)
    print("TEST: ApproachSkill Precondition Failure")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        env.reset()
        
        perception = OraclePerception()
        perc_result = perception.perceive(env)
        
        world_state = WorldState()
        world_state.update_from_perception(perc_result)
        
        # Try to approach nonexistent object
        skill = ApproachSkill(max_steps=100)
        result = skill.run(env, world_state, {"obj": "nonexistent_object_xyz"})
        
        print(f"Result: success={result.success}")
        print(f"Precondition failed: {result.info.get('precondition_failed', False)}")
        
        if not result.success and result.info.get('precondition_failed', False):
            print("✓ Precondition correctly rejected nonexistent object")
            return True
        else:
            print("✗ Expected precondition failure")
            return False
            
    finally:
        env.close()


def test_full_pick_and_place():
    """Test complete pick-and-place sequence with hardcoded skills."""
    print("\n" + "=" * 60)
    print("TEST: Full Pick and Place Sequence")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        env.reset()
        
        perception = OraclePerception()
        perc_result = perception.perceive(env)
        
        world_state = WorldState()
        world_state.update_from_perception(perc_result)
        
        if len(perc_result.object_names) < 2:
            print("SKIP: Need at least 2 objects for pick and place")
            return False
        
        source_obj = perc_result.object_names[0]
        target_obj = perc_result.object_names[1]
        
        print(f"Source: {source_obj}")
        print(f"Target: {target_obj}")
        
        config = SkillConfig()
        
        # Skill sequence
        skills = [
            (ApproachSkill(config=config), {"obj": source_obj}),
            (GraspSkill(config=config), {"obj": source_obj}),
            (MoveSkill(config=config), {"obj": source_obj, "region": target_obj}),
            (PlaceSkill(config=config), {"obj": source_obj, "region": target_obj}),
        ]
        
        all_success = True
        for skill, args in skills:
            print(f"\nExecuting: {skill.name}")
            
            # Update perception before each skill
            perc_result = perception.perceive(env)
            world_state.update_from_perception(perc_result)
            
            result = skill.run(env, world_state, args)
            
            print(f"  Success: {result.success}")
            print(f"  Steps: {result.info.get('steps_taken', 'N/A')}")
            
            if not result.success:
                print(f"  Error: {result.info.get('error_msg', 'Unknown')}")
                all_success = False
                break
        
        if all_success:
            print("\n✓ Full pick-and-place sequence completed")
            print(f"Final world state:")
            print(f"  Holding: {world_state.holding}")
            print(f"  On: {world_state.on}")
            print(f"  Inside: {world_state.inside}")
        else:
            print("\n✗ Pick-and-place sequence failed")
        
        return all_success
        
    finally:
        env.close()


if __name__ == "__main__":
    results = []
    
    results.append(("Approach Success", test_approach_success()))
    results.append(("Approach Timeout", test_approach_timeout()))
    results.append(("Approach Precondition", test_approach_precondition_fail()))
    results.append(("Full Pick and Place", test_full_pick_and_place()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "PASS" if p else "FAIL/SKIP"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
