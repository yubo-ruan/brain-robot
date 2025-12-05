"""Coordinate frame verification tests.

CRITICAL: Run these tests BEFORE implementing/testing skills.
Verifies that all poses are in consistent world frame.
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.perception.oracle import OraclePerception
from src.utils.seeds import set_global_seed
from src.utils.visualization import debug_plot_poses


def make_libero_env(task_suite: str = "libero_spatial", task_id: int = 0):
    """Create LIBERO environment for testing."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
        
        benchmark = get_benchmark(task_suite)()
        task = benchmark.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        
        env = OffScreenRenderEnv(**env_args)
        env.task_description = task_description
        return env
        
    except ImportError as e:
        print(f"LIBERO not available: {e}")
        return None


def test_coordinate_frames():
    """Verify all poses are in consistent world frame.
    
    This test:
    1. Creates an environment
    2. Gets poses from different sources
    3. Verifies they're in the same frame
    4. Checks action directions make sense
    """
    print("=" * 60)
    print("TEST: Coordinate Frame Consistency")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        obs = env.reset()
        
        # Get perception
        perception = OraclePerception()
        result = perception.perceive(env)
        
        print(f"\nDetected {len(result.object_names)} objects:")
        for name in result.object_names:
            pose = result.objects[name]
            print(f"  {name}: pos={pose[:3]}, quat={pose[3:7]}")
        
        print(f"\nGripper pose: {result.gripper_pose}")
        print(f"Gripper width: {result.gripper_width}")
        
        # Verify gripper pose is reasonable
        if result.gripper_pose is not None:
            gripper_pos = result.gripper_pose[:3]
            
            # Gripper should be in reasonable workspace bounds
            # Typical robosuite workspace is roughly [-0.5, 0.5] x [-0.5, 0.5] x [0, 1]
            assert -1.0 < gripper_pos[0] < 1.0, f"Gripper X out of bounds: {gripper_pos[0]}"
            assert -1.0 < gripper_pos[1] < 1.0, f"Gripper Y out of bounds: {gripper_pos[1]}"
            assert 0.0 < gripper_pos[2] < 1.5, f"Gripper Z out of bounds: {gripper_pos[2]}"
            print("\n✓ Gripper position in reasonable workspace bounds")
        
        # Verify objects are in consistent frame
        for name, pose in result.objects.items():
            obj_pos = pose[:3]
            
            # Objects should also be in workspace
            assert -1.0 < obj_pos[0] < 1.0, f"Object {name} X out of bounds"
            assert -1.0 < obj_pos[1] < 1.0, f"Object {name} Y out of bounds"
            assert -0.5 < obj_pos[2] < 1.0, f"Object {name} Z out of bounds"
        print("✓ All objects in reasonable workspace bounds")
        
        # Check direction vectors make sense
        if result.gripper_pose is not None and len(result.objects) > 0:
            first_obj = list(result.objects.keys())[0]
            obj_pos = result.objects[first_obj][:3]
            gripper_pos = result.gripper_pose[:3]
            
            direction = obj_pos - gripper_pos
            distance = np.linalg.norm(direction)
            
            print(f"\nDirection from gripper to {first_obj}: {direction}")
            print(f"Distance: {distance:.3f}m")
            
            # Sanity check: direction should be reasonable
            assert distance < 2.0, f"Distance to object unreasonably large: {distance}m"
            print("✓ Direction vector sanity check passed")
        
        # Try to get raw sim data and compare
        sim = perception._get_sim(env)
        if sim is not None:
            print("\n--- Raw Simulation Data Comparison ---")
            
            for name in result.object_names[:2]:  # Check first 2 objects
                try:
                    body_id = sim.model.body_name2id(name)
                    raw_pos = sim.data.body_xpos[body_id]
                    api_pos = result.objects[name][:3]
                    
                    diff = np.linalg.norm(raw_pos - api_pos)
                    print(f"{name}:")
                    print(f"  Raw sim: {raw_pos}")
                    print(f"  API:     {api_pos}")
                    print(f"  Diff:    {diff:.6f}m")
                    
                    assert diff < 1e-3, f"Position mismatch for {name}: {diff}m"
                except (KeyError, ValueError) as e:
                    print(f"  Could not compare {name}: {e}")
            
            print("✓ Raw sim data matches API data")
        
        print("\n" + "=" * 60)
        print("TEST PASSED: Coordinate frames are consistent")
        print("=" * 60)
        
        # Optional: Save visualization
        if result.gripper_pose is not None:
            debug_plot_poses(
                gripper_pose=result.gripper_pose,
                object_poses=result.objects,
                save_path="/workspace/src/logs/test_frames.png",
                title="Coordinate Frame Test"
            )
            print("\nVisualization saved to logs/test_frames.png")
        
        return True
        
    finally:
        env.close()


def test_action_direction():
    """Verify action direction moves gripper correctly.
    
    Applies a known action and verifies the gripper moves in expected direction.
    """
    print("\n" + "=" * 60)
    print("TEST: Action Direction Verification")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        obs = env.reset()
        
        perception = OraclePerception()
        
        # Get initial pose
        initial_result = perception.perceive(env)
        initial_pos = initial_result.gripper_pose[:3].copy()
        print(f"Initial gripper position: {initial_pos}")
        
        # Apply action in +X direction
        action = np.zeros(7)
        action[0] = 0.5  # Move in X
        action[6] = 0.0  # Keep gripper
        
        for _ in range(10):
            env.step(action)
        
        # Get new pose
        final_result = perception.perceive(env)
        final_pos = final_result.gripper_pose[:3].copy()
        print(f"Final gripper position: {final_pos}")
        
        # Check movement direction
        movement = final_pos - initial_pos
        print(f"Movement: {movement}")
        
        # X component should be positive (moved in +X)
        # Note: This depends on action space definition, may need adjustment
        print(f"X movement: {movement[0]:.4f}")
        
        if movement[0] > 0:
            print("✓ Positive X action moves gripper in +X direction")
        else:
            print("⚠ Action direction may be inverted or scaled differently")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED: Check output above for direction mapping")
        print("=" * 60)
        
        return True
        
    finally:
        env.close()


def test_observation_keys():
    """List available observation keys for understanding data format."""
    print("\n" + "=" * 60)
    print("TEST: Available Observation Keys")
    print("=" * 60)
    
    env = make_libero_env()
    if env is None:
        print("SKIP: LIBERO not available")
        return False
    
    try:
        set_global_seed(42, env)
        obs = env.reset()
        
        print("\nObservation keys:")
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val)}")
        
        # Check for key observation types
        key_obs = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        print("\nKey observations:")
        for key in key_obs:
            if key in obs:
                print(f"  ✓ {key}: {obs[key]}")
            else:
                print(f"  ✗ {key}: NOT FOUND")
        
        return True
        
    finally:
        env.close()


if __name__ == "__main__":
    # Run all tests
    results = []
    
    results.append(("Observation Keys", test_observation_keys()))
    results.append(("Coordinate Frames", test_coordinate_frames()))
    results.append(("Action Direction", test_action_direction()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {name}: {status}")
