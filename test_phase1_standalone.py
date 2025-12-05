#!/usr/bin/env python3
"""
Standalone Phase 1 Test - avoids robosuite import issues.
"""

import sys
import os

# Add path but don't trigger brain_robot package imports
sys.path.insert(0, '/workspace/brain_robot')

# Import only the HDF5 logger module directly (bypassing __init__.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hdf5_episode_logger",
    "/workspace/brain_robot/brain_robot/logging/hdf5_episode_logger.py"
)
hdf5_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hdf5_module)

MOKAOutput = hdf5_module.MOKAOutput
Timestep = hdf5_module.Timestep
HDF5EpisodeLogger = hdf5_module.HDF5EpisodeLogger
load_hdf5_episode = hdf5_module.load_hdf5_episode
get_moka_outputs = hdf5_module.get_moka_outputs

import numpy as np
import tempfile


def test_moka_output_serialization():
    """Test MOKAOutput serialization roundtrip."""
    print("Testing MOKAOutput serialization...")

    moka = MOKAOutput(
        grasp_kp=(0.3, 0.4),
        function_kp=(0.35, 0.45),
        target_kp=(0.6, 0.7),
        pre_tile='b2',
        target_tile='c3',
        post_tile='d4',
        pre_height='same',
        post_height='above',
        target_angle='downward',
        conf_grasp=0.95,
        conf_target=0.88,
    )

    # Convert to array and back
    arr = moka.to_array()
    assert arr.shape == (16,), f"Expected shape (16,), got {arr.shape}"

    reconstructed = MOKAOutput.from_array(arr)

    # Check keypoints
    assert reconstructed.grasp_kp is not None
    assert abs(reconstructed.grasp_kp[0] - 0.3) < 0.001
    assert abs(reconstructed.grasp_kp[1] - 0.4) < 0.001

    assert reconstructed.target_kp is not None
    assert abs(reconstructed.target_kp[0] - 0.6) < 0.001

    # Check tiles
    assert reconstructed.pre_tile == 'b2', f"Expected 'b2', got {reconstructed.pre_tile}"
    assert reconstructed.target_tile == 'c3'
    assert reconstructed.post_tile == 'd4'

    # Check heights
    assert reconstructed.pre_height == 'same'
    assert reconstructed.post_height == 'above'

    # Check angle
    assert reconstructed.target_angle == 'downward'

    print("  PASS: MOKAOutput serialization works correctly")


def test_moka_output_none_values():
    """Test MOKAOutput with None values."""
    print("Testing MOKAOutput with None values...")

    moka = MOKAOutput()  # All defaults

    arr = moka.to_array()
    assert arr[0] == -1  # grasp_kp should be -1 when None

    reconstructed = MOKAOutput.from_array(arr)
    assert reconstructed.grasp_kp is None
    assert reconstructed.target_kp is None

    print("  PASS: MOKAOutput handles None values correctly")


def test_hdf5_episode_logging():
    """Test full episode logging cycle."""
    print("Testing HDF5 episode logging...")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = HDF5EpisodeLogger(log_dir=tmpdir, image_size=(64, 64))

        # Create test episode
        with logger.start_episode(task='test_task', skill_id=1, metadata={'test': True}) as ep:
            for i in range(10):
                # Create fake observations
                rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                depth = np.random.rand(64, 64).astype(np.float32)

                moka = MOKAOutput(
                    grasp_kp=(0.3 + i*0.01, 0.4),
                    target_kp=(0.6, 0.7),
                    target_tile='c3',
                )

                timestep = Timestep(
                    rgb=rgb,
                    depth=depth,
                    ee_pos=np.array([0.1, 0.2, 0.3 + i*0.01], dtype=np.float32),
                    ee_quat=np.array([0, 0, 0, 1], dtype=np.float32),
                    gripper_state=0.5,
                    moka_output=moka,
                    action=np.zeros(10, dtype=np.float32),
                    skill_id=1,
                    phase_id=i // 3,
                    timestamp=i * 0.05,
                )
                ep.add_timestep(timestep)

            ep.mark_success(True)

        # Load and verify
        episodes = logger.list_episodes()
        assert len(episodes) == 1, f"Expected 1 episode, got {len(episodes)}"

        data = load_hdf5_episode(str(episodes[0]))

        assert data['task'] == 'test_task'
        assert data['skill_id'] == 1
        assert data['success'] == True
        assert data['num_timesteps'] == 10

        # Check images
        assert data['rgb'].shape == (10, 64, 64, 3), f"RGB shape: {data['rgb'].shape}"
        assert data['depth'].shape == (10, 64, 64), f"Depth shape: {data['depth'].shape}"

        # Check proprioception
        assert data['ee_pos'].shape == (10, 3)
        assert abs(data['ee_pos'][5, 2] - 0.35) < 0.001

        # Check MOKA
        assert data['moka'].shape == (10, 16)
        moka_outputs = get_moka_outputs(data)
        assert len(moka_outputs) == 10
        assert moka_outputs[0].grasp_kp is not None

        print("  PASS: HDF5 episode logging works correctly")


def test_empty_episode():
    """Test that empty episodes don't crash."""
    print("Testing empty episode handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = HDF5EpisodeLogger(log_dir=tmpdir)

        with logger.start_episode(task='empty') as ep:
            pass  # No timesteps added

        # Should not create a file for empty episode
        episodes = logger.list_episodes()
        assert len(episodes) == 0, f"Expected 0 episodes, got {len(episodes)}"

    print("  PASS: Empty episodes handled correctly")


def main():
    print("=" * 60)
    print("Phase 1: Logging & Replay Infrastructure Tests")
    print("=" * 60)

    test_moka_output_serialization()
    test_moka_output_none_values()
    test_hdf5_episode_logging()
    test_empty_episode()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
