"""
Phase 1 Tests: Logging & Replay Infrastructure

Tests the HDF5 episode logger and replay viewer.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import pytest

# Import directly to avoid robosuite dependency chain
from brain_robot.logging.hdf5_episode_logger import (
    MOKAOutput,
    Timestep,
    HDF5EpisodeLogger,
    load_hdf5_episode,
    get_moka_outputs,
)


class TestMOKAOutput:
    """Tests for MOKAOutput dataclass."""

    def test_to_array_and_back(self):
        """Test MOKAOutput serialization roundtrip."""
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
        assert arr.shape == (16,)

        reconstructed = MOKAOutput.from_array(arr)

        # Check keypoints
        assert reconstructed.grasp_kp is not None
        assert abs(reconstructed.grasp_kp[0] - 0.3) < 0.001
        assert abs(reconstructed.grasp_kp[1] - 0.4) < 0.001

        assert reconstructed.target_kp is not None
        assert abs(reconstructed.target_kp[0] - 0.6) < 0.001

        # Check tiles
        assert reconstructed.pre_tile == 'b2'
        assert reconstructed.target_tile == 'c3'
        assert reconstructed.post_tile == 'd4'

        # Check heights
        assert reconstructed.pre_height == 'same'
        assert reconstructed.post_height == 'above'

        # Check angle
        assert reconstructed.target_angle == 'downward'

        # Check confidence
        assert abs(reconstructed.conf_grasp - 0.95) < 0.001

    def test_none_values(self):
        """Test MOKAOutput with None values."""
        moka = MOKAOutput()  # All defaults

        arr = moka.to_array()
        assert arr[0] == -1  # grasp_kp should be -1 when None

        reconstructed = MOKAOutput.from_array(arr)
        assert reconstructed.grasp_kp is None
        assert reconstructed.target_kp is None


class TestHDF5EpisodeLogger:
    """Tests for HDF5 episode logging."""

    def test_log_and_load_episode(self):
        """Test full episode logging cycle."""
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
                        ee_pos=np.array([0.1, 0.2, 0.3 + i*0.01]),
                        ee_quat=np.array([0, 0, 0, 1]),
                        gripper_state=0.5,
                        moka_output=moka,
                        action=np.zeros(10),
                        skill_id=1,
                        phase_id=i // 3,
                        timestamp=i * 0.05,
                    )
                    ep.add_timestep(timestep)

                ep.mark_success(True)

            # Load and verify
            episodes = logger.list_episodes()
            assert len(episodes) == 1

            data = load_hdf5_episode(str(episodes[0]))

            assert data['task'] == 'test_task'
            assert data['skill_id'] == 1
            assert data['success'] == True
            assert data['num_timesteps'] == 10

            # Check images
            assert data['rgb'].shape == (10, 64, 64, 3)
            assert data['depth'].shape == (10, 64, 64)

            # Check proprioception
            assert data['ee_pos'].shape == (10, 3)
            assert abs(data['ee_pos'][5, 2] - 0.35) < 0.001

            # Check MOKA
            assert data['moka'].shape == (10, 16)
            moka_outputs = get_moka_outputs(data)
            assert len(moka_outputs) == 10
            assert moka_outputs[0].grasp_kp is not None

    def test_empty_episode(self):
        """Test that empty episodes don't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HDF5EpisodeLogger(log_dir=tmpdir)

            with logger.start_episode(task='empty') as ep:
                pass  # No timesteps added

            # Should not create a file for empty episode
            episodes = logger.list_episodes()
            assert len(episodes) == 0


class TestTimestep:
    """Tests for Timestep dataclass."""

    def test_default_values(self):
        """Test Timestep default values."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        ts = Timestep(rgb=rgb)

        assert ts.depth is None
        assert ts.ee_pos.shape == (3,)
        assert ts.ee_quat.shape == (4,)
        assert ts.gripper_state == 0.0
        assert ts.skill_id == -1
        assert ts.phase_id == -1
        assert ts.is_terminal == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
