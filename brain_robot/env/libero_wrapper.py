"""
LIBERO Environment Wrapper.
Provides gym-style interface for RL training.

Supports both state-based and image-based observations for
testing different policy architectures.
"""

import os
# Set rendering backend before importing mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from PIL import Image


class LIBEROEnvWrapper(gym.Env):
    """
    Wrapper for LIBERO environment with standard gym interface.

    Supports:
    - State-based observations for MLP policies
    - Image-based observations for vision policies
    - Both modalities for multi-modal policies
    """

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        task_id: int = 0,
        max_episode_steps: int = 300,
        action_scale: float = 1.0,
        image_size: Tuple[int, int] = (128, 128),
        obs_mode: str = "state",  # "state", "image", or "both"
    ):
        super().__init__()

        self.task_suite = task_suite
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.image_size = image_size
        self.obs_mode = obs_mode
        self._env = None

        # Import LIBERO
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        # Get benchmark and task
        self.benchmark = get_benchmark(task_suite)()
        self.task = self.benchmark.get_task(task_id)
        self.task_description = self.task.language

        # Get BDDL file path
        bddl_file = self.benchmark.get_task_bddl_file_path(task_id)

        # Create environment
        env_args = {
            'bddl_file_name': bddl_file,
            'camera_heights': image_size[0],
            'camera_widths': image_size[1],
        }
        self._env = OffScreenRenderEnv(**env_args)

        # Get initial states for deterministic resets
        try:
            import torch
            # Handle torch.load with weights_only=False for older checkpoints
            init_states_path = os.path.join(
                self._get_init_states_folder(),
                self.task.problem_folder,
                self.task.init_states_file,
            )
            self._init_states = torch.load(init_states_path, weights_only=False)
        except Exception as e:
            print(f"[Warning] Could not load init states: {e}")
            print("[Warning] Will use random initialization")
            self._init_states = None
        self._init_state_idx = 0

        # Define spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0, high=255, shape=(*image_size, 3), dtype=np.uint8
            ),
            'proprio': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            ),
            'gripper_state': gym.spaces.Discrete(2),  # 0=open, 1=closed
        })

        self.step_count = 0

    def _get_init_states_folder(self):
        """Get path to init states folder."""
        from libero.libero import get_libero_path
        try:
            return get_libero_path("init_states")
        except:
            # Fallback to default path
            import libero.libero
            libero_path = os.path.dirname(libero.libero.__file__)
            return os.path.join(libero_path, "init_files")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset with initial state if available
        self._env.reset()
        if self._init_states is not None and len(self._init_states) > 0:
            init_state = self._init_states[self._init_state_idx % len(self._init_states)]
            self._init_state_idx += 1
            obs = self._env.set_init_state(init_state)
            # Run a few dummy steps for physics to settle
            dummy_action = np.zeros(7)
            for _ in range(5):
                obs, _, _, _ = self._env.step(dummy_action)
        else:
            obs = self._env.reset()
        self.step_count = 0

        processed_obs = self._process_observation(obs)

        info = {
            'task_description': self.task_description,
            'task_id': self.task_id,
        }

        return processed_obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action."""
        # Scale action
        scaled_action = action * self.action_scale

        # Step environment
        obs, reward, done, info = self._env.step(scaled_action)

        self.step_count += 1

        # Check truncation
        truncated = self.step_count >= self.max_episode_steps

        # Process observation
        processed_obs = self._process_observation(obs)

        # Add info
        info['step_count'] = self.step_count
        info['success'] = info.get('success', False)

        return processed_obs, reward, done, truncated, info

    def _process_observation(self, obs: Dict) -> Dict[str, Any]:
        """Process raw observation into standard format."""
        # Get image
        if 'agentview_image' in obs:
            image = obs['agentview_image']
        elif 'image' in obs:
            image = obs['image']
        else:
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Resize if needed
        if image.shape[:2] != self.image_size:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(self.image_size)
            image = np.array(pil_image)

        # Get proprioception
        proprio = self._extract_proprio(obs)

        # Get gripper state
        gripper_qpos = obs.get('robot0_gripper_qpos', [0.04])
        gripper_state = 0 if gripper_qpos[0] > 0.04 else 1

        return {
            'image': image,
            'proprio': proprio,
            'gripper_state': gripper_state,
        }

    def _extract_proprio(self, obs: Dict) -> np.ndarray:
        """Extract proprioception from observation."""
        proprio_parts = []

        # End-effector position (3)
        if 'robot0_eef_pos' in obs:
            proprio_parts.append(obs['robot0_eef_pos'])

        # End-effector orientation (4 quaternion or 3 euler)
        if 'robot0_eef_quat' in obs:
            proprio_parts.append(obs['robot0_eef_quat'])
        elif 'robot0_eef_euler' in obs:
            proprio_parts.append(obs['robot0_eef_euler'])

        # Gripper position (2)
        if 'robot0_gripper_qpos' in obs:
            proprio_parts.append(obs['robot0_gripper_qpos'])

        # Joint positions (optional)
        if 'robot0_joint_pos' in obs:
            proprio_parts.append(obs['robot0_joint_pos'][:6])  # First 6 joints

        if proprio_parts:
            proprio = np.concatenate(proprio_parts)
        else:
            proprio = np.zeros(15, dtype=np.float32)

        # Pad or truncate to fixed size
        if len(proprio) < 15:
            proprio = np.pad(proprio, (0, 15 - len(proprio)))
        elif len(proprio) > 15:
            proprio = proprio[:15]

        return proprio.astype(np.float32)

    def render(self) -> np.ndarray:
        """Render current frame."""
        obs = self._env._get_observations()
        if 'agentview_image' in obs:
            return obs['agentview_image']
        return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get full state for state-based policies."""
        obs = self._env._get_observations()
        return {
            'eef_pos': obs.get('robot0_eef_pos', np.zeros(3)),
            'eef_quat': obs.get('robot0_eef_quat', np.zeros(4)),
            'gripper_qpos': obs.get('robot0_gripper_qpos', np.zeros(2)),
            'joint_pos': obs.get('robot0_joint_pos', np.zeros(7)),
            'object_state': obs.get('object-state', np.zeros(10)),
        }

    def close(self):
        """Close environment."""
        if self._env is not None:
            self._env.close()

    @property
    def unwrapped(self):
        return self._env

    @property
    def env(self):
        """Backward compatibility."""
        return self._env


def make_libero_env(
    task_suite: str = "libero_spatial",
    task_id: int = 0,
    **kwargs,
) -> LIBEROEnvWrapper:
    """Factory function to create LIBERO environment."""
    return LIBEROEnvWrapper(
        task_suite=task_suite,
        task_id=task_id,
        **kwargs,
    )
