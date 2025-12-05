"""
Mock Environment for testing without LIBERO.
Simulates a simple pick-and-place task.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional


class MockRobotEnv(gym.Env):
    """
    Mock robot environment for testing the training pipeline.
    Simulates a simple pick-and-place task.
    """

    def __init__(
        self,
        max_episode_steps: int = 100,
        action_scale: float = 1.0,
        image_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.image_size = image_size

        # Task description
        self.task_description = "Pick up the red object and place it on the target"

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
            'gripper_state': gym.spaces.Discrete(2),
        })

        self.step_count = 0
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        # Initialize robot state
        self.robot_pos = np.array([0.0, 0.0, 0.5])  # Starting position
        self.robot_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion
        self.gripper_open = True
        self.gripper_qpos = np.array([0.04, 0.04])  # Open gripper

        # Initialize object and target (ensure they're not too close)
        self.object_pos = np.array([
            np.random.uniform(-0.3, 0.0),  # Object on left side
            np.random.uniform(0.2, 0.4),
            0.1,
        ])
        self.target_pos = np.array([
            np.random.uniform(0.0, 0.3),  # Target on right side
            np.random.uniform(0.2, 0.4),
            0.1,
        ])
        self._initial_object_pos = self.object_pos.copy()  # Track initial position

        # Track if object is grasped
        self.object_grasped = False

        self.step_count = 0

        obs = self._get_observation()
        info = {
            'task_description': self.task_description,
            'task_id': 0,
        }

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action."""
        # Scale action
        scaled_action = action * self.action_scale

        # Apply position delta
        pos_delta = scaled_action[:3] * 0.05  # Scale down for safety
        self.robot_pos += pos_delta

        # Clip to workspace bounds
        self.robot_pos = np.clip(self.robot_pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])

        # Handle gripper
        gripper_action = scaled_action[6]
        if gripper_action < -0.5:
            self.gripper_open = True
            self.gripper_qpos = np.array([0.04, 0.04])
        elif gripper_action > 0.5:
            self.gripper_open = False
            self.gripper_qpos = np.array([0.0, 0.0])

        # Check grasp
        dist_to_object = np.linalg.norm(self.robot_pos - self.object_pos)
        if dist_to_object < 0.1 and not self.gripper_open:
            self.object_grasped = True

        # Move object with gripper if grasped
        if self.object_grasped and not self.gripper_open:
            self.object_pos = self.robot_pos.copy()
            self.object_pos[2] = 0.1  # Keep at table level when placed

        # Release object
        if self.object_grasped and self.gripper_open:
            self.object_grasped = False

        self.step_count += 1

        # Compute reward
        reward = self._compute_reward()

        # Check success - object must have been grasped at some point and now placed at target
        dist_to_target = np.linalg.norm(self.object_pos[:2] - self.target_pos[:2])
        # Success requires: object at target AND not currently grasped AND object was moved
        object_moved = np.linalg.norm(self.object_pos[:2] - self._initial_object_pos[:2]) > 0.05
        success = dist_to_target < 0.1 and not self.object_grasped and object_moved

        # Check termination
        done = success
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = {
            'step_count': self.step_count,
            'success': success,
            'dist_to_object': dist_to_object,
            'dist_to_target': dist_to_target,
            'object_grasped': self.object_grasped,
        }

        return obs, reward, done, truncated, info

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        # Create simple image
        image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Draw object (red)
        obj_pixel = self._world_to_pixel(self.object_pos)
        self._draw_circle(image, obj_pixel, 15, [255, 0, 0])

        # Draw target (green)
        target_pixel = self._world_to_pixel(self.target_pos)
        self._draw_circle(image, target_pixel, 15, [0, 255, 0])

        # Draw gripper (gray)
        gripper_pixel = self._world_to_pixel(self.robot_pos)
        self._draw_circle(image, gripper_pixel, 10, [128, 128, 128])

        # Build proprioception
        proprio = np.concatenate([
            self.robot_pos,
            self.robot_orientation,
            self.gripper_qpos,
            np.zeros(6),  # Padding
        ])[:15]

        return {
            'image': image,
            'proprio': proprio.astype(np.float32),
            'gripper_state': 0 if self.gripper_open else 1,
        }

    def _world_to_pixel(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to pixel coordinates."""
        x = int((pos[0] + 0.5) * self.image_size[0])
        y = int((0.5 - pos[1]) * self.image_size[1])
        return (x, y)

    def _draw_circle(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: list,
    ):
        """Draw a filled circle on the image."""
        cx, cy = center
        for x in range(max(0, cx - radius), min(self.image_size[0], cx + radius + 1)):
            for y in range(max(0, cy - radius), min(self.image_size[1], cy + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    image[y, x] = color

    def _compute_reward(self) -> float:
        """Compute reward - sparse with shaping."""
        reward = 0.0

        dist_to_object = np.linalg.norm(self.robot_pos - self.object_pos)
        dist_to_target = np.linalg.norm(self.object_pos[:2] - self.target_pos[:2])

        if not self.object_grasped:
            # Phase 1: Approach object
            # Dense reward for getting close to object
            reward = -dist_to_object
        else:
            # Phase 2: Transport to target
            # Dense reward for getting object close to target
            # Add small bonus for having grasped (but not per-step!)
            reward = -dist_to_target + 0.5  # Bonus offset for grasping

        # Success bonus (large, sparse)
        object_moved = np.linalg.norm(self.object_pos[:2] - self._initial_object_pos[:2]) > 0.05
        if dist_to_target < 0.1 and not self.object_grasped and object_moved:
            reward += 10.0

        return reward

    def render(self) -> np.ndarray:
        """Render current frame."""
        return self._get_observation()['image']

    def close(self):
        """Close environment."""
        pass


def make_mock_env(**kwargs) -> MockRobotEnv:
    """Factory function to create mock environment."""
    return MockRobotEnv(**kwargs)
