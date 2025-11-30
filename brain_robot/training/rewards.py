"""
Reward Shaping for RL Training.
Provides dense rewards to guide learning.
"""

import numpy as np
from typing import Dict, Any, Optional


class RewardShaper:
    """
    Computes dense rewards for RL training.

    Rewards:
    - Task success (sparse)
    - Direction following (dense)
    - Speed matching (dense)
    - Gripper consistency (dense)
    - Forward model bonus (dense)
    - Time penalty
    """

    def __init__(
        self,
        task_success_reward: float = 100.0,
        direction_reward_scale: float = 2.0,
        speed_reward_scale: float = 0.5,
        gripper_reward_scale: float = 1.0,
        forward_model_bonus: float = 0.5,
        time_penalty: float = 0.01,
        collision_penalty: float = 1.0,
    ):
        self.task_success_reward = task_success_reward
        self.direction_reward_scale = direction_reward_scale
        self.speed_reward_scale = speed_reward_scale
        self.gripper_reward_scale = gripper_reward_scale
        self.forward_model_bonus = forward_model_bonus
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty

        # Direction vectors for reward computation
        self.direction_vectors = {
            'left': np.array([-1, 0, 0]),
            'right': np.array([1, 0, 0]),
            'forward': np.array([0, 1, 0]),
            'backward': np.array([0, -1, 0]),
            'up': np.array([0, 0, 1]),
            'down': np.array([0, 0, -1]),
        }

        # Speed targets
        self.speed_targets = {
            'very_slow': 0.05,
            'slow': 0.15,
            'medium': 0.3,
            'fast': 0.5,
        }

    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        plan: Dict[str, Any],
        info: Dict[str, Any],
        predicted_proprio: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute total reward.

        Args:
            obs: Current observation
            action: Executed action
            next_obs: Next observation
            plan: Current LLM plan
            info: Environment info
            predicted_proprio: Forward model prediction (optional)

        Returns:
            reward: Total reward
        """
        reward = 0.0

        # 1. Task success (sparse)
        if info.get('success', False):
            reward += self.task_success_reward
            return reward  # Early return on success

        # 2. Direction following
        direction_reward = self._compute_direction_reward(obs, next_obs, plan)
        reward += direction_reward * self.direction_reward_scale

        # 3. Speed matching
        speed_reward = self._compute_speed_reward(action, plan)
        reward += speed_reward * self.speed_reward_scale

        # 4. Gripper consistency
        gripper_reward = self._compute_gripper_reward(action, plan)
        reward += gripper_reward * self.gripper_reward_scale

        # 5. Forward model bonus
        if predicted_proprio is not None:
            fm_reward = self._compute_forward_model_reward(
                next_obs['proprio'], predicted_proprio
            )
            reward += fm_reward * self.forward_model_bonus

        # 6. Time penalty
        reward -= self.time_penalty

        # 7. Collision penalty
        if info.get('collision', False):
            reward -= self.collision_penalty

        return reward

    def _compute_direction_reward(
        self,
        obs: Dict[str, Any],
        next_obs: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> float:
        """Reward for moving in planned direction."""
        movements = plan.get('plan', {}).get('movements', [])
        if not movements:
            return 0.0

        # Get movement delta
        pos_before = obs['proprio'][:3]
        pos_after = next_obs['proprio'][:3]
        delta = pos_after - pos_before

        # Get expected direction
        direction = movements[0].get('direction', 'forward')
        expected_dir = self.direction_vectors.get(direction, np.zeros(3))

        # Compute alignment
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-4 and np.linalg.norm(expected_dir) > 0:
            delta_normalized = delta / delta_norm
            alignment = np.dot(delta_normalized, expected_dir)
            return alignment  # -1 to 1

        return 0.0

    def _compute_speed_reward(
        self,
        action: np.ndarray,
        plan: Dict[str, Any],
    ) -> float:
        """Reward for matching planned speed."""
        movements = plan.get('plan', {}).get('movements', [])
        if not movements:
            return 0.0

        speed = movements[0].get('speed', 'medium')
        target_speed = self.speed_targets.get(speed, 0.3)

        actual_speed = np.linalg.norm(action[:3])
        speed_error = abs(actual_speed - target_speed)

        return -speed_error  # Negative error as reward

    def _compute_gripper_reward(
        self,
        action: np.ndarray,
        plan: Dict[str, Any],
    ) -> float:
        """Reward for correct gripper action."""
        gripper_plan = plan.get('plan', {}).get('gripper', 'maintain')
        gripper_action = action[6]

        if gripper_plan == 'open' and gripper_action < 0:
            return 1.0
        elif gripper_plan == 'close' and gripper_action > 0:
            return 1.0
        elif gripper_plan == 'maintain':
            return 0.1  # Small reward for stability
        else:
            return -0.5  # Penalty for wrong action

    def _compute_forward_model_reward(
        self,
        actual_proprio: np.ndarray,
        predicted_proprio: np.ndarray,
    ) -> float:
        """Reward for accurate forward model predictions."""
        error = np.mean((actual_proprio - predicted_proprio) ** 2)
        # Convert to reward (lower error = higher reward)
        return np.exp(-error * 10)  # 0 to 1
