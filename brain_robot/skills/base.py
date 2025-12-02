"""Base classes for skills.

Defines the Skill interface and SkillResult structure.
All skills must inherit from Skill and return SkillResult.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, NamedTuple, Any, Optional
import numpy as np

from ..world_model.state import WorldState
from ..config import SkillConfig


class SkillResult(NamedTuple):
    """Structured result from skill execution.
    
    Every skill returns this structure to ensure consistent handling.
    
    Attributes:
        success: Whether the skill completed successfully.
        info: Dictionary with skill-specific details.
        
    Common info keys:
        - "steps_taken": int - Number of env steps used
        - "reached_target": bool - Whether target was reached
        - "final_pose": np.ndarray - Final gripper pose
        - "error_msg": str - Error description if failed
        - "timeout": bool - Whether skill timed out
    """
    success: bool
    info: Dict[str, Any]


class Skill(ABC):
    """Base class for all skills.
    
    Skills are composable motor primitives that:
    1. Check preconditions before execution
    2. Execute control loop with timeout
    3. Update world state on success
    4. Return structured SkillResult
    
    Subclasses must implement:
    - name: Skill identifier
    - preconditions(): Check if skill can execute
    - postconditions(): Check if skill succeeded
    - execute(): Run the skill control loop
    - update_world_state(): Update state after success
    """
    
    name: str = "BaseSkill"
    
    def __init__(self, max_steps: int = 100, config: Optional[SkillConfig] = None):
        """Initialize skill.
        
        Args:
            max_steps: Maximum steps before timeout (REQUIRED).
            config: Optional skill configuration.
        """
        self.max_steps = max_steps
        self.config = config or SkillConfig()
    
    @abstractmethod
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if skill can execute.
        
        Args:
            world_state: Current world state.
            args: Skill arguments (e.g., {"obj": "bowl_1"}).
            
        Returns:
            Tuple of (can_execute: bool, reason: str).
        """
        pass
    
    @abstractmethod
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if skill succeeded.
        
        Args:
            world_state: World state after execution.
            args: Skill arguments.
            
        Returns:
            Tuple of (succeeded: bool, reason: str).
        """
        pass
    
    @abstractmethod
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute the skill.
        
        This is the main control loop. Must respect max_steps timeout.
        
        Args:
            env: The robosuite/LIBERO environment.
            world_state: Current world state.
            args: Skill arguments.
            
        Returns:
            SkillResult with success status and info.
        """
        pass
    
    @abstractmethod
    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after successful execution.
        
        Args:
            world_state: World state to update (modified in place).
            args: Skill arguments.
            result: Result from execute().
        """
        pass
    
    def run(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Full skill execution with pre/post checks.
        
        This is the main entry point for skill execution.
        
        Args:
            env: The robosuite/LIBERO environment.
            world_state: Current world state.
            args: Skill arguments.
            
        Returns:
            SkillResult with full execution info.
        """
        # Check preconditions
        can_execute, reason = self.preconditions(world_state, args)
        if not can_execute:
            return SkillResult(
                success=False,
                info={
                    "error_msg": f"Precondition failed: {reason}",
                    "steps_taken": 0,
                    "precondition_failed": True,
                }
            )
        
        # Execute skill
        result = self.execute(env, world_state, args)
        
        # Update world state if successful
        if result.success:
            self.update_world_state(world_state, args, result)
        
        return result
    
    def _get_gripper_pose_from_obs(self, obs: dict) -> Optional[np.ndarray]:
        """Helper to get gripper pose from observation dictionary."""
        if obs is None:
            return None
        if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
            return np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat']])
        return None

    def _get_gripper_pose(self, env, obs: Optional[dict] = None) -> Optional[np.ndarray]:
        """Helper to get current gripper pose from environment.

        Args:
            env: The environment.
            obs: Optional observation dict. If not provided, will try to get from env.

        Returns:
            7D pose [x, y, z, qw, qx, qy, qz] or None.
        """
        if obs is not None:
            return self._get_gripper_pose_from_obs(obs)

        # Try different methods to get observations
        try:
            if hasattr(env, '_get_observations'):
                obs = env._get_observations()
            elif hasattr(env, 'get_obs'):
                obs = env.get_obs()
            elif hasattr(env, 'observation'):
                obs = env.observation
            elif hasattr(env, '_obs'):
                obs = env._obs
            else:
                # Do a zero-action step to get observation
                action = np.zeros(7)
                result = env.step(action)
                if len(result) >= 1:
                    obs = result[0]
                else:
                    return None

            return self._get_gripper_pose_from_obs(obs)
        except Exception as e:
            return None

    def _step_env(self, env, action: np.ndarray):
        """Helper to step environment with action.

        Returns:
            Tuple of (obs, reward, done, info).
        """
        return env.step(action)
