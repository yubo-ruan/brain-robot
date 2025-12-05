"""ApproachObject skill.

Moves gripper to pre-grasp pose above target object.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController, compute_pregrasp_pose
from ..config import SkillConfig


class ApproachSkill(Skill):
    """Move gripper to pre-grasp pose above object.
    
    Preconditions:
    - Object exists in world state
    - Gripper is not holding anything
    
    Postconditions:
    - Gripper is within threshold of pre-grasp pose
    """
    
    name = "ApproachObject"
    
    def __init__(
        self,
        max_steps: int = 100,
        pos_threshold: float = 0.03,
        pregrasp_height: float = 0.10,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize ApproachSkill.
        
        Args:
            max_steps: Maximum steps before timeout.
            pos_threshold: Position error threshold for success (meters).
            pregrasp_height: Height above object for pre-grasp pose.
            config: Optional configuration (overrides other params).
        """
        super().__init__(max_steps=max_steps, config=config)
        
        if config:
            self.pos_threshold = config.approach_pos_threshold
            self.pregrasp_height = config.approach_pregrasp_height
            self.max_steps = config.approach_max_steps
        else:
            self.pos_threshold = pos_threshold
            self.pregrasp_height = pregrasp_height
        
        self.controller = CartesianPDController.from_config(self.config)
    
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check approach preconditions."""
        obj_name = args.get("obj")
        
        if obj_name is None:
            return False, "Missing 'obj' argument"
        
        if obj_name not in world_state.objects:
            return False, f"Object '{obj_name}' not found in world state"
        
        if world_state.is_holding():
            return False, "Gripper is already holding an object"
        
        return True, "OK"
    
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if approach succeeded."""
        obj_name = args.get("obj")
        
        if world_state.gripper_pose is None:
            return False, "No gripper pose available"
        
        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return False, f"Object '{obj_name}' not found"
        
        # Check if gripper is at pre-grasp pose
        target = compute_pregrasp_pose(obj_pose, self.pregrasp_height)
        distance = np.linalg.norm(world_state.gripper_pose[:3] - target[:3])
        
        if distance > self.pos_threshold:
            return False, f"Gripper not at target: {distance:.3f}m > {self.pos_threshold}m"
        
        return True, "OK"
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute approach motion."""
        obj_name = args.get("obj")

        # Get target object pose
        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Object '{obj_name}' not found", "steps_taken": 0}
            )

        steps_taken = 0
        trajectory = []
        last_obs = None

        # Get initial pose from world state
        current_pose = world_state.gripper_pose
        if current_pose is None:
            # Do initial step to get observation
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Compute pre-grasp target with current gripper orientation
        gripper_ori = current_pose[3:7] if current_pose is not None else None
        target_pose = compute_pregrasp_pose(obj_pose, self.pregrasp_height, gripper_orientation=gripper_ori)
        self.controller.set_target(target_pose, gripper=1.0)  # Open gripper

        for step in range(self.max_steps):
            steps_taken = step + 1

            if current_pose is None:
                return SkillResult(
                    success=False,
                    info={"error_msg": "Failed to get gripper pose", "steps_taken": steps_taken}
                )

            trajectory.append(current_pose[:3].copy())

            # Check if at target - use relaxed orientation threshold for approach
            # Position is what matters for pre-grasp; orientation can be off
            if self.controller.is_at_target(current_pose, self.pos_threshold, ori_threshold=3.0):
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "reached_target": True,
                        "final_pose": current_pose,
                        "final_error": self.controller.position_error(current_pose),
                    }
                )

            # Compute and apply action
            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Final check after all steps - might have reached target on last step
        if current_pose is not None and self.controller.is_at_target(current_pose, self.pos_threshold, ori_threshold=3.0):
            return SkillResult(
                success=True,
                info={
                    "steps_taken": steps_taken,
                    "reached_target": True,
                    "final_pose": current_pose,
                    "final_error": self.controller.position_error(current_pose),
                }
            )

        # Timeout
        final_pose = current_pose
        final_error = self.controller.position_error(final_pose) if final_pose is not None else float('inf')

        return SkillResult(
            success=False,
            info={
                "error_msg": f"Timeout after {self.max_steps} steps",
                "steps_taken": steps_taken,
                "timeout": True,
                "final_pose": final_pose,
                "final_error": final_error,
            }
        )
    
    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after successful approach.
        
        Approach doesn't change symbolic state, just updates gripper pose.
        """
        if "final_pose" in result.info and result.info["final_pose"] is not None:
            world_state.gripper_pose = result.info["final_pose"]
