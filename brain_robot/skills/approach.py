"""ApproachObject skill.

Moves gripper to pre-grasp pose above target object.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController, compute_pregrasp_pose
from ..control.approach_selection import (
    select_approach_strategy,
    compute_angled_pregrasp_pose,
    APPROACH_ORIENTATIONS,
)
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
        xy_threshold: float = 0.05,
        pregrasp_height: float = 0.10,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize ApproachSkill.

        Args:
            max_steps: Maximum steps before timeout.
            pos_threshold: Total position error threshold for success (meters).
            xy_threshold: XY-specific error threshold (meters).
            pregrasp_height: Height above object for pre-grasp pose.
            config: Optional configuration (overrides other params).
        """
        super().__init__(max_steps=max_steps, config=config)

        if config:
            self.pos_threshold = config.approach_pos_threshold
            self.xy_threshold = config.approach_xy_threshold
            self.pregrasp_height = config.approach_pregrasp_height
            self.max_steps = config.approach_max_steps
        else:
            self.pos_threshold = pos_threshold
            self.xy_threshold = xy_threshold
            self.pregrasp_height = pregrasp_height

        self.controller = CartesianPDController.from_config(self.config)

    def _is_at_target(self, current_pose: np.ndarray, target_pose: np.ndarray) -> bool:
        """Check if gripper is at target with both total and XY thresholds.

        Uses a dual threshold: total position error AND XY-specific error.
        This prevents declaring success when Z is correct but XY is far off.
        """
        total_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
        xy_error = np.linalg.norm(current_pose[:2] - target_pose[:2])

        return total_error < self.pos_threshold and xy_error < self.xy_threshold

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
        """Execute approach motion with adaptive approach direction."""
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

        # Determine object context for approach selection
        obj_type = None
        if obj_name in world_state.objects:
            obj_type = world_state.objects[obj_name].object_type

        # Check if object is inside a drawer/cabinet (using spatial relations)
        in_drawer = False
        if obj_name in world_state.inside:
            container = world_state.inside[obj_name]
            in_drawer = any(x in container.lower() for x in ['drawer', 'cabinet'])
        # Fallback to name-based check
        if not in_drawer:
            in_drawer = 'drawer' in obj_name.lower() or (obj_type and 'drawer' in str(obj_type).lower())

        on_elevated = obj_pose[2] > 1.05  # Above normal table height

        # Select optimal approach strategy based on object position
        strategy_name, approach_dir, gripper_ori = select_approach_strategy(
            obj_pos=obj_pose[:3],
            in_drawer=in_drawer,
            on_elevated_surface=on_elevated,
        )

        # Compute pre-grasp pose using selected approach
        target_pose = compute_angled_pregrasp_pose(
            object_pose=obj_pose,
            approach_direction=approach_dir,
            gripper_orientation=gripper_ori,
            pregrasp_distance=self.pregrasp_height,
        )

        # Store approach strategy in world state for use by GraspSkill
        world_state.approach_strategy = strategy_name
        world_state.approach_direction = approach_dir
        world_state.gripper_orientation = gripper_ori

        # NOTE: In robosuite OSC, action[6] = -1.0 opens gripper, +1.0 closes
        self.controller.set_target(target_pose, gripper=-1.0)  # Open gripper (inverted)

        for step in range(self.max_steps):
            steps_taken = step + 1

            if current_pose is None:
                return SkillResult(
                    success=False,
                    info={"error_msg": "Failed to get gripper pose", "steps_taken": steps_taken}
                )

            trajectory.append(current_pose[:3].copy())

            # Check if at target with both total and XY thresholds
            # This prevents declaring success when Z is correct but XY is far off
            if self._is_at_target(current_pose, target_pose):
                xy_error = np.linalg.norm(current_pose[:2] - target_pose[:2])
                z_error = abs(current_pose[2] - target_pose[2])
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "reached_target": True,
                        "final_pose": current_pose,
                        "final_error": self.controller.position_error(current_pose),
                        "xy_error": xy_error,
                        "z_error": z_error,
                        "approach_strategy": strategy_name,
                    }
                )

            # Compute and apply action
            action = self.controller.compute_action(current_pose)
            # Zero out orientation control - position-only control is more reliable
            action[3:6] = 0.0
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Final check after all steps - might have reached target on last step
        if current_pose is not None and self._is_at_target(current_pose, target_pose):
            xy_error = np.linalg.norm(current_pose[:2] - target_pose[:2])
            z_error = abs(current_pose[2] - target_pose[2])
            return SkillResult(
                success=True,
                info={
                    "steps_taken": steps_taken,
                    "reached_target": True,
                    "final_pose": current_pose,
                    "final_error": self.controller.position_error(current_pose),
                    "xy_error": xy_error,
                    "z_error": z_error,
                }
            )

        # Timeout - log XY/Z errors for diagnosis
        final_pose = current_pose
        final_error = self.controller.position_error(final_pose) if final_pose is not None else float('inf')
        xy_error = np.linalg.norm(final_pose[:2] - target_pose[:2]) if final_pose is not None else float('inf')
        z_error = abs(final_pose[2] - target_pose[2]) if final_pose is not None else float('inf')

        return SkillResult(
            success=False,
            info={
                "error_msg": f"Timeout after {self.max_steps} steps",
                "steps_taken": steps_taken,
                "timeout": True,
                "final_pose": final_pose,
                "final_error": final_error,
                "xy_error": xy_error,
                "z_error": z_error,
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
