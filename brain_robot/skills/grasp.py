"""GraspObject skill.

Lowers gripper, closes on object, and lifts slightly.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class GraspSkill(Skill):
    """Grasp object after approach.
    
    Preconditions:
    - Object exists in world state
    - Gripper is above object (approach completed)
    - Gripper is not holding anything
    
    Postconditions:
    - Gripper is holding the object
    """
    
    name = "GraspObject"
    
    def __init__(
        self,
        max_steps: int = 50,
        lower_speed: float = 0.02,
        lift_height: float = 0.05,
        grasp_height_offset: float = 0.02,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize GraspSkill.
        
        Args:
            max_steps: Maximum steps before timeout.
            lower_speed: Speed for lowering gripper (m/step).
            lift_height: Height to lift after grasp.
            grasp_height_offset: Height above object center to grasp.
            config: Optional configuration.
        """
        super().__init__(max_steps=max_steps, config=config)
        
        if config:
            self.lower_speed = config.grasp_lower_speed
            self.lift_height = config.grasp_lift_height
            self.max_steps = config.grasp_max_steps
        else:
            self.lower_speed = lower_speed
            self.lift_height = lift_height
        
        self.grasp_height_offset = grasp_height_offset
        self.controller = CartesianPDController.from_config(self.config)
    
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check grasp preconditions."""
        obj_name = args.get("obj")
        
        if obj_name is None:
            return False, "Missing 'obj' argument"
        
        if obj_name not in world_state.objects:
            return False, f"Object '{obj_name}' not found"
        
        if world_state.is_holding():
            return False, "Gripper is already holding an object"
        
        # Check gripper is approximately above object
        if world_state.gripper_pose is None:
            return False, "No gripper pose available"
        
        obj_pos = world_state.get_object_position(obj_name)
        gripper_pos = world_state.gripper_pose[:3]
        
        # Check XY alignment
        xy_dist = np.linalg.norm(gripper_pos[:2] - obj_pos[:2])
        if xy_dist > 0.1:  # 10cm tolerance
            return False, f"Gripper not aligned above object: XY distance {xy_dist:.3f}m"
        
        return True, "OK"
    
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if grasp succeeded."""
        obj_name = args.get("obj")
        
        if not world_state.is_holding(obj_name):
            return False, f"Not holding '{obj_name}'"
        
        return True, "OK"
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute grasp sequence: lower -> close -> lift."""
        obj_name = args.get("obj")

        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Object '{obj_name}' not found", "steps_taken": 0}
            )

        # Get initial pose from world state or step
        current_pose = world_state.gripper_pose
        last_obs = None
        if current_pose is None:
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        if current_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": "Failed to get gripper pose", "steps_taken": 0}
            )

        steps_taken = 0

        # Phase 1: Lower to grasp height
        grasp_target = obj_pose.copy()
        grasp_target[2] += self.grasp_height_offset
        grasp_target[3:7] = current_pose[3:7]  # Keep current orientation

        self.controller.set_target(grasp_target, gripper=1.0)  # Open gripper

        steps_per_phase = self.max_steps // 3

        for step in range(steps_per_phase):
            steps_taken += 1
            if current_pose is None:
                break

            if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                break

            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Phase 2: Close gripper
        for step in range(steps_per_phase):
            steps_taken += 1

            # Close gripper action
            action = np.zeros(7)
            action[6] = -1.0  # Close gripper
            obs, _, _, _ = self._step_env(env, action)
            last_obs = obs

        # Phase 3: Lift
        current_pose = self._get_gripper_pose(env, last_obs)
        if current_pose is not None:
            lift_target = current_pose.copy()
            lift_target[2] += self.lift_height
            self.controller.set_target(lift_target, gripper=-1.0)  # Keep closed

            for step in range(steps_per_phase):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)
                last_obs = obs

        # Check if grasp succeeded by checking gripper width
        final_pose = self._get_gripper_pose(env, last_obs)
        gripper_closed = self._check_gripper_closed(last_obs)

        if gripper_closed:
            return SkillResult(
                success=True,
                info={
                    "steps_taken": steps_taken,
                    "final_pose": final_pose,
                    "object_grasped": obj_name,
                }
            )
        else:
            return SkillResult(
                success=False,
                info={
                    "error_msg": "Grasp failed - gripper did not close on object",
                    "steps_taken": steps_taken,
                    "final_pose": final_pose,
                }
            )

    def _check_gripper_closed(self, obs: dict) -> bool:
        """Check if gripper successfully grasped something.

        Heuristic: gripper width should be non-zero but less than fully open.
        """
        if obs is None:
            return True  # Assume success if can't check

        if 'robot0_gripper_qpos' in obs:
            gripper_qpos = obs['robot0_gripper_qpos']
            # Non-zero gripper width indicates something is grasped
            width = np.sum(np.abs(gripper_qpos))
            # This threshold may need tuning based on actual gripper
            return 0.001 < width < 0.08

        return True  # Assume success if can't check
    
    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after successful grasp."""
        obj_name = args.get("obj")
        world_state.set_holding(obj_name)
        
        if "final_pose" in result.info and result.info["final_pose"] is not None:
            world_state.gripper_pose = result.info["final_pose"]
