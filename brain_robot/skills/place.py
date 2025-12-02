"""PlaceObject skill.

Lowers held object and releases at target location.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class PlaceSkill(Skill):
    """Place held object at target location.
    
    Preconditions:
    - Holding the specified object
    - Gripper is above target location
    
    Postconditions:
    - No longer holding object
    - Object is at/in target location
    """
    
    name = "PlaceObject"
    
    def __init__(
        self,
        max_steps: int = 50,
        lower_speed: float = 0.02,
        release_height: float = 0.02,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize PlaceSkill.
        
        Args:
            max_steps: Maximum steps before timeout.
            lower_speed: Speed for lowering.
            release_height: Height above target to release.
            config: Optional configuration.
        """
        super().__init__(max_steps=max_steps, config=config)
        
        if config:
            self.lower_speed = config.place_lower_speed
            self.release_height = config.place_release_height
            self.max_steps = config.place_max_steps
        else:
            self.lower_speed = lower_speed
            self.release_height = release_height
        
        self.controller = CartesianPDController.from_config(self.config)
    
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check place preconditions."""
        obj_name = args.get("obj")
        target = args.get("region") or args.get("target")
        
        if obj_name is None:
            return False, "Missing 'obj' argument"
        
        if target is None:
            return False, "Missing 'region' or 'target' argument"
        
        if not world_state.is_holding(obj_name):
            return False, f"Not holding '{obj_name}'"
        
        return True, "OK"
    
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if place succeeded."""
        obj_name = args.get("obj")
        
        if world_state.is_holding(obj_name):
            return False, f"Still holding '{obj_name}'"
        
        return True, "OK"
    
    def _get_target_height(self, world_state: WorldState, target: str) -> float:
        """Get target height for placement."""
        if target in world_state.objects:
            target_pose = world_state.get_object_pose(target)
            if target_pose is not None:
                # Place above target object
                return target_pose[2] + 0.05  # 5cm above target
        
        # Default to table height + release offset
        return 0.02 + self.release_height
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute place: lower and release."""
        obj_name = args.get("obj")
        target = args.get("region") or args.get("target")

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
        steps_per_phase = self.max_steps // 2

        # Phase 1: Lower to release height
        target_height = self._get_target_height(world_state, target)
        lower_target = current_pose.copy()
        lower_target[2] = target_height
        self.controller.set_target(lower_target, gripper=-1.0)  # Keep closed while lowering

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

        # Phase 2: Open gripper to release
        for step in range(steps_per_phase):
            steps_taken += 1

            action = np.zeros(7)
            action[6] = 1.0  # Open gripper
            obs, _, _, _ = self._step_env(env, action)
            last_obs = obs

        final_pose = self._get_gripper_pose(env, last_obs)

        return SkillResult(
            success=True,
            info={
                "steps_taken": steps_taken,
                "final_pose": final_pose,
                "placed_object": obj_name,
                "placed_at": target,
            }
        )
    
    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after place."""
        obj_name = args.get("obj")
        target = args.get("region") or args.get("target")
        
        # No longer holding
        world_state.set_holding(None)
        
        # Update object location
        if target in world_state.objects:
            # Placed on/in another object
            # Determine if it's "in" or "on" based on object type
            target_obj = world_state.objects.get(target)
            if target_obj and target_obj.object_type in ['drawer', 'cabinet', 'box', 'container']:
                world_state.set_inside(obj_name, target)
            else:
                world_state.set_on(obj_name, target)
        else:
            # Placed on generic surface/region
            world_state.set_on(obj_name, target)
        
        if "final_pose" in result.info and result.info["final_pose"] is not None:
            world_state.gripper_pose = result.info["final_pose"]
