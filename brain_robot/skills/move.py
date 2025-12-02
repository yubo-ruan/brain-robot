"""MoveObjectToRegion skill.

Transports held object to target region.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class MoveSkill(Skill):
    """Move held object to target region.
    
    Preconditions:
    - Holding the specified object
    - Target region/object exists
    
    Postconditions:
    - Gripper (and held object) is above target region
    """
    
    name = "MoveObjectToRegion"
    
    def __init__(
        self,
        max_steps: int = 150,
        pos_threshold: float = 0.03,
        move_height: float = 0.15,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize MoveSkill.
        
        Args:
            max_steps: Maximum steps before timeout.
            pos_threshold: Position error threshold.
            move_height: Height to maintain during transport.
            config: Optional configuration.
        """
        super().__init__(max_steps=max_steps, config=config)
        
        if config:
            self.pos_threshold = config.move_pos_threshold
            self.max_steps = config.move_max_steps
        else:
            self.pos_threshold = pos_threshold
        
        self.move_height = move_height
        self.controller = CartesianPDController.from_config(self.config)
    
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check move preconditions."""
        obj_name = args.get("obj")
        target = args.get("region") or args.get("target")
        
        if obj_name is None:
            return False, "Missing 'obj' argument"
        
        if target is None:
            return False, "Missing 'region' or 'target' argument"
        
        if not world_state.is_holding(obj_name):
            return False, f"Not holding '{obj_name}'"
        
        # Check target exists (either as object or special region)
        if target not in world_state.objects and not self._is_special_region(target):
            return False, f"Target '{target}' not found"
        
        return True, "OK"
    
    def _is_special_region(self, target: str) -> bool:
        """Check if target is a special region name."""
        # Special regions that don't correspond to objects
        special_regions = ['table', 'workspace', 'left', 'right', 'center']
        return any(r in target.lower() for r in special_regions)
    
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if move succeeded."""
        target = args.get("region") or args.get("target")
        
        if world_state.gripper_pose is None:
            return False, "No gripper pose available"
        
        target_pos = self._get_target_position(world_state, target)
        if target_pos is None:
            return False, f"Cannot determine target position for '{target}'"
        
        # Check XY alignment (Z doesn't matter much during move)
        gripper_xy = world_state.gripper_pose[:2]
        target_xy = target_pos[:2]
        xy_dist = np.linalg.norm(gripper_xy - target_xy)
        
        if xy_dist > self.pos_threshold:
            return False, f"Not above target: {xy_dist:.3f}m"
        
        return True, "OK"
    
    def _get_target_position(self, world_state: WorldState, target: str) -> Optional[np.ndarray]:
        """Get 3D position of target."""
        if target in world_state.objects:
            return world_state.get_object_position(target)
        
        # Handle special regions (these would need to be defined per-task)
        # For now, return None for unknown regions
        return None
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute move: go up, translate, position above target."""
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

        target_pos = self._get_target_position(world_state, target)
        if target_pos is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Cannot find target '{target}'", "steps_taken": 0}
            )

        steps_taken = 0

        # Dynamic step allocation: lift needs fewer steps than translate
        lift_budget = min(50, self.max_steps // 6)  # Quick lift
        translate_budget = self.max_steps - lift_budget  # Rest for translation

        # Phase 1: Move up to safe height (if needed)
        if current_pose[2] < self.move_height - 0.02:
            up_target = current_pose.copy()
            up_target[2] = self.move_height
            self.controller.set_target(up_target, gripper=-1.0)  # Keep closed

            for step in range(lift_budget):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=0.02, ori_threshold=3.0):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)
                last_obs = obs

        # Phase 2: Translate to above target (gets most of the step budget)
        if current_pose is not None:
            above_target = current_pose.copy()
            above_target[0] = target_pos[0]
            above_target[1] = target_pos[1]
            above_target[2] = max(current_pose[2], self.move_height)  # Maintain height
            self.controller.set_target(above_target, gripper=-1.0)

            for step in range(translate_budget):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=self.pos_threshold, ori_threshold=3.0):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)
                last_obs = obs

        # Check final position
        final_pose = current_pose
        if final_pose is not None:
            xy_dist = np.linalg.norm(final_pose[:2] - target_pos[:2])
            if xy_dist < self.pos_threshold:
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "final_pose": final_pose,
                        "target_reached": target,
                    }
                )

        return SkillResult(
            success=False,
            info={
                "error_msg": f"Failed to reach target '{target}'",
                "steps_taken": steps_taken,
                "timeout": steps_taken >= self.max_steps,
                "final_pose": final_pose,
            }
        )
    
    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after move.
        
        Move doesn't change symbolic holding state, just position.
        """
        if "final_pose" in result.info and result.info["final_pose"] is not None:
            world_state.gripper_pose = result.info["final_pose"]
