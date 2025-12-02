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

        # Velocity-based early termination
        # Tuned to allow robot to work through kinematic constraints
        self.min_velocity = 0.0005  # m/step - halved to be less aggressive (was 0.001)
        self.velocity_window = 20  # Check velocity over longer window (was 10)
        self.stuck_threshold = 5  # Need to be stuck for 5 consecutive checks before giving up (was 3)
    
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
        position_history = []  # Track positions for velocity calculation

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
        stuck_counter = 0
        height_boost = 0.0  # Additional height when stuck
        max_height_boost = 0.25  # Maximum extra height (increased for edge-of-workspace targets)
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

                # Check if at target (XY only, Z can be higher due to height boost)
                xy_dist = np.linalg.norm(current_pose[:2] - target_pos[:2])
                if xy_dist < self.pos_threshold:
                    break

                # Track position for velocity check
                pose_array = np.array(current_pose) if not isinstance(current_pose, np.ndarray) else current_pose
                position_history.append(pose_array[:3].copy())
                if len(position_history) > self.velocity_window:
                    position_history.pop(0)

                # Check if stuck (low velocity)
                if len(position_history) >= self.velocity_window:
                    start_pos = position_history[0]
                    end_pos = position_history[-1]
                    displacement = np.linalg.norm(end_pos - start_pos)
                    velocity = displacement / self.velocity_window

                    if velocity < self.min_velocity:
                        stuck_counter += 1
                        # Recovery: boost height to extend reach
                        if stuck_counter >= 2 and height_boost < max_height_boost:
                            height_boost += 0.03  # Increase height by 3cm
                            above_target[2] = max(current_pose[2], self.move_height + height_boost)
                            self.controller.set_target(above_target, gripper=-1.0)
                            position_history.clear()  # Reset velocity tracking after adjustment
                            stuck_counter = 0  # Reset stuck counter

                        # If stuck for multiple consecutive windows even with max height, give up
                        if stuck_counter >= self.stuck_threshold:
                            break
                    else:
                        stuck_counter = 0

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)
                last_obs = obs

        # Check final position with relaxed threshold
        # Use 10cm for marginal cases - PlaceSkill will handle final approach
        final_pose = current_pose
        relaxed_threshold = self.pos_threshold * 2.0  # 5cm -> 10cm

        if final_pose is not None:
            final_pose_arr = np.array(final_pose) if not isinstance(final_pose, np.ndarray) else final_pose
            xy_dist = np.linalg.norm(final_pose_arr[:2] - target_pos[:2])

            if xy_dist < self.pos_threshold:
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "final_pose": final_pose_arr.tolist(),
                        "target_reached": target,
                        "xy_error": float(xy_dist),
                    }
                )

            # Accept marginal success if within relaxed threshold
            if xy_dist < relaxed_threshold:
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "final_pose": final_pose_arr.tolist(),
                        "target_reached": target,
                        "xy_error": float(xy_dist),
                        "marginal": True,
                    }
                )

        # Build failure info
        fail_info = {
            "error_msg": f"Failed to reach target '{target}'",
            "steps_taken": steps_taken,
            "timeout": steps_taken >= self.max_steps,
            "stuck": stuck_counter >= 3,
        }
        if final_pose is not None:
            final_pose_arr = np.array(final_pose) if not isinstance(final_pose, np.ndarray) else final_pose
            fail_info["final_pose"] = final_pose_arr.tolist()
            fail_info["xy_error"] = float(np.linalg.norm(final_pose_arr[:2] - target_pos[:2]))

        return SkillResult(success=False, info=fail_info)
    
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
            # Ensure it's stored as numpy array for downstream skills
            pose = result.info["final_pose"]
            world_state.gripper_pose = np.array(pose) if isinstance(pose, list) else pose
