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
from ..control.collision_avoidance import (
    generate_collision_aware_path,
    is_waypoint_reached,
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
        """Execute approach motion with adaptive direction and collision avoidance."""
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

        # Get initial pose from world state
        current_pose = world_state.gripper_pose
        if current_pose is None:
            # Do initial step to get observation
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)

        # Determine object context for approach selection
        obj_type = None
        if obj_name in world_state.objects:
            obj_type = world_state.objects[obj_name].object_type

        # Determine object context based on HEIGHT first, then spatial relations
        # Height is more reliable than semantic "inside" relations which can be wrong
        #
        # Height thresholds:
        # - z > 1.20: ON TOP of cabinet (very high) -> front_angled_steep
        # - z > 1.05: On elevated surface (stove, etc) -> front_angled
        # - z < 1.05 with "inside" relation: Actually inside drawer/cabinet -> front_horizontal

        in_drawer = False
        on_cabinet = False
        on_elevated = obj_pose[2] > 1.05  # Above normal table height

        # Check if object is ON TOP of a cabinet (high surface)
        # Cabinet tops are typically at z > 1.20
        if obj_pose[2] > 1.20:
            on_cabinet = True
        elif hasattr(world_state, 'on_top') and world_state.on_top and obj_name in world_state.on_top:
            surface = world_state.on_top[obj_name]
            if 'cabinet' in surface.lower():
                on_cabinet = True

        # Check if object is INSIDE a drawer/cabinet (only if NOT on top)
        # Objects truly inside cabinets are at lower heights (z < 1.20)
        if not on_cabinet:
            if hasattr(world_state, 'inside') and world_state.inside and obj_name in world_state.inside:
                container = world_state.inside[obj_name]
                # Check for drawer OR cabinet - but only if height suggests inside
                if any(x in container.lower() for x in ['drawer', 'cabinet']):
                    # Sanity check: objects inside drawers/cabinets should be below 1.20
                    if obj_pose[2] < 1.20:
                        in_drawer = True
            # Fallback to name-based check
            if not in_drawer:
                in_drawer = 'drawer' in obj_name.lower() or (obj_type and 'drawer' in str(obj_type).lower())

        # Select optimal approach strategy based on object position
        strategy_name, approach_dir, gripper_ori = select_approach_strategy(
            obj_pos=obj_pose[:3],
            in_drawer=in_drawer,
            on_elevated_surface=on_elevated,
            on_cabinet=on_cabinet,
        )

        # Compute final pre-grasp pose using selected approach
        final_target_pose = compute_angled_pregrasp_pose(
            object_pose=obj_pose,
            approach_direction=approach_dir,
            gripper_orientation=gripper_ori,
            pregrasp_distance=self.pregrasp_height,
        )

        # Store approach strategy in world state for use by GraspSkill
        world_state.approach_strategy = strategy_name
        world_state.approach_direction = approach_dir
        world_state.gripper_orientation = gripper_ori

        # Generate collision-aware waypoints for constrained locations
        use_waypoints = in_drawer or on_cabinet or on_elevated
        waypoints = []
        if use_waypoints and current_pose is not None:
            waypoint_positions = generate_collision_aware_path(
                start_pos=current_pose[:3],
                target_pos=final_target_pose[:3],
                in_drawer=in_drawer,
                on_cabinet=on_cabinet,
                on_elevated=on_elevated,
            )
            # Convert position waypoints to full 7D poses
            for wp_pos in waypoint_positions[:-1]:  # Exclude final (that's target)
                wp_pose = np.zeros(7)
                wp_pose[:3] = wp_pos
                wp_pose[3:7] = gripper_ori
                waypoints.append(wp_pose)

        # Add final target
        waypoints.append(final_target_pose)

        # Execute waypoint-based motion
        current_waypoint_idx = 0
        target_pose = waypoints[current_waypoint_idx]

        # NOTE: In robosuite OSC, action[6] = -1.0 opens gripper, +1.0 closes
        self.controller.set_target(target_pose, gripper=-1.0)

        for step in range(self.max_steps):
            steps_taken = step + 1

            if current_pose is None:
                return SkillResult(
                    success=False,
                    info={"error_msg": "Failed to get gripper pose", "steps_taken": steps_taken}
                )

            trajectory.append(current_pose[:3].copy())

            # Check if reached current waypoint
            if is_waypoint_reached(current_pose[:3], target_pose[:3], threshold=0.04):
                current_waypoint_idx += 1
                if current_waypoint_idx >= len(waypoints):
                    # Reached final target
                    xy_error = np.linalg.norm(current_pose[:2] - final_target_pose[:2])
                    z_error = abs(current_pose[2] - final_target_pose[2])
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
                            "waypoints_used": len(waypoints),
                        }
                    )
                # Move to next waypoint
                target_pose = waypoints[current_waypoint_idx]
                self.controller.set_target(target_pose, gripper=-1.0)

            # Also check final target directly (for non-waypoint paths)
            if self._is_at_target(current_pose, final_target_pose):
                xy_error = np.linalg.norm(current_pose[:2] - final_target_pose[:2])
                z_error = abs(current_pose[2] - final_target_pose[2])
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
                        "waypoints_used": len(waypoints),
                    }
                )

            # Compute and apply action
            action = self.controller.compute_action(current_pose)
            # Zero out orientation control - position-only control is more reliable
            action[3:6] = 0.0
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)

        # Final check after all steps
        if current_pose is not None and self._is_at_target(current_pose, final_target_pose):
            xy_error = np.linalg.norm(current_pose[:2] - final_target_pose[:2])
            z_error = abs(current_pose[2] - final_target_pose[2])
            return SkillResult(
                success=True,
                info={
                    "steps_taken": steps_taken,
                    "reached_target": True,
                    "final_pose": current_pose,
                    "final_error": self.controller.position_error(current_pose),
                    "xy_error": xy_error,
                    "z_error": z_error,
                    "waypoints_used": len(waypoints),
                }
            )

        # Timeout - log errors for diagnosis
        final_pose = current_pose
        final_error = self.controller.position_error(final_pose) if final_pose is not None else float('inf')
        xy_error = np.linalg.norm(final_pose[:2] - final_target_pose[:2]) if final_pose is not None else float('inf')
        z_error = abs(final_pose[2] - final_target_pose[2]) if final_pose is not None else float('inf')

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
                "waypoint_progress": f"{current_waypoint_idx}/{len(waypoints)}",
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
