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
        release_height: float = 0.02,
        config: Optional[SkillConfig] = None,
    ):
        """Initialize PlaceSkill.

        Args:
            max_steps: Maximum steps before timeout.
            release_height: Height above target to release.
            config: Optional configuration.
        """
        super().__init__(max_steps=max_steps, config=config)

        if config:
            self.release_height = config.place_release_height
            self.max_steps = config.place_max_steps
        else:
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

    def _verify_placement(
        self,
        env,
        obj_name: str,
        target_name: str,
        target_pos: Optional[np.ndarray],
        grasp_offset: np.ndarray,
    ) -> Tuple[bool, dict]:
        """Verify object was placed correctly near target.

        Args:
            env: Environment
            obj_name: Name of placed object
            target_name: Name of target
            target_pos: Target position (may be stale, will get live)
            grasp_offset: Grasp offset used during placement

        Returns:
            Tuple of (success, info_dict)
        """
        info = {}

        # Get live object position
        obj_pos = self._get_live_object_position(env, obj_name)
        if obj_pos is None:
            info["failure_reason"] = "could_not_find_object"
            return False, info

        # Get live target position
        live_target = self._get_live_target_position(env, target_name)
        if live_target is not None:
            target_pos = live_target

        if target_pos is None:
            # No target position available, can't verify - be conservative
            info["failure_reason"] = "no_target_position"
            return False, info

        # Check XY distance from object center to target center
        xy_distance = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        info["final_xy_error"] = float(xy_distance)

        # LIBERO typically requires <3cm XY precision for "on" relation
        xy_threshold = 0.04  # 4cm - slightly relaxed
        if xy_distance > xy_threshold:
            info["failure_reason"] = f"xy_error_too_large ({xy_distance:.3f}m > {xy_threshold}m)"
            return False, info

        # Check Z is reasonable (object on/above target surface)
        z_diff = obj_pos[2] - target_pos[2]
        info["final_z_diff"] = float(z_diff)

        # Object should be within a reasonable band above target
        # (slightly below is OK due to settling, but not falling through)
        if z_diff < -0.02:  # More than 2cm below target surface
            info["failure_reason"] = f"object_below_target (z_diff={z_diff:.3f}m)"
            return False, info

        return True, info

    def _get_live_object_position(self, env, obj_name: str) -> Optional[np.ndarray]:
        """Get current object position from simulator."""
        try:
            body_name = obj_name
            try:
                body_id = env.sim.model.body_name2id(body_name)
            except ValueError:
                if body_name.endswith('_main'):
                    body_name = body_name[:-5]
                else:
                    body_name = body_name + '_main'
                try:
                    body_id = env.sim.model.body_name2id(body_name)
                except ValueError:
                    return None
            return env.sim.data.body_xpos[body_id].copy()
        except Exception:
            return None

    def _get_live_target_position(self, env, target: str) -> np.ndarray:
        """Get real-time target position from simulator.

        This is critical for accurate placement since objects may move
        during the place skill execution.
        """
        try:
            # Try exact name first
            body_name = target
            try:
                body_id = env.sim.model.body_name2id(body_name)
            except ValueError:
                # Try with/without _main suffix
                if body_name.endswith('_main'):
                    body_name = body_name[:-5]
                else:
                    body_name = body_name + '_main'
                try:
                    body_id = env.sim.model.body_name2id(body_name)
                except ValueError:
                    return None
            return env.sim.data.body_xpos[body_id].copy()
        except (ValueError, AttributeError):
            return None
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute place: center over target, lower, and release."""
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

        # Ensure current_pose is numpy array
        current_pose = np.array(current_pose) if not isinstance(current_pose, np.ndarray) else current_pose

        steps_taken = 0
        # Allocate steps: lift (15%), XY centering (25%), lower (30%), release (30%)
        lift_steps = self.max_steps // 7
        xy_center_steps = self.max_steps // 4
        lower_steps = self.max_steps // 3
        release_steps = self.max_steps - lift_steps - xy_center_steps - lower_steps

        # Get target XY position
        target_pos = None
        if target in world_state.objects:
            target_pos = world_state.get_object_position(target)

        xy_error_before = 0.0
        xy_error_after = 0.0

        # Phase -1: LIFT before centering to avoid collision
        # Critical fix: Bowl may be at same height as plate after Move skill
        # Must lift to safe height before any lateral movement
        safe_clearance = 0.05  # 5cm above target surface
        if target_pos is not None:
            target_surface_z = target_pos[2] + 0.02  # Approximate surface height
            required_z = target_surface_z + safe_clearance

            if current_pose[2] < required_z:
                lift_target = current_pose.copy()
                lift_target[2] = required_z
                self.controller.set_target(lift_target, gripper=1.0)  # Keep closed

                for step in range(lift_steps):
                    steps_taken += 1
                    if current_pose is None:
                        break

                    z_error = required_z - current_pose[2]
                    if z_error < 0.01:  # Within 1cm of target height
                        break

                    action = self.controller.compute_action(current_pose)
                    # Zero out orientation control - same fix as grasp
                    action[3:6] = 0.0
                    obs, _, _, _ = self._step_env(env, action)
                    current_pose = self._get_gripper_pose(env, obs)
                    last_obs = obs

        # Phase 0: XY Centering with GRASP OFFSET COMPENSATION
        # Critical for LIBERO success which requires <3cm XY precision
        #
        # KEY INSIGHT: For rim grasps, bowl_center != gripper_center
        # grasp_offset_xy = bowl_center - gripper_center (computed during grasp)
        # To place bowl_center over plate_center:
        #   desired_gripper_pos = plate_center - grasp_offset_xy
        #
        # We also use real-time tracking since the plate may move

        # Get grasp offset (defaults to [0,0] if not available)
        grasp_offset = getattr(world_state, 'grasp_offset_xy', np.zeros(2))
        if grasp_offset is None:
            grasp_offset = np.zeros(2)
        grasp_offset = np.array(grasp_offset)

        if target_pos is not None:
            # Compute where gripper should be so that OBJECT center is over target
            # gripper_target = plate_center - offset
            gripper_target_xy = target_pos[:2] - grasp_offset
            xy_error_before = np.linalg.norm(current_pose[:2] - gripper_target_xy)

            if xy_error_before > 0.015:  # Only if needed (>1.5cm error)
                for step in range(xy_center_steps):
                    steps_taken += 1
                    if current_pose is None:
                        break

                    # REAL-TIME TRACKING: Get fresh target position every step
                    live_target = self._get_live_target_position(env, target)
                    if live_target is not None:
                        target_pos = live_target
                        # Recompute gripper target with updated plate position
                        gripper_target_xy = target_pos[:2] - grasp_offset

                    xy_error = np.linalg.norm(current_pose[:2] - gripper_target_xy)
                    if xy_error < 0.015:  # 1.5cm threshold - tight for LIBERO
                        break

                    # Update controller target - move gripper to offset position
                    center_target = current_pose.copy()
                    center_target[0] = gripper_target_xy[0]
                    center_target[1] = gripper_target_xy[1]
                    # Keep current Z height during centering
                    self.controller.set_target(center_target, gripper=1.0)  # Keep closed

                    action = self.controller.compute_action(current_pose)
                    # Zero out orientation control during lateral movement
                    action[3:6] = 0.0
                    obs, _, _, _ = self._step_env(env, action)
                    current_pose = self._get_gripper_pose(env, obs)
                    last_obs = obs

            # Get final live target for accurate error measurement
            live_target = self._get_live_target_position(env, target)
            if live_target is not None:
                target_pos = live_target
                gripper_target_xy = target_pos[:2] - grasp_offset
            xy_error_after = np.linalg.norm(current_pose[:2] - gripper_target_xy) if current_pose is not None else xy_error_before

        # Phase 1: Lower to release height WITH VISUAL SERVO
        # Track target position during descent to correct for any drift
        target_height = self._get_target_height(world_state, target)
        lower_target = current_pose.copy()
        lower_target[2] = target_height
        # NOTE: In robosuite OSC, action[6] = +1.0 closes gripper, -1.0 opens
        self.controller.set_target(lower_target, gripper=1.0)  # Keep closed while lowering

        servo_interval = 5  # Check target position every 5 steps
        for step in range(lower_steps):
            steps_taken += 1
            if current_pose is None:
                break

            if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                break

            # VISUAL SERVO: Update XY target based on live plate position
            if step > 0 and step % servo_interval == 0:
                live_target = self._get_live_target_position(env, target)
                if live_target is not None:
                    # Recompute gripper target with offset compensation
                    gripper_target_xy = live_target[:2] - grasp_offset
                    lower_target[0] = gripper_target_xy[0]
                    lower_target[1] = gripper_target_xy[1]
                    self.controller.set_target(lower_target, gripper=1.0)

            action = self.controller.compute_action(current_pose)
            # Zero out orientation control during lowering
            action[3:6] = 0.0
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Phase 2: Open gripper to release
        # NOTE: In robosuite OSC, action[6] = -1.0 opens gripper
        for step in range(release_steps):
            steps_taken += 1

            action = np.zeros(7)
            action[6] = -1.0  # Open gripper (inverted in robosuite)
            obs, _, _, _ = self._step_env(env, action)
            last_obs = obs

        final_pose = self._get_gripper_pose(env, last_obs)

        # Verify placement: check if object is near target
        placed_successfully, placement_info = self._verify_placement(
            env, obj_name, target, target_pos, grasp_offset
        )

        result_info = {
            "steps_taken": steps_taken,
            "final_pose": final_pose,
            "placed_object": obj_name,
            "placed_at": target,
            "xy_error_before": float(xy_error_before),
            "xy_error_after": float(xy_error_after),
            **placement_info,
        }

        if placed_successfully:
            return SkillResult(success=True, info=result_info)
        else:
            result_info["error_msg"] = f"Placement verification failed: {placement_info.get('failure_reason', 'unknown')}"
            # Propagate failure - don't mask it
            return SkillResult(success=False, info=result_info)
    
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

        # Clear grasp offset - it's no longer valid after releasing
        if hasattr(world_state, 'grasp_offset_xy'):
            world_state.grasp_offset_xy = None

        # Only update symbolic state if placement succeeded
        if result.success:
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
