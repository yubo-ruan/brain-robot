"""GraspObject skill.

Lowers gripper, closes on object, and lifts slightly.
Includes XY refinement phase to correct ApproachSkill residual error.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from .grasp_selection import GraspSelector, get_grasp_selector
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class GraspSkill(Skill):
    """Grasp object after approach.

    Includes XY refinement phase at the start to correct residual
    positioning error from ApproachSkill. This is a "soft precondition fix"
    that allows grasping even when approach leaves the gripper 5-15cm off.

    For hollow objects (bowls, mugs, cups), targets the rim rather than
    center to avoid "ghost grasps" where fingers descend into empty interior.

    Supports pluggable grasp selection via GraspSelector interface:
    - HeuristicGraspSelector: Rule-based (default)
    - GIGAGraspSelector: 6-DoF learned affordances

    Preconditions:
    - Object exists in world state
    - Gripper is above object (approach completed)
    - Gripper is not holding anything

    Postconditions:
    - Gripper is holding the object
    """

    name = "GraspObject"

    # Object types that are hollow and need rim grasping
    HOLLOW_OBJECTS = {'bowl', 'mug', 'cup', 'ramekin'}

    # Approximate radii for hollow objects (meters)
    # Used to compute rim offset: offset = radius - (finger_width / 2)
    # Reduced from actual radii to stay within robot workspace
    OBJECT_RADII = {
        'bowl': 0.045,      # Reduced from 0.056 to stay in workspace
        'mug': 0.035,       # Reduced from 0.04
        'cup': 0.035,       # Reduced from 0.04
        'ramekin': 0.030,   # Reduced from 0.035
    }

    # Gripper finger half-width (meters)
    GRIPPER_FINGER_HALF_WIDTH = 0.005  # ~0.5cm

    # Workspace limits - gripper can't reach certain regions without fighting
    MIN_WORKSPACE_Y = 0.12  # Don't go below Y=0.12m for top-down approaches

    # Default gripper-down orientation quaternion [w, x, y, z]
    DEFAULT_ORIENTATION = np.array([-0.02, 0.707, 0.707, -0.02])

    def __init__(
        self,
        max_steps: int = 100,
        lift_height: float = 0.05,
        grasp_height_offset: float = 0.04,  # Grasp 4cm above body origin to hit rim
        config: Optional[SkillConfig] = None,
        grasp_selector: Optional[GraspSelector] = None,
        grasp_selector_type: str = "heuristic",
    ):
        """Initialize GraspSkill.

        Args:
            max_steps: Maximum steps before timeout.
            lift_height: Height to lift after grasp.
            grasp_height_offset: Height relative to object center to grasp.
                                 0 = center, negative = below center (for cylinders).
            config: Optional configuration.
            grasp_selector: Custom GraspSelector instance (overrides grasp_selector_type).
            grasp_selector_type: Type of grasp selector ("heuristic" or "giga").
        """
        super().__init__(max_steps=max_steps, config=config)

        if config:
            self.lift_height = config.grasp_lift_height
            self.max_steps = config.grasp_max_steps
            # XY refinement config
            self.xy_refine_enabled = config.grasp_xy_refine_enabled
            self.xy_refine_max_steps = config.grasp_xy_refine_max_steps
            self.xy_refine_threshold = config.grasp_xy_refine_threshold
            self.xy_refine_min_improvement = config.grasp_xy_refine_min_improvement
        else:
            self.lift_height = lift_height
            # Default XY refinement settings
            self.xy_refine_enabled = True
            self.xy_refine_max_steps = 30
            self.xy_refine_threshold = 0.03
            self.xy_refine_min_improvement = 0.005

        self.grasp_height_offset = grasp_height_offset
        self.controller = CartesianPDController.from_config(self.config)

        # Initialize grasp selector
        if grasp_selector is not None:
            self.grasp_selector = grasp_selector
        else:
            self.grasp_selector = get_grasp_selector(
                grasp_selector_type,
                grasp_height_offset=grasp_height_offset,
            )
    
    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check grasp preconditions.

        With XY refinement enabled, we relax the alignment threshold since
        the refinement phase will correct residual positioning errors.
        """
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

        # Check XY alignment - relaxed threshold when XY refinement enabled
        xy_dist = np.linalg.norm(gripper_pos[:2] - obj_pos[:2])
        # With refinement: allow up to 20cm (refinement will correct)
        # Without refinement: strict 10cm threshold
        max_xy_dist = 0.20 if self.xy_refine_enabled else 0.10
        if xy_dist > max_xy_dist:
            return False, f"Gripper not aligned above object: XY distance {xy_dist:.3f}m"

        return True, "OK"
    
    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if grasp succeeded."""
        obj_name = args.get("obj")
        
        if not world_state.is_holding(obj_name):
            return False, f"Not holding '{obj_name}'"
        
        return True, "OK"
    
    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute grasp sequence: [XY refine] -> lower -> close -> lift."""
        obj_name = args.get("obj")

        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Object '{obj_name}' not found", "steps_taken": 0}
            )

        # Get LIVE object position from simulator (after physics settling)
        # World state may have stale position from before physics settled
        live_obj_pos = self._get_live_object_position(env, obj_name)
        if live_obj_pos is not None:
            obj_pose = obj_pose.copy()
            obj_pose[:3] = live_obj_pos

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
        xy_refine_info = {}

        # Initialize grasp strategy early to avoid undefined in edge cases
        grasp_strategy = "heuristic"  # Default fallback
        using_learned_grasp = False
        grasp_width = 0.0  # Default gripper width

        # Phase 0: Ensure gripper is fully open while holding position
        # NOTE: In robosuite OSC, action[6] = -1.0 opens gripper, +1.0 closes
        # We use the PD controller to hold current position while opening
        self.controller.set_target(current_pose.copy(), gripper=-1.0)  # Open gripper
        for step in range(10):
            steps_taken += 1
            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Compute full 6-DoF grasp pose BEFORE XY refinement
        # For learned selectors (CGN), this gives us position + orientation + width
        # For heuristic, gives position with default orientation
        grasp_xyz, grasp_orientation, grasp_width, grasp_point_info = self._compute_grasp_pose(
            obj_pose, obj_name, world_state, current_pose
        )

        # Determine if we're using a learned grasp (affects servo behavior)
        using_learned_grasp = grasp_point_info.get("learned_grasp", False)
        grasp_strategy = grasp_point_info.get("grasp_strategy_used", "heuristic")

        # Phase 0.5: XY Refinement - target the grasp point, not object center!
        if self.xy_refine_enabled:
            # Create a pseudo-pose at the grasp XY for refinement targeting
            grasp_pose_for_refine = obj_pose.copy()
            grasp_pose_for_refine[0] = grasp_xyz[0]
            grasp_pose_for_refine[1] = grasp_xyz[1]

            xy_refine_info = self._xy_refinement_phase(
                env, current_pose, grasp_pose_for_refine, last_obs
            )
            steps_taken += xy_refine_info.get("steps", 0)
            current_pose = xy_refine_info.get("final_pose", current_pose)
            last_obs = xy_refine_info.get("last_obs", last_obs)

        # Phase 1: Descend to grasp point
        # Get approach strategy - use learned grasp strategy if available
        if using_learned_grasp:
            approach_strategy = grasp_strategy  # "contact_graspnet" or "giga"
        else:
            approach_strategy = getattr(world_state, 'approach_strategy', 'top_down')

        # Need to refresh current_pose from the environment
        obs, _, _, _ = self._step_env(env, np.zeros(7))
        current_pose = self._get_gripper_pose(env, obs)
        last_obs = obs

        # Build full 6-DoF grasp target
        grasp_target = current_pose.copy()
        grasp_target[0] = grasp_xyz[0]
        grasp_target[1] = grasp_xyz[1]
        grasp_target[2] = grasp_xyz[2]

        # For learned grasps, use predicted orientation
        # For heuristic grasps, keep current orientation (gripper-down)
        if using_learned_grasp and grasp_orientation is not None:
            grasp_target[3:7] = grasp_orientation

        # Compute grasp offset from object center ONCE for visual servo
        # This is the fixed offset between object center and grasp point
        # During servo, we apply: grasp_target = live_obj_pos + grasp_offset
        grasp_offset = grasp_xyz - obj_pose[:3]  # [x_off, y_off, z_off]

        # Gripper starts fully open during descent
        # Robosuite: -1.0 = fully open (~0.08m), +1.0 = fully closed (~0.0m)
        gripper_action = -1.0

        self.controller.set_target(grasp_target, gripper=gripper_action)

        # Use more steps for descent (50% of budget for this critical phase)
        descent_steps = self.max_steps // 2

        # Track descent for debugging
        descent_start_z = current_pose[2] if current_pose is not None else 0.0
        descent_converged = False
        visual_servo_corrections = 0
        missing_live_pose_count = 0  # Track when we can't get live object position

        for step in range(descent_steps):
            steps_taken += 1
            if current_pose is None:
                break

            # VISUAL SERVO: Check gripper-to-object error periodically
            # Update target if object moved or gripper drifted significantly
            servo_interval = 5
            if step > 0 and step % servo_interval == 0:
                live_obj_pos = self._get_live_object_position(env, obj_name)

                # Learned grasps: apply fixed offset, skip heuristic rim logic entirely
                if using_learned_grasp and live_obj_pos is not None:
                    # NOTE: We only servo XY, not Z. This assumes object height is
                    # constant on the table in LIBERO. If tasks involve vertical
                    # motion (stacked objects, non-table surfaces), change to:
                    # grasp_target[:3] = live_obj_pos + grasp_offset
                    grasp_target[:2] = live_obj_pos[:2] + grasp_offset[:2]
                    self.controller.set_target(grasp_target, gripper=gripper_action)
                    visual_servo_corrections += 1
                    missing_live_pose_count = 0

                # Heuristic grasps: full recompute with rim logic
                elif live_obj_pos is not None:
                    updated_obj_pose = obj_pose.copy()
                    updated_obj_pose[:3] = live_obj_pos
                    new_grasp_xyz, _ = self._compute_grasp_point(
                        updated_obj_pose, obj_name, world_state, current_pose
                    )
                    grasp_target[:3] = new_grasp_xyz
                    self.controller.set_target(grasp_target, gripper=gripper_action)
                    visual_servo_corrections += 1
                    missing_live_pose_count = 0

                else:
                    # No live pose available - keep previous target, track for debugging
                    missing_live_pose_count += 1

            # Check convergence with tight threshold for precise grasping
            pos_error = np.linalg.norm(current_pose[:3] - grasp_target[:3])
            if pos_error < 0.02:  # 2cm threshold for angled approaches
                descent_converged = True
                break

            action = self.controller.compute_action(current_pose)
            # Orientation control strategy depends on grasp type
            if using_learned_grasp:
                # For learned grasps (CGN/GIGA): use full orientation control
                # CGN gives us the target orientation, let controller track it
                action[3:6] *= 0.5  # Moderate gain to avoid oscillation
            elif approach_strategy == 'top_down':
                # For top-down heuristic: zero orientation to avoid fighting
                action[3:6] = 0.0
            else:
                # For angled heuristic approaches: reduced orientation control
                action[3:6] *= 0.3
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Record descent info for debugging
        descent_end_z = current_pose[2] if current_pose is not None else 0.0
        descent_info = {
            "start_z": descent_start_z,
            "target_z": grasp_xyz[2],
            "end_z": descent_end_z,
            "delta_z": descent_start_z - descent_end_z,
            "steps": step + 1 if 'step' in dir() else 0,
            "converged": descent_converged,
            "target_x": grasp_xyz[0],
            "target_y": grasp_xyz[1],
            "start_x": grasp_target[0] if current_pose is not None else 0.0,
            "start_y": grasp_target[1] if current_pose is not None else 0.0,
            "visual_servo_corrections": visual_servo_corrections,
            "missing_live_pose_count": missing_live_pose_count,
            "approach_strategy": approach_strategy,
            # CGN-specific logging
            "using_learned_grasp": using_learned_grasp,
            "grasp_strategy": grasp_strategy,
            "predicted_gripper_width": grasp_width,
            "target_orientation": grasp_orientation.tolist() if grasp_orientation is not None else None,
            "grasp_offset": grasp_offset.tolist(),  # Fixed offset for visual servo
        }

        # Phase 2: Close gripper with width-based stop condition
        # NOTE: In robosuite OSC, action[6] = +1.0 closes gripper
        close_steps = 20

        # For learned grasps: use predicted width as stop condition
        # For heuristic grasps: close fully (action=+1.0)
        if grasp_width > 0 and using_learned_grasp:
            # Target qpos based on predicted width
            # Empirical mapping: gripper_qpos_sum ≈ width (in meters)
            # Add 10% margin to ensure firm grasp
            target_qpos = grasp_width * 0.9
        else:
            target_qpos = 0.0  # Fully closed

        prev_qpos = None
        stall_count = 0
        gripper_close_action = 1.0  # Always command close
        close_stop_reason = "timeout"  # Default if loop completes
        stall_eps = 0.001  # Threshold for detecting gripper stall

        for step in range(close_steps):
            steps_taken += 1

            # Close gripper action
            action = np.zeros(7)
            action[6] = gripper_close_action
            obs, _, _, _ = self._step_env(env, action)
            last_obs = obs

            # Check stop conditions for learned grasps
            if using_learned_grasp and grasp_width > 0:
                if 'robot0_gripper_qpos' in obs:
                    current_qpos = np.sum(np.abs(obs['robot0_gripper_qpos']))

                    # Stop if reached target width
                    # In robosuite: smaller qpos = more closed
                    if current_qpos <= target_qpos:
                        close_stop_reason = "width"
                        break

                    # Stop if gripper stalled (contact detected)
                    if prev_qpos is not None:
                        if abs(current_qpos - prev_qpos) < stall_eps:
                            stall_count += 1
                            if stall_count >= 3:
                                close_stop_reason = "stall"
                                break
                        else:
                            stall_count = 0
                    prev_qpos = current_qpos

        # Record close phase info
        final_qpos_raw = None
        if last_obs and 'robot0_gripper_qpos' in last_obs:
            final_qpos_raw = last_obs['robot0_gripper_qpos'].tolist()

        close_info = {
            "stop_reason": close_stop_reason,
            "target_qpos": target_qpos,
            "final_qpos": prev_qpos if prev_qpos is not None else -1,
            "final_qpos_raw": final_qpos_raw,  # Raw qpos vector for analysis
            "steps": step + 1 if 'step' in dir() else close_steps,
            # Include grasp metadata for analysis
            "using_learned_grasp": using_learned_grasp,
            "grasp_width": float(grasp_width),
            "grasp_strategy": grasp_strategy,
        }

        # Phase 3: Lift
        current_pose = self._get_gripper_pose(env, last_obs)
        if current_pose is not None:
            lift_target = current_pose.copy()
            lift_target[2] += self.lift_height
            # Keep gripper at same closing level during lift
            self.controller.set_target(lift_target, gripper=gripper_close_action)

            lift_steps = 30
            for step in range(lift_steps):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)
                last_obs = obs

        # Check if grasp succeeded
        final_pose = self._get_gripper_pose(env, last_obs)
        gripper_closed = self._check_gripper_closed(last_obs)

        # Get gripper width for debugging
        if last_obs and 'robot0_gripper_qpos' in last_obs:
            gripper_width = np.sum(np.abs(last_obs['robot0_gripper_qpos']))
        else:
            gripper_width = -1

        # Verify object actually moved with gripper (lifted)
        # Use the initial position recorded AFTER getting live position
        object_lifted = self._verify_object_lifted(env, obj_name, obj_pose[2])

        # Build result info
        result_info = {
            "steps_taken": steps_taken,
            "final_pose": final_pose,
            "xy_refinement": xy_refine_info,
            "grasp_point_info": grasp_point_info,
            "descent_info": descent_info,
            "close_info": close_info,  # Width-based close diagnostics
            "gripper_closed": gripper_closed,
            "gripper_width": gripper_width,
            "object_lifted": object_lifted,
            "initial_obj_z": obj_pose[2],
        }

        # Success requires both gripper closed AND object lifted
        if gripper_closed and object_lifted:
            result_info["object_grasped"] = obj_name
            # Compute gripper-to-object offset for accurate placement
            # This offset is critical for rim grasps where object center != gripper center
            try:
                obj_body_id = env.sim.model.body_name2id(obj_name)
                current_obj_pos = env.sim.data.body_xpos[obj_body_id].copy()
                if final_pose is not None:
                    # offset = object_center - gripper_center (in XY only)
                    grasp_offset_xy = current_obj_pos[:2] - final_pose[:2]
                    result_info["grasp_offset_xy"] = grasp_offset_xy.tolist()
            except (ValueError, AttributeError):
                pass  # Can't compute offset, will use default (0,0)
            return SkillResult(success=True, info=result_info)
        elif not gripper_closed:
            result_info["error_msg"] = "Grasp failed - gripper did not close on object"
            return SkillResult(success=False, info=result_info)
        else:
            result_info["error_msg"] = "Grasp failed - object not lifted (gripper closed on air)"
            return SkillResult(success=False, info=result_info)

    def _xy_refinement_phase(
        self,
        env,
        current_pose: np.ndarray,
        obj_pose: np.ndarray,
        last_obs: Optional[dict],
    ) -> dict:
        """XY refinement phase to correct ApproachSkill residual error.

        Uses the existing PD controller to servo the gripper directly
        above the object before lowering.

        Args:
            env: Environment
            current_pose: Current gripper pose (7D)
            obj_pose: Target object pose (7D)
            last_obs: Last observation

        Returns:
            Dict with refinement metrics:
                - xy_error_before: XY error before refinement (m)
                - xy_error_after: XY error after refinement (m)
                - steps: Steps used for refinement
                - converged: Whether refinement converged
                - final_pose: Gripper pose after refinement
                - last_obs: Last observation
        """
        # Compute initial XY error
        xy_error_before = np.linalg.norm(current_pose[:2] - obj_pose[:2])

        # Skip refinement if already aligned
        if xy_error_before <= self.xy_refine_threshold:
            return {
                "xy_error_before": xy_error_before,
                "xy_error_after": xy_error_before,
                "steps": 0,
                "converged": True,
                "skipped": True,
                "final_pose": current_pose,
                "last_obs": last_obs,
            }

        # Create target pose: same Z and orientation, but XY aligned with object
        refine_target = current_pose.copy()
        refine_target[0] = obj_pose[0]  # Target X
        refine_target[1] = obj_pose[1]  # Target Y
        # Keep current Z and orientation

        self.controller.set_target(refine_target, gripper=-1.0)  # Keep gripper open (inverted)

        steps = 0
        prev_error = xy_error_before
        no_improvement_count = 0

        for step in range(self.xy_refine_max_steps):
            steps += 1

            if current_pose is None:
                break

            # Check if converged
            xy_error = np.linalg.norm(current_pose[:2] - obj_pose[:2])
            if xy_error <= self.xy_refine_threshold:
                break

            # Check for stalled progress (oscillation guard)
            improvement = prev_error - xy_error
            if improvement < self.xy_refine_min_improvement:
                no_improvement_count += 1
                if no_improvement_count >= 5:
                    # Stalled - stop refinement
                    break
            else:
                no_improvement_count = 0
            prev_error = xy_error

            # Compute and apply action using PD controller
            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)
            last_obs = obs

        # Compute final XY error
        xy_error_after = np.linalg.norm(current_pose[:2] - obj_pose[:2]) if current_pose is not None else xy_error_before
        converged = xy_error_after <= self.xy_refine_threshold

        return {
            "xy_error_before": xy_error_before,
            "xy_error_after": xy_error_after,
            "steps": steps,
            "converged": converged,
            "skipped": False,
            "final_pose": current_pose,
            "last_obs": last_obs,
        }

    def _check_gripper_closed(self, obs: dict) -> bool:
        """Check if gripper successfully grasped something.

        Heuristic: gripper width should be non-zero but less than fully open.
        For small objects (~4cm), we need tighter thresholds to distinguish
        between grasping an object vs closing on air.

        Robosuite gripper qpos values:
        - Fully open (action=-1): width ~0.08
        - Fully closed on nothing (action=+1): width ~0.001
        - Closed on small object (~4cm): width ~0.02-0.04
        """
        if obs is None:
            return True  # Assume success if can't check

        if 'robot0_gripper_qpos' in obs:
            gripper_qpos = obs['robot0_gripper_qpos']
            # Non-zero gripper width indicates something is grasped
            width = np.sum(np.abs(gripper_qpos))
            # Threshold range for successful grasp:
            # - Fully closed on nothing: ~0.001 (too small - no object)
            # - Closed on small object (~4cm): ~0.01-0.04 (good)
            # - Fully open: ~0.08 (too large - not closed)
            # Lower threshold to 0.003 - bowl rim grasps can be very tight
            # Values of 0.0038-0.0050 indicate partial rim grasp
            return 0.003 < width < 0.07

        # Can't verify without gripper data - return False to be conservative
        return False

    def _verify_object_lifted(self, env, obj_name: str, initial_z: float) -> bool:
        """Verify the object was actually lifted by checking its Z position.

        Args:
            env: Environment
            obj_name: Name of the object
            initial_z: Initial Z position of the object

        Returns:
            True if object Z increased by at least lift_height/2
        """
        try:
            # Get current object position from simulator
            # Handle both "obj_name" and "obj_name_main" naming conventions
            body_name = obj_name
            try:
                body_id = env.sim.model.body_name2id(body_name)
            except ValueError:
                # Try without _main suffix
                if body_name.endswith('_main'):
                    body_name = body_name[:-5]
                else:
                    body_name = body_name + '_main'
                try:
                    body_id = env.sim.model.body_name2id(body_name)
                except ValueError:
                    # Can't find object - return False to be conservative
                    return False

            current_z = env.sim.data.body_xpos[body_id][2]

            # Object should have lifted by at least 2cm (relaxed from lift_height/2)
            # The lift check is a sanity check, not a precise measure
            min_lift = 0.02  # 2cm minimum lift
            lifted = (current_z - initial_z) > min_lift

            return lifted

        except Exception:
            # If we can't verify, return False to be conservative
            return False
    
    def _get_live_object_position(self, env, obj_name: str) -> Optional[np.ndarray]:
        """Get current object position directly from simulator.

        This bypasses world_state which may have stale data from before
        physics settling.

        Args:
            env: Environment with simulator
            obj_name: Object name to look up

        Returns:
            Current position (3D) or None if not found
        """
        try:
            # Try exact name first
            body_name = obj_name
            try:
                body_id = env.sim.model.body_name2id(body_name)
            except ValueError:
                # Try without _main suffix
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

    def _compute_grasp_pose(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        world_state: WorldState,
        gripper_pose: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """Compute optimal 6-DoF grasp pose using the configured GraspSelector.

        For learned selectors (Contact-GraspNet), returns full 6-DoF pose.
        For heuristic selectors, returns position with default orientation.

        Args:
            obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
            obj_name: Object name for lookup
            world_state: World state with object metadata
            gripper_pose: Current gripper pose (used to determine offset direction)
            point_cloud: Optional point cloud (N, 3) for learned selectors
            depth_image: Optional depth image for point cloud generation
            camera_intrinsics: Optional camera K matrix (3x3)

        Returns:
            Tuple of (position, orientation, gripper_width, info_dict):
            - position: (3,) grasp position [x, y, z]
            - orientation: (4,) quaternion [qw, qx, qy, qz]
            - gripper_width: predicted gripper width (meters)
            - info: dict with grasp metadata
        """
        # Get object type from world state
        obj_type = None
        if obj_name in world_state.objects:
            obj_type = world_state.objects[obj_name].object_type

        # Fallback: infer type from name
        if obj_type is None:
            obj_name_lower = obj_name.lower()
            for hollow_type in self.HOLLOW_OBJECTS:
                if hollow_type in obj_name_lower:
                    obj_type = hollow_type
                    break

        # Get approach strategy from world state
        approach_strategy = getattr(world_state, 'approach_strategy', 'top_down')

        # Use grasp selector to compute grasp pose
        grasp_pose, info = self.grasp_selector.select_grasp(
            obj_pose=obj_pose,
            obj_name=obj_name,
            obj_type=obj_type,
            gripper_pose=gripper_pose,
            approach_strategy=approach_strategy,
            point_cloud=point_cloud,
            depth_image=depth_image,
            camera_intrinsics=camera_intrinsics,
        )

        # Extract position
        position = grasp_pose.position.copy()

        # Extract orientation (quaternion [w, x, y, z])
        if grasp_pose.orientation is not None and len(grasp_pose.orientation) == 4:
            orientation = grasp_pose.orientation.copy()
        else:
            # Fallback to default gripper-down orientation
            orientation = self.DEFAULT_ORIENTATION.copy()

        # Extract gripper width
        gripper_width = grasp_pose.gripper_width

        # Add grasp strategy info
        info["grasp_strategy_used"] = grasp_pose.strategy
        info["grasp_confidence"] = grasp_pose.confidence

        # If using learned selector (not heuristic), update approach strategy
        if grasp_pose.strategy in ("contact_graspnet", "giga"):
            # Compute approach direction from grasp pose
            if grasp_pose.approach_direction is not None:
                info["approach_direction"] = grasp_pose.approach_direction.tolist()
            # Mark that we're using learned grasp - affects servo behavior
            info["learned_grasp"] = True
        else:
            info["learned_grasp"] = False

        return position, orientation, gripper_width, info

    def _compute_grasp_point(
        self,
        obj_pose: np.ndarray,
        obj_name: str,
        world_state: WorldState,
        gripper_pose: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal grasp point (legacy interface).

        This is a compatibility wrapper around _compute_grasp_pose that
        returns only position for backward compatibility.

        Args:
            obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
            obj_name: Object name for lookup
            world_state: World state with object metadata
            gripper_pose: Current gripper pose

        Returns:
            Tuple of (grasp_target_xyz, info_dict)
        """
        position, orientation, gripper_width, info = self._compute_grasp_pose(
            obj_pose, obj_name, world_state, gripper_pose
        )
        # Store orientation and width in info for callers that want them
        info["grasp_orientation"] = orientation.tolist()
        info["grasp_gripper_width"] = gripper_width
        return position, info

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

        # Store grasp offset for accurate placement
        # This offset accounts for rim grasps where object center != gripper center
        if "grasp_offset_xy" in result.info:
            world_state.grasp_offset_xy = np.array(result.info["grasp_offset_xy"])
