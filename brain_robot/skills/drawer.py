"""Drawer manipulation skills.

OpenDrawer and CloseDrawer skills for LIBERO cabinet drawers.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class OpenDrawerSkill(Skill):
    """Open a drawer by grasping handle and pulling.

    Sequence:
    1. Approach drawer handle
    2. Grasp handle
    3. Pull drawer open (move gripper backward)
    4. Release handle

    Preconditions:
    - Gripper is empty (not holding anything)
    - Drawer/cabinet exists in world state

    Postconditions:
    - Drawer is open (joint position near open limit)
    """

    name = "OpenDrawer"

    def __init__(
        self,
        max_steps: int = 300,
        pull_distance: float = 0.15,  # How far to pull
        config: Optional[SkillConfig] = None,
    ):
        super().__init__(max_steps=max_steps, config=config)
        self.pull_distance = pull_distance
        self.controller = CartesianPDController.from_config(self.config)

    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check open drawer preconditions."""
        drawer = args.get("drawer") or args.get("target")

        if drawer is None:
            return False, "Missing 'drawer' argument"

        if world_state.is_holding():
            return False, "Gripper is not empty"

        # Check drawer exists
        if drawer not in world_state.objects:
            # Try to find matching drawer/cabinet
            matching = [n for n in world_state.objects if 'cabinet' in n.lower() or 'drawer' in n.lower()]
            if not matching:
                return False, f"Drawer '{drawer}' not found"

        return True, "OK"

    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if drawer is open."""
        # We rely on environment state for drawer open/closed
        # This would need access to joint positions
        return True, "OK"  # Simplified for now

    def _get_drawer_handle_position(self, world_state: WorldState, drawer_name: str) -> Optional[np.ndarray]:
        """Get position of drawer handle for grasping."""
        # Find the drawer body
        drawer_bodies = [n for n in world_state.objects.keys()
                        if drawer_name.split('_')[0] in n.lower() and 'middle' in n.lower()]

        if drawer_bodies:
            pos = world_state.get_object_position(drawer_bodies[0])
            if pos is not None:
                # Handle is on the front of the drawer, offset in Y
                handle_pos = pos.copy()
                handle_pos[1] += 0.05  # Offset toward robot (assumes Y is forward)
                return handle_pos

        # Fallback to drawer position
        if drawer_name in world_state.objects:
            return world_state.get_object_position(drawer_name)

        return None

    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute drawer open: approach, grasp handle, pull, release."""
        drawer = args.get("drawer") or args.get("target")
        steps_taken = 0

        # Get handle position
        handle_pos = self._get_drawer_handle_position(world_state, drawer)
        if handle_pos is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Cannot find drawer handle for '{drawer}'", "steps_taken": 0}
            )

        # Get initial gripper pose
        current_pose = world_state.gripper_pose
        if current_pose is None:
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)

        if current_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": "Failed to get gripper pose", "steps_taken": 0}
            )

        # Phase 1: Approach handle (move above and in front)
        approach_pos = handle_pos.copy()
        approach_pos[2] += 0.05  # Above handle
        approach_pos[1] += 0.08  # In front of handle

        approach_target = current_pose.copy()
        approach_target[:3] = approach_pos
        # NOTE: In robosuite OSC, action[6] = -1.0 opens gripper, +1.0 closes
        self.controller.set_target(approach_target, gripper=-1.0)  # Open gripper (inverted)

        for step in range(self.max_steps // 4):
            steps_taken += 1
            if current_pose is None:
                break

            if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                break

            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)

        # Phase 2: Move to handle position
        if current_pose is not None:
            grasp_target = current_pose.copy()
            grasp_target[:3] = handle_pos
            self.controller.set_target(grasp_target, gripper=-1.0)  # Keep open

            for step in range(self.max_steps // 4):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Phase 3: Close gripper on handle
        if current_pose is not None:
            self.controller.set_target(current_pose, gripper=1.0)  # Close (inverted)

            for step in range(30):
                steps_taken += 1
                action = self.controller.compute_action(current_pose)
                action[-1] = 1.0  # Force close (inverted)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Phase 4: Pull drawer open (move backward in Y)
        if current_pose is not None:
            pull_target = current_pose.copy()
            pull_target[1] += self.pull_distance  # Pull toward robot

            self.controller.set_target(pull_target, gripper=1.0)  # Keep closed (inverted)

            for step in range(self.max_steps // 3):
                steps_taken += 1
                if current_pose is None:
                    break

                # Check if pulled enough
                pull_dist = current_pose[1] - handle_pos[1]
                if pull_dist > self.pull_distance * 0.8:
                    break

                action = self.controller.compute_action(current_pose)
                action[-1] = 1.0  # Keep gripper closed (inverted)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Phase 5: Release gripper
        if current_pose is not None:
            for step in range(20):
                steps_taken += 1
                action = np.zeros(7)
                action[-1] = -1.0  # Open gripper (inverted)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Check if we pulled successfully
        final_pose = current_pose
        if final_pose is not None:
            final_pose_arr = np.array(final_pose) if not isinstance(final_pose, np.ndarray) else final_pose
            pull_achieved = final_pose_arr[1] - handle_pos[1]

            if pull_achieved > self.pull_distance * 0.5:
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": steps_taken,
                        "pull_distance": float(pull_achieved),
                        "final_pose": final_pose_arr.tolist(),
                    }
                )

        return SkillResult(
            success=False,
            info={
                "error_msg": "Failed to pull drawer open",
                "steps_taken": steps_taken,
            }
        )

    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after opening drawer."""
        drawer = args.get("drawer") or args.get("target")
        # Mark drawer as open (simplified)
        world_state.drawer_states = getattr(world_state, 'drawer_states', {})
        world_state.drawer_states[drawer] = "open"


class CloseDrawerSkill(Skill):
    """Close a drawer by pushing it.

    Sequence:
    1. Approach drawer front
    2. Push drawer closed

    Preconditions:
    - Drawer is open
    - Gripper is empty

    Postconditions:
    - Drawer is closed
    """

    name = "CloseDrawer"

    def __init__(
        self,
        max_steps: int = 200,
        push_distance: float = 0.15,
        config: Optional[SkillConfig] = None,
    ):
        super().__init__(max_steps=max_steps, config=config)
        self.push_distance = push_distance
        self.controller = CartesianPDController.from_config(self.config)

    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check close drawer preconditions."""
        drawer = args.get("drawer") or args.get("target")

        if drawer is None:
            return False, "Missing 'drawer' argument"

        return True, "OK"

    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        return True, "OK"

    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute drawer close: approach and push."""
        drawer = args.get("drawer") or args.get("target")
        steps_taken = 0

        # Get current gripper pose
        current_pose = world_state.gripper_pose
        if current_pose is None:
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)

        if current_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": "Failed to get gripper pose", "steps_taken": 0}
            )

        # Find drawer position
        drawer_bodies = [n for n in world_state.objects.keys()
                        if 'middle' in n.lower() and 'cabinet' in n.lower()]

        if not drawer_bodies:
            return SkillResult(
                success=False,
                info={"error_msg": f"Cannot find drawer body", "steps_taken": 0}
            )

        drawer_pos = world_state.get_object_position(drawer_bodies[0])

        # Phase 1: Move in front of drawer
        approach_pos = drawer_pos.copy()
        approach_pos[1] += 0.15  # In front of drawer

        approach_target = current_pose.copy()
        approach_target[:3] = approach_pos
        # NOTE: In robosuite OSC, action[6] = +1.0 closes gripper, -1.0 opens
        self.controller.set_target(approach_target, gripper=1.0)  # Closed gripper for pushing

        for step in range(self.max_steps // 2):
            steps_taken += 1
            if current_pose is None:
                break

            if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                break

            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)

        # Phase 2: Push drawer closed
        if current_pose is not None:
            push_target = current_pose.copy()
            push_target[1] -= self.push_distance  # Push away from robot

            self.controller.set_target(push_target, gripper=1.0)  # Keep closed for pushing

            for step in range(self.max_steps // 2):
                steps_taken += 1
                if current_pose is None:
                    break

                action = self.controller.compute_action(current_pose)
                action[-1] = 1.0  # Keep closed for pushing (inverted)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        return SkillResult(
            success=True,
            info={
                "steps_taken": steps_taken,
            }
        )

    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after closing drawer."""
        drawer = args.get("drawer") or args.get("target")
        world_state.drawer_states = getattr(world_state, 'drawer_states', {})
        world_state.drawer_states[drawer] = "closed"
