"""Stove manipulation skills.

TurnOn and TurnOff skills for LIBERO stove controls.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..control.cartesian_pd import CartesianPDController
from ..config import SkillConfig


class TurnOnStoveSkill(Skill):
    """Turn on the stove by pressing the button.

    Sequence:
    1. Approach stove button
    2. Press button down
    3. Release

    Preconditions:
    - Stove exists in world state
    - Gripper is empty or can press

    Postconditions:
    - Stove button is pressed (stove is on)
    """

    name = "TurnOnStove"

    def __init__(
        self,
        max_steps: int = 150,
        press_depth: float = 0.02,  # How far to press button
        config: Optional[SkillConfig] = None,
    ):
        super().__init__(max_steps=max_steps, config=config)
        self.press_depth = press_depth
        self.controller = CartesianPDController.from_config(self.config)

    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Check turn on preconditions."""
        stove = args.get("stove") or args.get("target")

        if stove is None:
            return False, "Missing 'stove' argument"

        # Check stove exists (could be stove or stove_button)
        stove_objs = [n for n in world_state.objects if 'stove' in n.lower()]
        if not stove_objs:
            return False, f"Stove not found in world state"

        return True, "OK"

    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        return True, "OK"

    def _get_button_position(self, world_state: WorldState) -> Optional[np.ndarray]:
        """Get stove button position."""
        # Look for button specifically
        button_objs = [n for n in world_state.objects if 'button' in n.lower() and 'stove' in n.lower()]
        if button_objs:
            return world_state.get_object_position(button_objs[0])

        # Fallback to stove position with offset
        stove_objs = [n for n in world_state.objects if 'stove' in n.lower() and 'burner' not in n.lower()]
        if stove_objs:
            pos = world_state.get_object_position(stove_objs[0])
            if pos is not None:
                # Button is typically on the front of stove
                button_pos = pos.copy()
                button_pos[1] += 0.1  # Offset toward robot
                return button_pos

        return None

    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        """Execute stove turn on: approach button and press."""
        steps_taken = 0

        # Get button position
        button_pos = self._get_button_position(world_state)
        if button_pos is None:
            return SkillResult(
                success=False,
                info={"error_msg": "Cannot find stove button", "steps_taken": 0}
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

        # Phase 1: Approach above button
        approach_pos = button_pos.copy()
        approach_pos[2] += 0.08  # Above button

        approach_target = current_pose.copy()
        approach_target[:3] = approach_pos
        self.controller.set_target(approach_target, gripper=-1.0)  # Closed for pressing

        for step in range(self.max_steps // 3):
            steps_taken += 1
            if current_pose is None:
                break

            if self.controller.is_at_target(current_pose, pos_threshold=0.02):
                break

            action = self.controller.compute_action(current_pose)
            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)

        # Phase 2: Move down to button
        if current_pose is not None:
            press_target = current_pose.copy()
            press_target[:3] = button_pos
            press_target[2] += 0.02  # Slightly above button surface

            self.controller.set_target(press_target, gripper=-1.0)

            for step in range(self.max_steps // 3):
                steps_taken += 1
                if current_pose is None:
                    break

                if self.controller.is_at_target(current_pose, pos_threshold=0.015):
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Phase 3: Press button (move down slightly)
        if current_pose is not None:
            press_target = current_pose.copy()
            press_target[2] -= self.press_depth

            self.controller.set_target(press_target, gripper=-1.0)

            for step in range(30):
                steps_taken += 1
                if current_pose is None:
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        # Phase 4: Release (move up)
        if current_pose is not None:
            release_target = current_pose.copy()
            release_target[2] += 0.05

            self.controller.set_target(release_target, gripper=-1.0)

            for step in range(30):
                steps_taken += 1
                if current_pose is None:
                    break

                action = self.controller.compute_action(current_pose)
                obs, _, _, _ = self._step_env(env, action)
                current_pose = self._get_gripper_pose(env, obs)

        return SkillResult(
            success=True,
            info={
                "steps_taken": steps_taken,
                "button_position": button_pos.tolist() if button_pos is not None else None,
            }
        )

    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after turning on stove."""
        world_state.stove_on = True


class TurnOffStoveSkill(TurnOnStoveSkill):
    """Turn off the stove (same as turn on - it's a toggle)."""

    name = "TurnOffStove"

    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Update world state after turning off stove."""
        world_state.stove_on = False
