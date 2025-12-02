"""Keyframe selection for efficient data collection.

Not every frame is equally useful for training. This module selects
informative keyframes at state transitions and regular intervals.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class KeyframeSelector:
    """Select informative frames for training data collection.

    Collects frames at:
    - Regular intervals (every N steps)
    - State transitions (holding changed, gripper state changed)
    - Skill boundaries (start/end of skills)

    Usage:
        selector = KeyframeSelector(interval=20)

        for step in episode:
            if selector.should_collect(step, world_state, prev_world_state):
                collector.collect_frame(...)
    """

    # Minimum steps between keyframes
    interval: int = 20

    # Force collection on state transitions
    collect_on_holding_change: bool = True
    collect_on_gripper_change: bool = True

    # Gripper width change threshold (to detect open/close)
    gripper_threshold: float = 0.02

    # Internal state
    _last_keyframe_step: int = -100
    _last_holding: Optional[str] = None
    _last_gripper_width: float = 0.0

    def should_collect(
        self,
        step: int,
        holding: Optional[str] = None,
        gripper_width: float = 0.0,
        force: bool = False,
    ) -> bool:
        """Decide if current frame should be collected.

        Args:
            step: Current step in episode
            holding: Currently held object (or None)
            gripper_width: Current gripper opening
            force: Force collection regardless of conditions

        Returns:
            True if frame should be collected
        """
        collect = False
        reason = None

        # Check interval
        if step - self._last_keyframe_step >= self.interval:
            collect = True
            reason = "interval"

        # Check holding change (pick/place events)
        if self.collect_on_holding_change:
            if holding != self._last_holding:
                collect = True
                reason = f"holding_change: {self._last_holding} -> {holding}"

        # Check gripper change (open/close events)
        if self.collect_on_gripper_change:
            gripper_delta = abs(gripper_width - self._last_gripper_width)
            if gripper_delta > self.gripper_threshold:
                collect = True
                reason = f"gripper_change: {self._last_gripper_width:.3f} -> {gripper_width:.3f}"

        # Force collection
        if force:
            collect = True
            reason = "forced"

        # Update state if collecting
        if collect:
            self._last_keyframe_step = step
            self._last_holding = holding
            self._last_gripper_width = gripper_width

        return collect

    def reset(self):
        """Reset selector state for new episode."""
        self._last_keyframe_step = -100
        self._last_holding = None
        self._last_gripper_width = 0.0

    def force_collect_at_boundaries(
        self,
        step: int,
        is_episode_start: bool = False,
        is_episode_end: bool = False,
        is_skill_start: bool = False,
        is_skill_end: bool = False,
    ) -> bool:
        """Force collection at episode/skill boundaries.

        Always collect at:
        - Episode start (initial state)
        - Episode end (final state)
        - Skill transitions (pre/post skill execution)

        Args:
            step: Current step
            is_episode_start: True if this is first step of episode
            is_episode_end: True if this is last step of episode
            is_skill_start: True if skill is about to start
            is_skill_end: True if skill just finished

        Returns:
            True if should force collection
        """
        if is_episode_start or is_episode_end:
            return True
        if is_skill_start or is_skill_end:
            return True
        return False


class AdaptiveKeyframeSelector(KeyframeSelector):
    """Adaptive keyframe selection based on visual change.

    In addition to base selector logic, also collects when
    the scene has changed significantly (based on gripper motion).
    """

    # Gripper position change threshold (meters)
    position_threshold: float = 0.05

    _last_gripper_pos: Optional[np.ndarray] = None

    def should_collect(
        self,
        step: int,
        holding: Optional[str] = None,
        gripper_width: float = 0.0,
        gripper_position: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> bool:
        """Extended should_collect with position-based triggering."""

        # First check base conditions
        base_collect = super().should_collect(
            step, holding, gripper_width, force
        )

        if base_collect:
            if gripper_position is not None:
                self._last_gripper_pos = gripper_position.copy()
            return True

        # Check gripper position change
        if gripper_position is not None and self._last_gripper_pos is not None:
            dist = np.linalg.norm(gripper_position - self._last_gripper_pos)
            if dist > self.position_threshold:
                self._last_gripper_pos = gripper_position.copy()
                self._last_keyframe_step = step
                return True

        return False

    def reset(self):
        """Reset selector state for new episode."""
        super().reset()
        self._last_gripper_pos = None
