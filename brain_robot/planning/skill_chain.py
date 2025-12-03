"""Skill chaining for multi-step task execution.

Chains multiple skills together for long-horizon tasks like libero_10.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import re

from ..skills import Skill, SkillResult, SKILL_REGISTRY
from ..skills import ApproachSkill, GraspSkill, MoveSkill, PlaceSkill
from ..skills import OpenDrawerSkill, CloseDrawerSkill, TurnOnStoveSkill
from ..world_model.state import WorldState
from ..perception.interface import PerceptionInterface
from ..config import SkillConfig


@dataclass
class SkillStep:
    """A single step in a skill chain."""
    skill_name: str
    args: Dict[str, Any]
    description: str = ""


@dataclass
class ChainResult:
    """Result of executing a skill chain."""
    success: bool
    steps_completed: int
    total_steps: int
    failed_step: Optional[SkillStep] = None
    failure_reason: str = ""
    total_env_steps: int = 0


class SkillChain:
    """Chains skills together for multi-step execution.

    Features:
    - Decomposes task descriptions into skill sequences
    - Executes skills with perception updates between steps
    - Tracks world state through the chain
    - Provides failure recovery hooks
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or SkillConfig()

    def decompose_task(self, task_description: str, world_state: WorldState) -> List[SkillStep]:
        """Decompose a task description into skill steps.

        Uses pattern matching for common LIBERO task formats.

        Args:
            task_description: Natural language task
            world_state: Current world state for grounding

        Returns:
            List of SkillSteps to execute
        """
        task_lower = task_description.lower()
        steps = []

        # Pattern: "put both X and Y on/in Z" (check before simpler patterns)
        both_match = re.search(r'put both (?:the )?(.+?) and (?:the )?(.+?) (on|in) (?:the )?(.+?)$', task_lower)
        if both_match:
            source1 = self._ground_object(both_match.group(1), world_state)
            source2 = self._ground_object(both_match.group(2), world_state)
            target = self._ground_object(both_match.group(4), world_state)
            chain1 = self._make_pick_place_chain(source1, target)
            chain2 = self._make_pick_place_chain(source2, target)
            return chain1 + chain2

        # Pattern: "put X in Y and close it" (drawer tasks)
        put_close_match = re.search(r'put (?:the )?(.+?) in (?:the )?(.+?) and close it', task_lower)
        if put_close_match:
            source = self._ground_object(put_close_match.group(1), world_state)
            target = self._ground_object(put_close_match.group(2), world_state)
            drawer = self._find_drawer(world_state, 'bottom' if 'bottom' in task_lower else 'top' if 'top' in task_lower else None)
            chain = self._make_pick_place_chain(source, target)
            chain.append(SkillStep("CloseDrawer", {"drawer": drawer}, "Close drawer"))
            return chain

        # Pattern: "put X in Y and close it" (microwave tasks)
        microwave_match = re.search(r'put (?:the )?(.+?) in (?:the )?microwave and close it', task_lower)
        if microwave_match:
            source = self._ground_object(microwave_match.group(1), world_state)
            microwave = self._find_object_by_type(world_state, 'microwave')
            chain = self._make_pick_place_chain(source, microwave)
            chain.append(SkillStep("CloseMicrowave", {"microwave": microwave}, "Close microwave"))
            return chain

        # Pattern: "turn on the stove and put X on it"
        stove_put_match = re.search(r'turn on (?:the )?stove and put (?:the )?(.+?) on it', task_lower)
        if stove_put_match:
            source = self._ground_object(stove_put_match.group(1), world_state)
            stove = self._find_stove(world_state)
            steps = [SkillStep("TurnOnStove", {"stove": stove}, "Turn on stove")]
            steps.extend(self._make_pick_place_chain(source, stove))
            return steps

        # Pattern: "put X on left/right plate and put Y on left/right plate"
        two_plates_match = re.search(
            r'put (?:the )?(.+?) on (?:the )?(left|right) plate and put (?:the )?(.+?) on (?:the )?(left|right) plate',
            task_lower
        )
        if two_plates_match:
            source1 = self._ground_object(two_plates_match.group(1), world_state)
            plate1 = self._find_plate(world_state, two_plates_match.group(2))
            source2 = self._ground_object(two_plates_match.group(3), world_state)
            plate2 = self._find_plate(world_state, two_plates_match.group(4))
            chain1 = self._make_pick_place_chain(source1, plate1)
            chain2 = self._make_pick_place_chain(source2, plate2)
            return chain1 + chain2

        # Pattern: "put X on plate and put Y to the left/right of plate"
        plate_side_match = re.search(
            r'put (?:the )?(.+?) on (?:the )?plate and put (?:the )?(.+?) to the (left|right) of (?:the )?plate',
            task_lower
        )
        if plate_side_match:
            source1 = self._ground_object(plate_side_match.group(1), world_state)
            plate = self._find_object_by_type(world_state, 'plate')
            source2 = self._ground_object(plate_side_match.group(2), world_state)
            # For "to the right of plate", use plate as target and skill will offset
            chain1 = self._make_pick_place_chain(source1, plate)
            chain2 = self._make_pick_place_chain(source2, plate)  # PlaceSkill will need offset
            return chain1 + chain2

        # Pattern: "pick up X and place it Y"
        pick_place_match = re.search(
            r'pick up (?:the )?(.+?) and place it (on|in) (?:the )?(.+)',
            task_lower
        )
        if pick_place_match:
            source = self._ground_object(pick_place_match.group(1), world_state)
            target = self._ground_object(pick_place_match.group(3), world_state)
            return self._make_pick_place_chain(source, target)

        # Pattern: "put X on/in Y" (simple single placement)
        put_match = re.search(r'put (?:the )?(.+?) (on|in|on top of|into) (?:the )?(.+?)$', task_lower)
        if put_match:
            source = self._ground_object(put_match.group(1), world_state)
            target = self._ground_object(put_match.group(3), world_state)
            return self._make_pick_place_chain(source, target)

        # Pattern: "open the drawer"
        if 'open' in task_lower and ('drawer' in task_lower or 'cabinet' in task_lower):
            level = 'middle' if 'middle' in task_lower else 'top' if 'top' in task_lower else 'bottom' if 'bottom' in task_lower else None
            drawer = self._find_drawer(world_state, level)
            return [SkillStep("OpenDrawer", {"drawer": drawer}, "Open drawer")]

        # Pattern: "close the drawer"
        if 'close' in task_lower and ('drawer' in task_lower or 'cabinet' in task_lower):
            level = 'middle' if 'middle' in task_lower else 'top' if 'top' in task_lower else 'bottom' if 'bottom' in task_lower else None
            drawer = self._find_drawer(world_state, level)
            return [SkillStep("CloseDrawer", {"drawer": drawer}, "Close drawer")]

        # Pattern: "turn on the stove" (standalone)
        if 'turn on' in task_lower and 'stove' in task_lower:
            stove = self._find_stove(world_state)
            return [SkillStep("TurnOnStove", {"stove": stove}, "Turn on stove")]

        # Pattern: "X and Y" - two independent actions (fallback)
        and_match = re.search(r'(.+) and (.+)', task_lower)
        if and_match:
            # Avoid matching "alphabet soup and..." which is an object name
            part1 = and_match.group(1).strip()
            part2 = and_match.group(2).strip()
            # Check if part1 looks like an action (starts with verb)
            action_verbs = ['put', 'pick', 'place', 'open', 'close', 'turn']
            if any(part1.startswith(v) for v in action_verbs):
                steps1 = self.decompose_task(part1, world_state)
                steps2 = self.decompose_task(part2, world_state)
                if steps1 and steps2:
                    return steps1 + steps2

        # Fallback: single pick-place
        source = self._extract_source(task_lower, world_state)
        target = self._extract_target(task_lower, world_state)
        if source and target:
            return self._make_pick_place_chain(source, target)

        return []  # Could not decompose

    def _make_pick_place_chain(self, source: str, target: str) -> List[SkillStep]:
        """Create standard pick-place skill chain."""
        return [
            SkillStep("ApproachObject", {"obj": source}, f"Approach {source}"),
            SkillStep("GraspObject", {"obj": source}, f"Grasp {source}"),
            SkillStep("MoveObjectToRegion", {"obj": source, "region": target}, f"Move to {target}"),
            SkillStep("PlaceObject", {"obj": source, "region": target}, f"Place on {target}"),
        ]

    def _ground_object(self, description: str, world_state: WorldState) -> str:
        """Ground object description to object ID in world state."""
        desc_lower = description.lower().strip()

        # Direct match on object type
        for obj_name in world_state.objects.keys():
            obj_lower = obj_name.lower()
            # Check if description matches object type
            if desc_lower in obj_lower:
                return obj_name
            # Check individual words
            for word in desc_lower.split():
                if word in obj_lower and len(word) > 3:
                    return obj_name

        # Return description as-is if no match (will fail in skill execution)
        return description

    def _find_drawer(self, world_state: WorldState, level: Optional[str] = None) -> str:
        """Find drawer object in world state."""
        for obj_name in world_state.objects.keys():
            if 'cabinet' in obj_name.lower() or 'drawer' in obj_name.lower():
                if level is None or level in obj_name.lower():
                    return obj_name
        return "drawer"

    def _find_stove(self, world_state: WorldState) -> str:
        """Find stove object in world state."""
        for obj_name in world_state.objects.keys():
            if 'stove' in obj_name.lower():
                return obj_name
        return "stove"

    def _find_object_by_type(self, world_state: WorldState, obj_type: str) -> str:
        """Find object by type name."""
        for obj_name in world_state.objects.keys():
            if obj_type.lower() in obj_name.lower():
                return obj_name
        return obj_type

    def _find_plate(self, world_state: WorldState, side: Optional[str] = None) -> str:
        """Find plate object, optionally by left/right side."""
        plates = [n for n in world_state.objects.keys() if 'plate' in n.lower()]
        if not plates:
            return "plate"

        if side is None or len(plates) == 1:
            return plates[0]

        # Sort by x position to determine left/right
        plate_positions = []
        for p in plates:
            pos = world_state.get_object_position(p)
            if pos is not None:
                plate_positions.append((p, pos[0]))  # (name, x_coord)

        if len(plate_positions) < 2:
            return plates[0]

        plate_positions.sort(key=lambda x: x[1])  # Sort by x
        if side == 'left':
            return plate_positions[0][0]  # Leftmost (smallest x)
        else:  # right
            return plate_positions[-1][0]  # Rightmost (largest x)

    def _extract_source(self, task_lower: str, world_state: WorldState) -> Optional[str]:
        """Extract source object from task."""
        # Common source keywords
        source_patterns = [
            r'pick up the (.+?) and',
            r'put the (.+?) on',
            r'put the (.+?) in',
            r'move the (.+?) to',
        ]
        for pattern in source_patterns:
            match = re.search(pattern, task_lower)
            if match:
                return self._ground_object(match.group(1), world_state)
        return None

    def _extract_target(self, task_lower: str, world_state: WorldState) -> Optional[str]:
        """Extract target object/location from task."""
        # Common target keywords
        target_patterns = [
            r'place it on the (.+?)$',
            r'place it in the (.+?)$',
            r'on the (.+?)$',
            r'in the (.+?)$',
            r'to the (.+?)$',
        ]
        for pattern in target_patterns:
            match = re.search(pattern, task_lower)
            if match:
                return self._ground_object(match.group(1), world_state)
        return None

    def execute_chain(
        self,
        steps: List[SkillStep],
        env,
        world_state: WorldState,
        perception: PerceptionInterface,
        re_perceive_between_steps: bool = True,
    ) -> ChainResult:
        """Execute a chain of skill steps.

        Args:
            steps: List of skill steps to execute
            env: LIBERO environment
            world_state: Current world state (will be updated in-place)
            perception: Perception interface for updates
            re_perceive_between_steps: Whether to update perception between skills

        Returns:
            ChainResult with execution summary
        """
        total_env_steps = 0

        for i, step in enumerate(steps):
            # Get skill from registry
            skill_class = SKILL_REGISTRY.get(step.skill_name)
            if skill_class is None:
                return ChainResult(
                    success=False,
                    steps_completed=i,
                    total_steps=len(steps),
                    failed_step=step,
                    failure_reason=f"Unknown skill: {step.skill_name}",
                    total_env_steps=total_env_steps,
                )

            # Instantiate and run skill
            skill = skill_class(config=self.config)
            result = skill.run(env, world_state, step.args)

            total_env_steps += result.info.get("steps_taken", 0)

            if not result.success:
                return ChainResult(
                    success=False,
                    steps_completed=i,
                    total_steps=len(steps),
                    failed_step=step,
                    failure_reason=result.info.get("error_msg", "Unknown error"),
                    total_env_steps=total_env_steps,
                )

            # Re-perceive between steps if enabled
            if re_perceive_between_steps and i < len(steps) - 1:
                perc_result = perception.perceive(env)
                world_state.update_from_perception(perc_result)

        return ChainResult(
            success=True,
            steps_completed=len(steps),
            total_steps=len(steps),
            total_env_steps=total_env_steps,
        )

    def execute_task(
        self,
        task_description: str,
        env,
        world_state: WorldState,
        perception: PerceptionInterface,
    ) -> ChainResult:
        """High-level API: decompose and execute a task.

        Args:
            task_description: Natural language task
            env: LIBERO environment
            world_state: Current world state
            perception: Perception interface

        Returns:
            ChainResult with execution summary
        """
        # Decompose task
        steps = self.decompose_task(task_description, world_state)

        if not steps:
            return ChainResult(
                success=False,
                steps_completed=0,
                total_steps=0,
                failure_reason=f"Could not decompose task: {task_description}",
            )

        # Execute chain
        return self.execute_chain(steps, env, world_state, perception)
