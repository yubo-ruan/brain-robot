"""Qwen Skill Planner for Phase 2.

Uses Qwen2.5-VL to generate skill sequences from task descriptions.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .prompts import build_system_prompt, build_user_prompt, prepare_world_state_for_qwen
from .plan_validator import parse_qwen_output, validate_plan, validate_plan_semantics
from .skill_schema import get_skill_by_name
from .planner_metrics import PlannerMetrics
from ..world_model.state import WorldState
from ..config import SkillConfig


@dataclass
class PlanResult:
    """Result of planning attempt."""
    success: bool
    plan: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    raw_output: Optional[str]
    prompt: Optional[str]


class QwenSkillPlanner:
    """Skill-level planner using Qwen2.5-VL.

    Generates symbolic skill sequences, not low-level motion commands.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        """Initialize planner.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            max_retries: Max retries on parse/validation failure
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        """Load Qwen model (lazy loading)."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"Loading {self.model_name}...")

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print(f"Qwen loaded on {self.device}")
        self._loaded = True

    def plan(
        self,
        task_description: str,
        world_state: WorldState,
        metrics: Optional[PlannerMetrics] = None,
        task_id: str = "unknown",
    ) -> PlanResult:
        """Generate skill plan for task.

        Args:
            task_description: Natural language task
            world_state: Current world state
            metrics: Optional metrics tracker
            task_id: Task identifier for metrics

        Returns:
            PlanResult with plan or error
        """
        self.load_model()

        if metrics:
            metrics.record_attempt(task_id)

        # Build prompt
        system_prompt = build_system_prompt(include_schema=True)
        world_state_dict = prepare_world_state_for_qwen(world_state)
        user_prompt = build_user_prompt(
            task_description=task_description,
            world_state_dict=world_state_dict,
            include_few_shot=True,
        )

        full_prompt = system_prompt + "\n\n" + user_prompt

        # Try with retries
        last_error = None
        last_raw_output = None

        for attempt in range(self.max_retries + 1):
            # Query Qwen
            raw_output = self._query_qwen(full_prompt)
            last_raw_output = raw_output

            # Parse output
            plan, parse_error = parse_qwen_output(raw_output)
            if parse_error:
                last_error = f"Parse error: {parse_error}"
                if metrics and attempt == self.max_retries:
                    metrics.record_parse_failure(task_id, full_prompt, raw_output, parse_error)
                continue

            if metrics:
                metrics.record_parse_success(task_id)

            # Validate plan structure
            valid, validation_error = validate_plan(plan)
            if not valid:
                last_error = f"Validation error: {validation_error}"
                if metrics and attempt == self.max_retries:
                    metrics.record_validation_failure(task_id, plan, validation_error)
                continue

            # Validate plan semantics
            valid, semantic_error = validate_plan_semantics(plan, world_state_dict)
            if not valid:
                last_error = f"Semantic error: {semantic_error}"
                if metrics and attempt == self.max_retries:
                    metrics.record_validation_failure(task_id, plan, semantic_error)
                continue

            if metrics:
                metrics.record_validation_success(task_id)

            return PlanResult(
                success=True,
                plan=plan,
                error=None,
                raw_output=raw_output,
                prompt=full_prompt,
            )

        # All retries exhausted
        return PlanResult(
            success=False,
            plan=None,
            error=last_error,
            raw_output=last_raw_output,
            prompt=full_prompt,
        )

    def _query_qwen(self, prompt: str) -> str:
        """Query Qwen model with prompt.

        Args:
            prompt: Full prompt including system and user parts

        Returns:
            Raw model output string
        """
        import torch

        # For skill planning, we use text-only (no image needed)
        # The world state provides all necessary information
        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response

    def execute_plan(
        self,
        plan: List[Dict[str, Any]],
        env,
        world_state: WorldState,
        config: Optional[SkillConfig] = None,
        metrics: Optional[PlannerMetrics] = None,
        task_id: str = "unknown",
        logger=None,
        perception=None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a skill plan.

        Args:
            plan: List of skill calls
            env: Environment
            world_state: Current world state
            config: Skill configuration
            metrics: Optional metrics tracker
            task_id: Task identifier
            logger: Optional episode logger
            perception: Optional perception module for updates between skills

        Returns:
            (success, info) tuple
        """
        config = config or SkillConfig()

        steps_taken = 0
        skill_results = []

        for i, step in enumerate(plan):
            skill_name = step["skill"]
            args = step.get("args", {})

            # Update perception before each skill (critical for accuracy)
            if perception is not None:
                perc_result = perception.perceive(env)
                world_state.update_from_perception(perc_result)

            # Get skill instance
            skill = get_skill_by_name(skill_name, config)
            if skill is None:
                error = f"Unknown skill: {skill_name}"
                if metrics:
                    metrics.record_execution_failure(task_id, plan, i, error)
                return False, {"error": error, "failed_step": i}

            # Execute skill
            result = skill.run(env, world_state, args)

            # Log if available
            if logger:
                logger.log_skill(skill_name, args, result)

            skill_results.append({
                "skill": skill_name,
                "args": args,
                "success": result.success,
                "info": result.info,
            })

            steps_taken += result.info.get("steps_taken", 0)

            if not result.success:
                error = result.info.get("error_msg", "Unknown error")
                if metrics:
                    metrics.record_execution_failure(task_id, plan, i, error)
                return False, {
                    "error": error,
                    "failed_step": i,
                    "failed_skill": skill_name,
                    "steps_taken": steps_taken,
                    "skill_results": skill_results,
                }

        if metrics:
            metrics.record_execution_success(task_id)

        return True, {
            "steps_taken": steps_taken,
            "skill_results": skill_results,
        }
