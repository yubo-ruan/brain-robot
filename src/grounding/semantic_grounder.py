"""Qwen Semantic Grounder for Phase 3.

Maps task language to object IDs using Qwen.
"""

from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field

from .enriched_object import EnrichedObject
from .grounding_result import GroundingResult
from .grounding_prompts import (
    build_grounding_prompt,
    parse_grounding_output,
    validate_grounding,
    is_valid_region,
)


@dataclass
class GroundingMetrics:
    """Metrics for grounding performance."""

    total_attempts: int = 0
    parse_success: int = 0
    parse_failures: int = 0
    validation_success: int = 0
    validation_failures: int = 0
    ambiguous_cases: int = 0

    # Per-task tracking
    per_task: dict = field(default_factory=dict)

    def record_attempt(self, task_id: str):
        self.total_attempts += 1
        if task_id not in self.per_task:
            self.per_task[task_id] = {"attempts": 0, "success": 0, "ambiguous": 0}
        self.per_task[task_id]["attempts"] += 1

    def record_parse_failure(self, task_id: str):
        self.parse_failures += 1

    def record_parse_success(self, task_id: str):
        self.parse_success += 1

    def record_validation_failure(self, task_id: str):
        self.validation_failures += 1

    def record_validation_success(self, task_id: str):
        self.validation_success += 1
        self.per_task[task_id]["success"] += 1

    def record_ambiguous(self, task_id: str):
        self.ambiguous_cases += 1
        self.per_task[task_id]["ambiguous"] += 1

    def summary(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "parse_rate": self.parse_success / self.total_attempts if self.total_attempts > 0 else 0,
            "validation_rate": self.validation_success / self.parse_success if self.parse_success > 0 else 0,
            "ambiguous_rate": self.ambiguous_cases / self.total_attempts if self.total_attempts > 0 else 0,
            "per_task": self.per_task,
        }


class QwenSemanticGrounder:
    """Uses Qwen to ground task language to object IDs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        """Initialize grounder.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Max retries on parse failure
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

        print(f"Loading {self.model_name} for grounding...")

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

        print(f"Qwen grounder loaded on {self.device}")
        self._loaded = True

    def share_model(self, model, processor):
        """Share model instance with another component (e.g., planner).

        Args:
            model: Pre-loaded Qwen model
            processor: Pre-loaded processor
        """
        self.model = model
        self.processor = processor
        self._loaded = True

    def ground(
        self,
        task_description: str,
        objects: List[EnrichedObject],
        metrics: Optional[GroundingMetrics] = None,
        task_id: str = "unknown",
    ) -> GroundingResult:
        """Ground task description to object IDs.

        Args:
            task_description: Natural language task
            objects: List of enriched objects in scene
            metrics: Optional metrics tracker
            task_id: Task identifier for metrics

        Returns:
            GroundingResult with source and target
        """
        self.load_model()

        if metrics:
            metrics.record_attempt(task_id)

        # Build prompt
        prompt = build_grounding_prompt(task_description, objects, include_few_shot=True)

        # Get valid object IDs
        valid_ids = {obj.id for obj in objects}

        # Try with retries
        last_error = None
        last_raw_output = None

        for attempt in range(self.max_retries + 1):
            # Query Qwen
            raw_output = self._query_qwen(prompt)
            last_raw_output = raw_output

            # Parse output
            result, parse_error = parse_grounding_output(raw_output)
            if parse_error:
                last_error = f"Parse error: {parse_error}"
                if metrics and attempt == self.max_retries:
                    metrics.record_parse_failure(task_id)
                continue

            if metrics:
                metrics.record_parse_success(task_id)

            # Validate
            valid, validation_error = validate_grounding(result, valid_ids)
            if not valid:
                last_error = f"Validation error: {validation_error}"
                if metrics and attempt == self.max_retries:
                    metrics.record_validation_failure(task_id)
                continue

            if metrics:
                metrics.record_validation_success(task_id)

            # Detect ambiguity
            ambiguous, alternatives = self._detect_ambiguity(
                task_description, objects, result["source_object"]
            )

            if ambiguous and metrics:
                metrics.record_ambiguous(task_id)

            return GroundingResult(
                source_object=result["source_object"],
                target_location=result["target_location"],
                confidence=result.get("confidence", "medium"),
                reasoning=result.get("reasoning", ""),
                raw_output=raw_output,
                prompt=prompt,
                ambiguous=ambiguous,
                alternative_sources=alternatives,
                valid=True,
            )

        # All retries exhausted
        return GroundingResult(
            source_object="",
            target_location="",
            raw_output=last_raw_output or "",
            prompt=prompt,
            valid=False,
            error=last_error,
        )

    def _query_qwen(self, prompt: str) -> str:
        """Query Qwen model with prompt."""
        import torch

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

    def _detect_ambiguity(
        self,
        task: str,
        objects: List[EnrichedObject],
        chosen_source: str,
    ) -> Tuple[bool, List[str]]:
        """Detect if grounding is ambiguous.

        Uses heuristic: if multiple objects share type+color and
        task doesn't contain spatial context that matches chosen object.

        Args:
            task: Task description
            objects: All objects in scene
            chosen_source: The source object Qwen chose

        Returns:
            (is_ambiguous, alternative_ids) tuple
        """
        # Find chosen object
        chosen = None
        for obj in objects:
            if obj.id == chosen_source:
                chosen = obj
                break

        if chosen is None:
            return False, []

        # Find alternatives with same type and color
        alternatives = [
            obj.id for obj in objects
            if obj.id != chosen_source
            and obj.type == chosen.type
            and obj.color == chosen.color
        ]

        if not alternatives:
            return False, []

        # Check if task contains spatial context that matches chosen
        # Use lowercase for matching
        task_lower = task.lower()
        spatial_text_lower = chosen.spatial_text.lower()

        # Extract key spatial phrase (e.g., "on cookie box" → "cookie box")
        spatial_key = spatial_text_lower.replace("on ", "").replace("inside ", "")

        if spatial_key in task_lower:
            # Task mentions the spatial context, so not ambiguous
            return False, []

        # Multiple similar objects and no disambiguating spatial context
        return True, alternatives
