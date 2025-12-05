"""
Qwen2.5-VL-7B Vision-Language Model Planner.
Acts as the "Prefrontal Cortex" - high-level planning and reasoning.
"""

import torch
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, Optional, Any
import numpy as np

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEEDBACK_PROMPT_TEMPLATE


class QwenVLPlanner:
    """
    Vision-Language Model for high-level robot planning.
    Outputs relative motion commands in JSON format.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading {model_name}...")

        # Load model with appropriate settings for 7B
        # Use AutoModelForVision2Seq which handles Qwen2.5-VL correctly
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"Qwen2.5-VL loaded on {device}")

        # Track conversation for feedback
        self.previous_plan = None
        self.previous_phase = "approach"

    def plan(
        self,
        image: np.ndarray,
        task_description: str,
        gripper_state: str = "open",
        steps_since_plan: int = 0,
        feedback: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate motion plan from image and task.

        Args:
            image: RGB image array (H, W, 3)
            task_description: Natural language task description
            gripper_state: "open" or "closed"
            steps_since_plan: Number of steps since last plan
            feedback: Optional feedback from previous execution

        Returns:
            plan: Dictionary with observation, plan, and reasoning
        """
        # Convert image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image

        # Build prompt
        if feedback is not None:
            user_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
                task_description=task_description,
                previous_plan=json.dumps(self.previous_plan, indent=2) if self.previous_plan else "None",
                result=feedback.get("result", "unknown"),
                gripper_state=gripper_state,
                success_text="succeeded" if feedback.get("success", False) else "failed",
                feedback_text=feedback.get("text", ""),
            )
        else:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                task_description=task_description,
                gripper_state=gripper_state,
                steps_since_plan=steps_since_plan,
                previous_phase=self.previous_phase,
            )

        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ]},
        ]

        # Process with Qwen2.5-VL
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode response
        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        # Parse JSON
        plan = self._parse_json_response(response)

        # Update state
        self.previous_plan = plan
        if "plan" in plan and "phase" in plan["plan"]:
            self.previous_phase = plan["plan"]["phase"]

        return plan

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from model response."""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return self._default_plan()

            json_str = response[json_start:json_end]
            plan = json.loads(json_str)

            # Validate required fields
            if "plan" not in plan:
                return self._default_plan()

            return plan

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response was: {response}")
            return self._default_plan()

    def _default_plan(self) -> Dict[str, Any]:
        """Return safe default plan if parsing fails."""
        return {
            "observation": {
                "target_object": "unknown",
                "gripper_position": "unknown",
                "distance_to_target": "far",
                "obstacles": [],
            },
            "plan": {
                "phase": "approach",
                "movements": [
                    {"direction": "forward", "speed": "slow", "steps": 1}
                ],
                "gripper": "maintain",
                "confidence": 0.5,
            },
            "reasoning": "Default plan due to parsing failure",
        }

    def reset(self):
        """Reset planner state for new episode."""
        self.previous_plan = None
        self.previous_phase = "approach"
