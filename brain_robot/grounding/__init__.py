"""Semantic grounding module for Phase 3.

Maps detected objects to task roles using Qwen.
"""

from .enriched_object import EnrichedObject, enrich_objects
from .grounding_result import GroundingResult
from .semantic_grounder import QwenSemanticGrounder, GroundingMetrics
from .grounding_prompts import GROUNDING_SYSTEM_PROMPT, build_grounding_prompt

__all__ = [
    "EnrichedObject",
    "enrich_objects",
    "GroundingResult",
    "QwenSemanticGrounder",
    "GroundingMetrics",
    "GROUNDING_SYSTEM_PROMPT",
    "build_grounding_prompt",
]
