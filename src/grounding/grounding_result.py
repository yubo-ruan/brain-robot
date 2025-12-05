"""Grounding result dataclass."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class GroundingResult:
    """Result of semantic grounding with ambiguity tracking."""

    # Core grounding
    source_object: str                         # Object ID to manipulate
    target_location: str                       # Object ID or region to place on

    # Metadata
    confidence: str = "high"                   # "high" | "medium" | "low" (diagnostic only)
    reasoning: str = ""                        # Qwen's explanation

    # Raw data for logging
    raw_output: str = ""                       # Raw Qwen response
    prompt: str = ""                           # Full prompt sent to Qwen

    # Ambiguity tracking (diagnostic, not control)
    ambiguous: bool = False                    # Were there multiple valid choices?
    alternative_sources: List[str] = field(default_factory=list)
    alternative_targets: List[str] = field(default_factory=list)

    # Validation
    valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        return {
            "source_object": self.source_object,
            "target_location": self.target_location,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "ambiguous": self.ambiguous,
            "alternative_sources": self.alternative_sources,
            "alternative_targets": self.alternative_targets,
            "valid": self.valid,
            "error": self.error,
            # Don't include raw_output and prompt in episode logs (too large)
            # They're logged separately if needed
        }
