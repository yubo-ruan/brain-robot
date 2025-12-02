"""Per-layer failure classification for debugging learned+Qwen pipeline.

Classifies failures into:
- DETECTION: YOLO failed to detect required object
- TRACKING: Lost track of object mid-episode
- GROUNDING: Qwen picked wrong object or failed to parse
- PLANNING: Skill sequence was incorrect (N/A for hardcoded)
- SKILL: Skill execution failed despite correct perception/grounding

This helps diagnose where the pipeline is failing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


class FailureLayer(Enum):
    """Layers where failure can occur."""
    DETECTION = "detection"       # YOLO failed to detect object
    TRACKING = "tracking"         # Lost track mid-episode
    GROUNDING = "grounding"       # Qwen picked wrong object
    PLANNING = "planning"         # Wrong skill sequence
    SKILL = "skill"               # Skill execution failed
    UNKNOWN = "unknown"           # Cannot determine


@dataclass
class FailureClassification:
    """Classification of a single failure."""
    layer: FailureLayer
    description: str
    details: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer.value,
            "description": self.description,
            "details": self.details,
        }


class FailureClassifier:
    """Classifies failures in the learned+Qwen pipeline.

    Usage:
        classifier = FailureClassifier()

        # After detection
        if not detected_bowl:
            failure = classifier.classify_detection_failure("bowl", detected_objects)

        # After grounding
        if grounded_wrong_object:
            failure = classifier.classify_grounding_failure(expected, actual, task)

        # After skill
        if skill_failed:
            failure = classifier.classify_skill_failure(skill_name, error_msg, world_state)
    """

    def classify_detection_failure(
        self,
        expected_class: str,
        detected_objects: List[str],
        perception_result: Any = None,
    ) -> FailureClassification:
        """Classify detection failure.

        Args:
            expected_class: Class that should have been detected (e.g., "bowl")
            detected_objects: List of object IDs that were detected
            perception_result: Optional perception result for more details
        """
        detected_classes = [self._get_class(obj) for obj in detected_objects]

        if expected_class not in detected_classes:
            return FailureClassification(
                layer=FailureLayer.DETECTION,
                description=f"Failed to detect {expected_class}",
                details={
                    "expected_class": expected_class,
                    "detected_objects": detected_objects,
                    "detected_classes": detected_classes,
                },
            )

        return FailureClassification(
            layer=FailureLayer.UNKNOWN,
            description="Detection appears successful",
            details={"detected_objects": detected_objects},
        )

    def classify_tracking_failure(
        self,
        object_id: str,
        last_known_position: Optional[List[float]],
        current_objects: List[str],
    ) -> FailureClassification:
        """Classify tracking failure (object lost mid-episode).

        Args:
            object_id: ID of object that was lost
            last_known_position: Last known 3D position
            current_objects: Current detected objects
        """
        return FailureClassification(
            layer=FailureLayer.TRACKING,
            description=f"Lost track of {object_id}",
            details={
                "lost_object_id": object_id,
                "last_known_position": last_known_position,
                "current_objects": current_objects,
            },
        )

    def classify_grounding_failure(
        self,
        expected_source_class: str,
        expected_target_class: str,
        grounded_source: str,
        grounded_target: str,
        task_description: str,
        grounding_result: Any = None,
    ) -> FailureClassification:
        """Classify grounding failure (Qwen picked wrong object).

        Args:
            expected_source_class: Expected class for source (e.g., "bowl")
            expected_target_class: Expected class for target (e.g., "plate")
            grounded_source: Object ID Qwen picked as source
            grounded_target: Object ID Qwen picked as target
            task_description: Original task description
            grounding_result: Optional grounding result for more details
        """
        source_class = self._get_class(grounded_source)
        target_class = self._get_class(grounded_target)

        source_correct = source_class == expected_source_class
        target_correct = target_class == expected_target_class

        if not source_correct or not target_correct:
            errors = []
            if not source_correct:
                errors.append(f"source: got {source_class}, expected {expected_source_class}")
            if not target_correct:
                errors.append(f"target: got {target_class}, expected {expected_target_class}")

            return FailureClassification(
                layer=FailureLayer.GROUNDING,
                description=f"Qwen grounded wrong objects: {'; '.join(errors)}",
                details={
                    "task_description": task_description,
                    "expected_source_class": expected_source_class,
                    "expected_target_class": expected_target_class,
                    "grounded_source": grounded_source,
                    "grounded_target": grounded_target,
                    "source_class": source_class,
                    "target_class": target_class,
                    "source_correct": source_correct,
                    "target_correct": target_correct,
                },
            )

        return FailureClassification(
            layer=FailureLayer.UNKNOWN,
            description="Grounding appears correct",
            details={
                "grounded_source": grounded_source,
                "grounded_target": grounded_target,
            },
        )

    def classify_skill_failure(
        self,
        skill_name: str,
        error_msg: str,
        object_id: str,
        object_position: Optional[List[float]] = None,
    ) -> FailureClassification:
        """Classify skill execution failure.

        Args:
            skill_name: Name of failed skill (e.g., "ApproachObject")
            error_msg: Error message from skill
            object_id: Object being manipulated
            object_position: Position of object at failure time
        """
        return FailureClassification(
            layer=FailureLayer.SKILL,
            description=f"{skill_name} failed: {error_msg}",
            details={
                "skill_name": skill_name,
                "error_msg": error_msg,
                "object_id": object_id,
                "object_position": object_position,
            },
        )

    def classify_from_episode_result(
        self,
        physical_success: bool,
        semantic_source_correct: bool,
        semantic_target_correct: bool,
        failed_skill: Optional[str] = None,
        skill_error: Optional[str] = None,
        grounding_valid: bool = True,
        detection_count: int = 0,
    ) -> FailureClassification:
        """Classify failure from episode result summary.

        This is a convenience method for post-hoc classification.
        """
        if physical_success and semantic_source_correct and semantic_target_correct:
            return FailureClassification(
                layer=FailureLayer.UNKNOWN,
                description="Episode succeeded",
                details={"physical_success": True, "semantic_success": True},
            )

        # Check each layer in order
        if detection_count == 0:
            return FailureClassification(
                layer=FailureLayer.DETECTION,
                description="No objects detected",
                details={"detection_count": detection_count},
            )

        if not grounding_valid:
            return FailureClassification(
                layer=FailureLayer.GROUNDING,
                description="Grounding failed to parse or validate",
                details={"grounding_valid": False},
            )

        if not semantic_source_correct or not semantic_target_correct:
            return FailureClassification(
                layer=FailureLayer.GROUNDING,
                description="Grounded wrong object class",
                details={
                    "semantic_source_correct": semantic_source_correct,
                    "semantic_target_correct": semantic_target_correct,
                },
            )

        if failed_skill:
            return FailureClassification(
                layer=FailureLayer.SKILL,
                description=f"Skill {failed_skill} failed",
                details={
                    "failed_skill": failed_skill,
                    "skill_error": skill_error,
                },
            )

        return FailureClassification(
            layer=FailureLayer.UNKNOWN,
            description="Cannot determine failure layer",
            details={},
        )

    def _get_class(self, obj_id: str) -> str:
        """Extract class from object ID."""
        obj_lower = obj_id.lower()
        classes = ['cookie_box', 'ramekin', 'cabinet', 'drawer', 'bottle', 'stove', 'bowl', 'plate', 'mug', 'can']
        for cls in classes:
            if cls in obj_lower:
                return cls
        return "unknown"


class FailureStats:
    """Aggregate failure statistics across episodes."""

    def __init__(self):
        self.failures: List[FailureClassification] = []
        self.total_episodes = 0
        self.successful_episodes = 0

    def add_success(self):
        """Record successful episode."""
        self.total_episodes += 1
        self.successful_episodes += 1

    def add_failure(self, classification: FailureClassification):
        """Record failed episode with classification."""
        self.total_episodes += 1
        self.failures.append(classification)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        layer_counts = {}
        for f in self.failures:
            layer = f.layer.value
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": len(self.failures),
            "success_rate": self.successful_episodes / self.total_episodes if self.total_episodes > 0 else 0,
            "failure_by_layer": layer_counts,
            "failures": [f.to_dict() for f in self.failures],
        }

    def print_summary(self):
        """Print human-readable summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS")
        print("=" * 60)
        print(f"Total Episodes: {s['total_episodes']}")
        print(f"Successful: {s['successful_episodes']}")
        print(f"Failed: {s['failed_episodes']}")
        print(f"Success Rate: {s['success_rate']:.1%}")

        if s['failure_by_layer']:
            print("\nFailures by Layer:")
            for layer, count in sorted(s['failure_by_layer'].items()):
                pct = 100 * count / s['failed_episodes'] if s['failed_episodes'] > 0 else 0
                print(f"  {layer:15s}: {count:3d} ({pct:.0f}%)")
