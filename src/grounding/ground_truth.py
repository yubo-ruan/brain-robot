"""Ground-truth grounding labels for LIBERO spatial tasks.

These labels are derived from the BDDL goal specifications in LIBERO.
They define the canonical correct source and target for each task.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class GroundingGroundTruth:
    """Ground truth for a single task."""
    task_id: int
    task_suite: str
    task_description: str
    true_source: str  # Canonical object ID (without _main suffix in some cases)
    true_target: str
    source_spatial_key: str  # Key phrase that should appear in spatial context
    notes: str = ""


# LIBERO Spatial Tasks Ground Truth
# Derived from BDDL files: goal is always (On akita_black_bowl_1 plate_1)
LIBERO_SPATIAL_GROUND_TRUTH: List[GroundingGroundTruth] = [
    GroundingGroundTruth(
        task_id=0,
        task_suite="libero_spatial",
        task_description="pick up the black bowl between the plate and the ramekin and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="between",  # Bowl 1 is in between_plate_ramekin_region
        notes="Bowl 1 is between plate and ramekin; Bowl 2 is next to ramekin",
    ),
    GroundingGroundTruth(
        task_id=1,
        task_suite="libero_spatial",
        task_description="pick up the black bowl next to the ramekin and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="ramekin",  # Bowl 1 is in next_to_ramekin_region
        notes="Bowl 1 is next to ramekin; Bowl 2 is next to cookie box",
    ),
    GroundingGroundTruth(
        task_id=2,
        task_suite="libero_spatial",
        task_description="pick up the black bowl from table center and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="table",  # Bowl 1 is at table_center
        notes="Bowl 1 is at table center; Bowl 2 is next to plate. Only task with single unambiguous bowl by position.",
    ),
    GroundingGroundTruth(
        task_id=3,
        task_suite="libero_spatial",
        task_description="pick up the black bowl on the cookie box and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="cookie",  # Bowl 1 is ON cookies_1
        notes="Bowl 1 is ON the cookie box; Bowl 2 is on cabinet top",
    ),
    GroundingGroundTruth(
        task_id=4,
        task_suite="libero_spatial",
        task_description="pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="drawer",  # Bowl 1 is INSIDE drawer
        notes="Bowl 1 is INSIDE the drawer; Bowl 2 is on cabinet top side",
    ),
    GroundingGroundTruth(
        task_id=5,
        task_suite="libero_spatial",
        task_description="pick up the black bowl on the ramekin and place it on the plate",
        true_source="akita_black_bowl_1_main",
        true_target="plate_1_main",
        source_spatial_key="ramekin",  # Bowl 1 is ON ramekin
        notes="Bowl 1 is ON the ramekin; Bowl 2 is on cookie box",
    ),
]


def get_ground_truth(task_suite: str, task_id: int) -> GroundingGroundTruth:
    """Get ground truth for a specific task.

    Args:
        task_suite: Task suite name (e.g., "libero_spatial")
        task_id: Task ID within the suite

    Returns:
        GroundingGroundTruth object

    Raises:
        ValueError: If task not found
    """
    if task_suite == "libero_spatial":
        for gt in LIBERO_SPATIAL_GROUND_TRUTH:
            if gt.task_id == task_id:
                return gt

    raise ValueError(f"No ground truth for {task_suite} task {task_id}")


def get_all_ground_truth(task_suite: str) -> List[GroundingGroundTruth]:
    """Get all ground truth entries for a task suite."""
    if task_suite == "libero_spatial":
        return LIBERO_SPATIAL_GROUND_TRUTH
    return []


def verify_grounding(
    grounding_source: str,
    grounding_target: str,
    ground_truth: GroundingGroundTruth,
) -> Dict[str, Any]:
    """Verify grounding output against ground truth.

    Args:
        grounding_source: Source object ID from grounder
        grounding_target: Target location from grounder
        ground_truth: Ground truth for this task

    Returns:
        Dict with:
            - source_correct: bool
            - target_correct: bool
            - both_correct: bool
            - details: str
    """
    source_correct = grounding_source == ground_truth.true_source
    target_correct = grounding_target == ground_truth.true_target

    details = []
    if not source_correct:
        details.append(f"Source mismatch: got '{grounding_source}', expected '{ground_truth.true_source}'")
    if not target_correct:
        details.append(f"Target mismatch: got '{grounding_target}', expected '{ground_truth.true_target}'")

    return {
        "source_correct": source_correct,
        "target_correct": target_correct,
        "both_correct": source_correct and target_correct,
        "details": "; ".join(details) if details else "OK",
    }
