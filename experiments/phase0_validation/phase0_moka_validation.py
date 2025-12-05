#!/usr/bin/env python3
"""
Phase 0: MOKA Validation on LIBERO Tasks

Validates that MOKA produces usable keypoints before building downstream systems.
- Runs MOKA pipeline on LIBERO spatial tasks
- Validates output schema
- Logs failure modes
- Produces comprehensive visual outputs for debugging
- Produces validation report
"""

import os
import sys
import json
import yaml
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime
from enum import Enum

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from PIL import Image
from easydict import EasyDict as edict

sys.path.insert(0, '/workspace/new_experiment')

# MOKA imports
from moka.gpt_utils import load_qwen_model, USE_GPT4V
from moka.vision.segmentation import get_scene_object_bboxes, get_segmentation_masks
from moka.planners.visual_prompt_utils import (
    propose_candidate_keypoints,
    annotate_visual_prompts,
    request_motion,
    request_plan,
)


# =============================================================================
# MOKA Output Schema (from spec v2.0)
# =============================================================================

class FailureMode(Enum):
    """Taxonomy of MOKA failure modes."""
    GOOD = "good"
    WRONG_OBJECT = "wrong_object"           # Detected wrong object
    MISSED_OBJECT = "missed_object"         # Failed to detect object
    WRONG_AFFORDANCE = "wrong_affordance"   # Wrong keypoint on correct object
    OCCLUSION = "occlusion"                 # Object occluded
    SCHEMA_VIOLATION = "schema_violation"   # Output doesn't match schema
    VLM_PARSE_ERROR = "vlm_parse_error"     # VLM output couldn't be parsed
    SEGMENTATION_FAIL = "segmentation_fail" # SAM failed


@dataclass
class MOKAOutput:
    """Typed MOKA output schema with validation."""
    # Keypoints in normalized image coordinates [0, 1]
    grasp_kp: Optional[Tuple[float, float]] = None
    function_kp: Optional[Tuple[float, float]] = None
    target_kp: Optional[Tuple[float, float]] = None

    # Discrete tiles
    pre_tile: Optional[str] = None
    target_tile: Optional[str] = None
    post_tile: Optional[str] = None

    # Heights
    pre_height: Optional[str] = None
    post_height: Optional[str] = None

    # Angle
    target_angle: Optional[str] = None

    # Raw JSON from VLM
    raw_json: Optional[Dict] = None

    # Validation
    is_valid: bool = False
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

    def validate(self, image_size: Tuple[int, int] = (512, 512)) -> bool:
        """Validate output against schema."""
        self.validation_errors = []
        w, h = image_size

        # Check keypoints are in valid range
        for name, kp in [("grasp_kp", self.grasp_kp),
                         ("function_kp", self.function_kp),
                         ("target_kp", self.target_kp)]:
            if kp is not None:
                u, v = kp
                if not (0 <= u <= w and 0 <= v <= h):
                    self.validation_errors.append(f"{name} out of bounds: ({u}, {v})")

        # Check heights are valid values
        valid_heights = {"same", "above", "below", "", None}
        if self.pre_height not in valid_heights:
            self.validation_errors.append(f"Invalid pre_height: {self.pre_height}")
        if self.post_height not in valid_heights:
            self.validation_errors.append(f"Invalid post_height: {self.post_height}")

        # Check tiles are valid format (letter + number, e.g., "a1", "b3")
        import re
        tile_pattern = re.compile(r'^[a-e][1-5]$')
        for name, tile in [("pre_tile", self.pre_tile),
                          ("target_tile", self.target_tile),
                          ("post_tile", self.post_tile)]:
            if tile is not None and tile != "":
                if not tile_pattern.match(tile):
                    self.validation_errors.append(f"Invalid {name}: {tile}")

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid


@dataclass
class ValidationResult:
    """Result of validating MOKA on a single task."""
    task_id: int
    task_description: str
    success: bool
    moka_output: Optional[MOKAOutput]
    failure_mode: FailureMode
    failure_details: str
    detected_objects: List[str]
    expected_objects: List[str]
    processing_time_sec: float

    # Detection info
    detection_boxes: Optional[List] = None
    detection_logits: Optional[List] = None

    # Candidate keypoints
    candidate_keypoints: Optional[Dict] = None

    # For pixel error analysis (if ground truth available)
    grasp_pixel_error: Optional[float] = None
    target_pixel_error: Optional[float] = None


# =============================================================================
# LIBERO Tasks Configuration - All 10 Spatial Tasks
# =============================================================================

LIBERO_SPATIAL_TASKS = {
    0: {
        "description": "pick up the black bowl on the stove and place it on the wooden cabinet",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_0_initial.png",
        "expected_objects": ["black bowl", "wooden cabinet"],
        "plan": [{
            "instruction": "Move the black bowl onto the wooden cabinet",
            "object_grasped": "black bowl",
            "object_unattached": "wooden cabinet",
            "motion_direction": "downward"
        }]
    },
    1: {
        "description": "pick up the black bowl on the plate and place it on the wooden cabinet",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_1_initial.png",
        "expected_objects": ["black bowl", "wooden cabinet"],
        "plan": [{
            "instruction": "Move the black bowl onto the wooden cabinet",
            "object_grasped": "black bowl",
            "object_unattached": "wooden cabinet",
            "motion_direction": "downward"
        }]
    },
    2: {
        "description": "pick up the black bowl next to the cookie box and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_2_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    3: {
        "description": "pick up the black bowl in the wooden cabinet and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_3_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    4: {
        "description": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_4_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    5: {
        "description": "pick up the black bowl on the ramekin and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_5_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    6: {
        "description": "pick up the black bowl next to the ramekin and place it on the stove",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_6_initial.png",
        "expected_objects": ["black bowl", "stove"],
        "plan": [{
            "instruction": "Move the black bowl onto the stove",
            "object_grasped": "black bowl",
            "object_unattached": "stove",
            "motion_direction": "downward"
        }]
    },
    7: {
        "description": "pick up the black bowl on the cookie box and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_7_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    8: {
        "description": "pick up the black bowl next to the plate and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_8_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
    9: {
        "description": "pick up the black bowl on the wooden cabinet and place it on the plate",
        "image": "/workspace/brain_robot/recordings/vlm_analysis/20251204_073201/task_9_initial.png",
        "expected_objects": ["black bowl", "plate"],
        "plan": [{
            "instruction": "Move the black bowl onto the plate",
            "object_grasped": "black bowl",
            "object_unattached": "plate",
            "motion_direction": "downward"
        }]
    },
}


# =============================================================================
# Visualization Functions
# =============================================================================

def save_detection_visualization(
    image: Image.Image,
    boxes: np.ndarray,
    logits: np.ndarray,
    phrases: List[str],
    output_path: Path
):
    """Save visualization of detection bounding boxes with confidence scores."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    colors = plt.cm.tab10(np.linspace(0, 1, len(boxes) if boxes is not None else 1))

    if boxes is not None and len(boxes) > 0:
        h, w = image.size[1], image.size[0]
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Box is in [cx, cy, w, h] normalized format
            cx, cy, bw, bh = box
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            box_w = bw * w
            box_h = bh * h

            rect = Rectangle((x1, y1), box_w, box_h,
                            linewidth=3, edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)

            # Add label with confidence
            label = f"{phrase} ({logit:.2f})"
            ax.text(x1, y1 - 5, label, fontsize=12, color='white',
                   bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))

    ax.set_title("Object Detection (GroundingDINO)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_segmentation_visualization(
    image: Image.Image,
    segmasks: Dict[str, np.ndarray],
    output_path: Path
):
    """Save visualization of segmentation masks."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    colors = plt.cm.tab10(np.linspace(0, 1, len(segmasks)))
    legend_patches = []

    for i, (obj_name, mask_data) in enumerate(segmasks.items()):
        # Extract mask array - handle both dict and ndarray formats
        if isinstance(mask_data, dict):
            mask = mask_data.get('mask', mask_data)
        else:
            mask = mask_data

        # Create colored overlay
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = [*colors[i][:3], 0.4]  # RGBA with alpha

        ax.imshow(colored_mask)

        # Find mask centroid for label
        if mask.sum() > 0:
            y_indices, x_indices = np.where(mask > 0)
            cx, cy = x_indices.mean(), y_indices.mean()
            ax.text(cx, cy, obj_name, fontsize=12, color='white',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor=colors[i][:3], alpha=0.8))

        legend_patches.append(mpatches.Patch(color=colors[i], label=obj_name))

    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
    ax.set_title("Segmentation Masks (SAM)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_candidate_keypoints_visualization(
    image: Image.Image,
    candidate_keypoints: Dict,
    output_path: Path
):
    """Save visualization of all candidate keypoints (P0-P5, Q0-Q5)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Draw grasped object keypoints (P)
    grasped_kps = candidate_keypoints.get('grasped')
    if grasped_kps is not None and len(grasped_kps) > 0:
        for i, kp in enumerate(grasped_kps):
            ax.scatter(kp[0], kp[1], c='red', s=100, marker='o', edgecolors='white', linewidths=2)
            ax.text(kp[0] + 10, kp[1], f'P{i}', fontsize=10, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Draw unattached object keypoints (Q)
    unattached_kps = candidate_keypoints.get('unattached')
    if unattached_kps is not None and len(unattached_kps) > 0:
        for i, kp in enumerate(unattached_kps):
            ax.scatter(kp[0], kp[1], c='blue', s=100, marker='o', edgecolors='white', linewidths=2)
            ax.text(kp[0] + 10, kp[1], f'Q{i}', fontsize=10, color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=10, label='P: Grasped object keypoints'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=10, label='Q: Target object keypoints'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title("All Candidate Keypoints (before VLM selection)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_grid_visualization(
    image: Image.Image,
    context_json: Dict,
    grid_size,  # Can be int or list [rows, cols]
    output_path: Path
):
    """Save visualization of tile grid with selected tiles highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Handle grid_size as list or int
    if isinstance(grid_size, (list, tuple)):
        grid_rows, grid_cols = grid_size[0], grid_size[1] if len(grid_size) > 1 else grid_size[0]
    else:
        grid_rows = grid_cols = grid_size

    h, w = image.size[1], image.size[0]
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    # Draw grid
    for i in range(grid_rows + 1):
        ax.axhline(y=i * cell_h, color='white', linewidth=0.5, alpha=0.5)
    for i in range(grid_cols + 1):
        ax.axvline(x=i * cell_w, color='white', linewidth=0.5, alpha=0.5)

    # Label grid cells
    letters = 'abcde'
    for row in range(grid_rows):
        for col in range(grid_cols):
            tile_name = f"{letters[col]}{row + 1}"
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            ax.text(cx, cy, tile_name, fontsize=8, color='white', alpha=0.5,
                   ha='center', va='center')

    # Highlight selected tiles
    def highlight_tile(tile_name, color, label):
        if tile_name and len(tile_name) == 2:
            col = ord(tile_name[0]) - ord('a')
            row = int(tile_name[1]) - 1
            if 0 <= col < grid_cols and 0 <= row < grid_rows:
                rect = Rectangle((col * cell_w, row * cell_h), cell_w, cell_h,
                                linewidth=3, edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                ax.text((col + 0.5) * cell_w, (row + 0.5) * cell_h, label,
                       fontsize=10, color=color, ha='center', va='center', weight='bold')

    highlight_tile(context_json.get('pre_contact_tile'), 'yellow', 'PRE')
    highlight_tile(context_json.get('target_tile'), 'green', 'TARGET')
    highlight_tile(context_json.get('post_contact_tile'), 'cyan', 'POST')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='yellow', alpha=0.5, label='Pre-contact tile'),
        mpatches.Patch(facecolor='green', alpha=0.5, label='Target tile'),
        mpatches.Patch(facecolor='cyan', alpha=0.5, label='Post-contact tile'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title(f"Tile Grid ({grid_cols}x{grid_rows})", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_visualization(
    original_image: Image.Image,
    annotated_image: Image.Image,
    motion_image: Image.Image,
    moka_output: MOKAOutput,
    context_json: Dict,
    output_path: Path
):
    """Save side-by-side comparison of original, annotated, and final motion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    # Annotated image with all keypoints
    axes[1].imshow(annotated_image)
    axes[1].set_title("Annotated (all candidates)", fontsize=12)
    axes[1].axis('off')

    # Motion image with selected keypoints
    if motion_image:
        axes[2].imshow(motion_image)
    else:
        axes[2].imshow(original_image)
    axes[2].set_title("Selected Motion", fontsize=12)
    axes[2].axis('off')

    # Add text summary below
    summary_text = f"""
    Grasp: {context_json.get('grasp_keypoint', 'N/A')} | Function: {context_json.get('function_keypoint', 'N/A')} | Target: {context_json.get('target_keypoint', 'N/A')}
    Tiles: pre={context_json.get('pre_contact_tile', 'N/A')} -> target={context_json.get('target_tile', 'N/A')} -> post={context_json.get('post_contact_tile', 'N/A')}
    Heights: pre={context_json.get('pre_contact_height', 'N/A')} | post={context_json.get('post_contact_height', 'N/A')} | angle={context_json.get('target_angle', 'N/A')}
    """
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_keypoints_json(
    candidate_keypoints: Dict,
    moka_output: MOKAOutput,
    context_json: Dict,
    output_path: Path
):
    """Save all keypoint coordinates to JSON file."""
    data = {
        "candidate_keypoints": {
            "grasped_P": [list(kp) for kp in candidate_keypoints.get('grasped', [])] if candidate_keypoints.get('grasped') is not None else [],
            "unattached_Q": [list(kp) for kp in candidate_keypoints.get('unattached', [])] if candidate_keypoints.get('unattached') is not None else [],
        },
        "selected_keypoints": {
            "grasp_kp_px": list(moka_output.grasp_kp) if moka_output.grasp_kp else None,
            "function_kp_px": list(moka_output.function_kp) if moka_output.function_kp else None,
            "target_kp_px": list(moka_output.target_kp) if moka_output.target_kp else None,
        },
        "vlm_output": context_json,
        "schema_valid": moka_output.is_valid,
        "validation_errors": moka_output.validation_errors,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# MOKA Validation Pipeline
# =============================================================================

class MOKAValidator:
    """Validates MOKA outputs on a set of tasks."""

    def __init__(self, config_path: str = '/workspace/new_experiment/config/moka.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = edict(yaml.load(f, Loader=yaml.SafeLoader))

        # Load prompts
        self.prompts = self._load_prompts()

        # Results storage
        self.results: List[ValidationResult] = []

    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompt_dir = os.path.join(self.config.prompt_root_dir, self.config.prompt_name)
        for filename in os.listdir(prompt_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(prompt_dir, filename), 'r') as f:
                    prompts[filename[:-4]] = f.read()
        return prompts

    def validate_task(self, task_id: int, task_info: Dict, output_dir: Path) -> ValidationResult:
        """Run MOKA on a single task and validate output."""
        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_info['description']}")
        print(f"{'='*60}")

        # Check if image exists
        if not os.path.exists(task_info['image']):
            print(f"WARNING: Image not found: {task_info['image']}")
            return ValidationResult(
                task_id=task_id,
                task_description=task_info['description'],
                success=False,
                moka_output=None,
                failure_mode=FailureMode.MISSED_OBJECT,
                failure_details=f"Image not found: {task_info['image']}",
                detected_objects=[],
                expected_objects=task_info['expected_objects'],
                processing_time_sec=time.time() - start_time,
            )

        # Create task output directory
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Load and resize image
        obs_image = Image.open(task_info['image']).convert('RGB')
        obs_image = obs_image.resize([512, 512], Image.LANCZOS)
        obs_image.save(task_dir / "original.png")

        # Use manually corrected plan (skip VLM planning which has known issues)
        plan = task_info['plan']
        subtask = plan[0]

        # Get objects to detect
        all_object_names = []
        if subtask['object_grasped']:
            all_object_names.append(subtask['object_grasped'])
        if subtask['object_unattached']:
            all_object_names.append(subtask['object_unattached'])

        print(f"Objects to detect: {all_object_names}")

        try:
            # Step 1: Object detection with GroundingDINO
            print("\n--- Step 1: Object Detection (GroundingDINO) ---")
            boxes, logits, phrases = get_scene_object_bboxes(
                obs_image, all_object_names,
                visualize=True,
                logdir=str(task_dir)
            )
            print(f"Detected: {phrases}")
            print(f"Logits: {logits}")

            # Save detection visualization
            if boxes is not None:
                save_detection_visualization(
                    obs_image, boxes.cpu().numpy(), logits.cpu().numpy(), phrases,
                    task_dir / "detection.png"
                )

            # Check if we detected expected objects
            detected_set = set(p.lower() for p in (phrases or []))
            expected_set = set(o.lower() for o in all_object_names)

            if not detected_set.intersection(expected_set):
                return ValidationResult(
                    task_id=task_id,
                    task_description=task_info['description'],
                    success=False,
                    moka_output=None,
                    failure_mode=FailureMode.MISSED_OBJECT,
                    failure_details=f"Expected {expected_set}, detected {detected_set}",
                    detected_objects=list(phrases or []),
                    expected_objects=all_object_names,
                    processing_time_sec=time.time() - start_time,
                )

            # Step 2: Segmentation with SAM
            print("\n--- Step 2: Segmentation (SAM) ---")
            segmasks = get_segmentation_masks(
                obs_image, all_object_names, boxes, logits, phrases,
                visualize=True,
                logdir=str(task_dir)
            )
            print(f"Segmented objects: {list(segmasks.keys())}")

            # Save segmentation visualization
            if segmasks:
                save_segmentation_visualization(obs_image, segmasks, task_dir / "segmentation.png")

            if len(segmasks) == 0:
                return ValidationResult(
                    task_id=task_id,
                    task_description=task_info['description'],
                    success=False,
                    moka_output=None,
                    failure_mode=FailureMode.SEGMENTATION_FAIL,
                    failure_details="SAM produced no segmentation masks",
                    detected_objects=list(phrases or []),
                    expected_objects=all_object_names,
                    processing_time_sec=time.time() - start_time,
                )

            # Step 3: Propose candidate keypoints
            print("\n--- Step 3: Candidate Keypoints ---")
            candidate_keypoints = propose_candidate_keypoints(
                subtask,
                segmasks,
                num_samples=self.config.num_candidate_keypoints
            )

            grasped_kps = candidate_keypoints.get('grasped')
            unattached_kps = candidate_keypoints.get('unattached')
            n_grasped = len(grasped_kps) if grasped_kps is not None else 0
            n_unattached = len(unattached_kps) if unattached_kps is not None else 0
            print(f"Grasped keypoints: {n_grasped}")
            print(f"Unattached keypoints: {n_unattached}")

            # Save candidate keypoints visualization
            save_candidate_keypoints_visualization(
                obs_image, candidate_keypoints, task_dir / "candidates.png"
            )

            # Step 4: Annotate image with visual marks
            print("\n--- Step 4: Visual Annotation ---")
            annotated_image = annotate_visual_prompts(
                obs_image,
                candidate_keypoints,
                waypoint_grid_size=self.config.waypoint_grid_size,
                log_dir=str(task_dir)
            )
            annotated_image.save(task_dir / "annotated.png")

            # Step 5: Motion selection from VLM
            print("\n--- Step 5: Motion Selection (VLM) ---")
            context, context_json, motion_img = request_motion(
                subtask,
                obs_image,
                annotated_image,
                candidate_keypoints,
                waypoint_grid_size=self.config.waypoint_grid_size,
                prompts=self.prompts,
                debug=True,
                log_dir=str(task_dir)
            )

            if motion_img:
                motion_img.save(task_dir / "motion.png")

            if context_json is None:
                return ValidationResult(
                    task_id=task_id,
                    task_description=task_info['description'],
                    success=False,
                    moka_output=None,
                    failure_mode=FailureMode.VLM_PARSE_ERROR,
                    failure_details="Failed to parse VLM motion output",
                    detected_objects=list(phrases or []),
                    expected_objects=all_object_names,
                    processing_time_sec=time.time() - start_time,
                    candidate_keypoints=candidate_keypoints,
                )

            print(f"\n=== MOTION CONTEXT ===")
            print(json.dumps(context_json, indent=2))

            # Save context JSON
            with open(task_dir / "context.json", 'w') as f:
                json.dump(context_json, f, indent=2)

            # Build MOKAOutput from context
            moka_output = MOKAOutput(
                grasp_kp=tuple(context['keypoints_2d']['grasp']) if context['keypoints_2d']['grasp'] is not None else None,
                function_kp=tuple(context['keypoints_2d']['function']) if context['keypoints_2d']['function'] is not None else None,
                target_kp=tuple(context['keypoints_2d']['target']) if context['keypoints_2d']['target'] is not None else None,
                pre_tile=context_json.get('pre_contact_tile'),
                target_tile=context_json.get('target_tile'),
                post_tile=context_json.get('post_contact_tile'),
                pre_height=context_json.get('pre_contact_height'),
                post_height=context_json.get('post_contact_height'),
                target_angle=context_json.get('target_angle'),
                raw_json=context_json,
            )

            # Validate schema
            moka_output.validate(image_size=(512, 512))

            # Save grid visualization
            save_grid_visualization(
                obs_image, context_json, self.config.waypoint_grid_size,
                task_dir / "grid.png"
            )

            # Save comparison visualization
            save_comparison_visualization(
                obs_image, annotated_image, motion_img, moka_output, context_json,
                task_dir / "comparison.png"
            )

            # Save keypoints JSON
            save_keypoints_json(
                candidate_keypoints, moka_output, context_json,
                task_dir / "keypoints.json"
            )

            if not moka_output.is_valid:
                return ValidationResult(
                    task_id=task_id,
                    task_description=task_info['description'],
                    success=False,
                    moka_output=moka_output,
                    failure_mode=FailureMode.SCHEMA_VIOLATION,
                    failure_details="; ".join(moka_output.validation_errors),
                    detected_objects=list(phrases or []),
                    expected_objects=all_object_names,
                    processing_time_sec=time.time() - start_time,
                    candidate_keypoints=candidate_keypoints,
                )

            # Check if we got essential keypoints
            if moka_output.grasp_kp is None:
                return ValidationResult(
                    task_id=task_id,
                    task_description=task_info['description'],
                    success=False,
                    moka_output=moka_output,
                    failure_mode=FailureMode.WRONG_AFFORDANCE,
                    failure_details="No grasp keypoint selected",
                    detected_objects=list(phrases or []),
                    expected_objects=all_object_names,
                    processing_time_sec=time.time() - start_time,
                    candidate_keypoints=candidate_keypoints,
                )

            # Success!
            return ValidationResult(
                task_id=task_id,
                task_description=task_info['description'],
                success=True,
                moka_output=moka_output,
                failure_mode=FailureMode.GOOD,
                failure_details="",
                detected_objects=list(phrases or []),
                expected_objects=all_object_names,
                processing_time_sec=time.time() - start_time,
                candidate_keypoints=candidate_keypoints,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ValidationResult(
                task_id=task_id,
                task_description=task_info['description'],
                success=False,
                moka_output=None,
                failure_mode=FailureMode.SCHEMA_VIOLATION,
                failure_details=str(e),
                detected_objects=[],
                expected_objects=all_object_names,
                processing_time_sec=time.time() - start_time,
            )

    def run_validation(self, tasks: Dict[int, Dict], output_dir: Path) -> Dict:
        """Run validation on all tasks."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if USE_GPT4V:
            print("Using GPT-4V backend (no local model loading needed)")
        else:
            print("Loading Qwen model...")
            load_qwen_model()
            print("Model loaded!")

        self.results = []
        for task_id, task_info in sorted(tasks.items()):
            result = self.validate_task(task_id, task_info, output_dir)
            self.results.append(result)

        return self.generate_report(output_dir)

    def generate_report(self, output_dir: Path) -> Dict:
        """Generate validation report."""
        # Count tasks that actually ran (excluding missing images)
        ran_tasks = [r for r in self.results if "not found" not in r.failure_details.lower()]

        report = {
            "timestamp": datetime.now().isoformat(),
            "vlm_backend": "GPT-4V" if USE_GPT4V else "Qwen2.5-VL-7B",
            "total_tasks": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in ran_tasks if not r.success),
            "skipped": len(self.results) - len(ran_tasks),
            "success_rate": sum(1 for r in ran_tasks if r.success) / max(1, len(ran_tasks)),
            "failure_mode_counts": {},
            "task_results": [],
        }

        # Count failure modes
        for mode in FailureMode:
            count = sum(1 for r in self.results if r.failure_mode == mode)
            if count > 0:
                report["failure_mode_counts"][mode.value] = count

        # Per-task results
        for r in self.results:
            task_result = {
                "task_id": r.task_id,
                "description": r.task_description,
                "success": r.success,
                "failure_mode": r.failure_mode.value,
                "failure_details": r.failure_details,
                "detected_objects": r.detected_objects,
                "expected_objects": r.expected_objects,
                "processing_time_sec": r.processing_time_sec,
            }

            if r.moka_output:
                task_result["moka_output"] = {
                    "grasp_kp": r.moka_output.grasp_kp,
                    "function_kp": r.moka_output.function_kp,
                    "target_kp": r.moka_output.target_kp,
                    "pre_tile": r.moka_output.pre_tile,
                    "target_tile": r.moka_output.target_tile,
                    "post_tile": r.moka_output.post_tile,
                    "pre_height": r.moka_output.pre_height,
                    "post_height": r.moka_output.post_height,
                    "target_angle": r.moka_output.target_angle,
                    "is_valid": r.moka_output.is_valid,
                    "raw_json": r.moka_output.raw_json,
                }

            report["task_results"].append(task_result)

        # Save report
        with open(output_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("MOKA VALIDATION REPORT")
        print("="*60)
        print(f"VLM Backend: {report['vlm_backend']}")
        print(f"Total tasks: {report['total_tasks']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"Skipped (no image): {report['skipped']}")
        print(f"Success rate (excl. skipped): {report['success_rate']*100:.1f}%")

        print("\nFailure modes:")
        for mode, count in report["failure_mode_counts"].items():
            print(f"  {mode}: {count}")

        print("\nPer-task results:")
        for r in self.results:
            status = "OK" if r.success else ("SKIP" if "not found" in r.failure_details.lower() else "FAIL")
            print(f"  Task {r.task_id}: [{status}] {r.failure_mode.value}")
            if r.moka_output and r.moka_output.grasp_kp:
                print(f"    grasp_kp: {r.moka_output.grasp_kp}")
                print(f"    target_kp: {r.moka_output.target_kp}")
            if r.failure_details and status != "OK":
                details = r.failure_details[:80] + "..." if len(r.failure_details) > 80 else r.failure_details
                print(f"    details: {details}")

        # Exit criteria check
        print("\n" + "="*60)
        print("EXIT CRITERIA CHECK")
        print("="*60)
        if len(ran_tasks) > 0:
            if report['success_rate'] >= 0.8:
                print("PASS: Success rate >= 80%")
                print("Ready to proceed to Phase 1: Logging & Replay")
            else:
                print("FAIL: Success rate < 80%")
                print("Need to fix MOKA or adjust tasks before proceeding")
        else:
            print("WARNING: No tasks could be run (missing images)")

        print(f"\nVisual outputs saved to each task folder:")
        print("  - original.png: Original input image")
        print("  - detection.png: GroundingDINO bounding boxes")
        print("  - segmentation.png: SAM segmentation masks")
        print("  - candidates.png: All P0-P5, Q0-Q5 keypoints")
        print("  - annotated.png: Annotated image sent to VLM")
        print("  - grid.png: Tile grid with selected tiles")
        print("  - motion.png: Final motion visualization")
        print("  - comparison.png: Side-by-side comparison")
        print("  - context.json: Raw VLM output")
        print("  - keypoints.json: All keypoint coordinates")

        return report


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path("/workspace/new_experiment/phase0_results")

    validator = MOKAValidator()

    # Filter to only tasks with available images
    available_tasks = {}
    for task_id, task_info in LIBERO_SPATIAL_TASKS.items():
        if os.path.exists(task_info['image']):
            available_tasks[task_id] = task_info
        else:
            print(f"Skipping task {task_id}: image not found")

    if not available_tasks:
        print("ERROR: No task images found!")
        print("Please capture LIBERO images first or check image paths.")
        return None

    print(f"\nRunning validation on {len(available_tasks)} tasks: {list(available_tasks.keys())}")

    report = validator.run_validation(available_tasks, output_dir)

    print(f"\nResults saved to: {output_dir}")
    return report


if __name__ == "__main__":
    main()
