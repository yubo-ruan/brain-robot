#!/usr/bin/env python3
"""
Phase 0: RoboPoint Validation on LIBERO Tasks

Tests RoboPoint's spatial grounding capabilities on LIBERO spatial tasks.
Compares against MOKA's ~25% accuracy baseline.
"""

import os
import sys
import json
import torch
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add RoboPoint to path
sys.path.insert(0, '/workspace/new_experiment/RoboPoint')

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from robopoint.conversation import conv_templates
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


# =============================================================================
# LIBERO Task Definitions with Spatial Context
# =============================================================================

LIBERO_SPATIAL_TASKS = {
    4: {
        "description": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "image": "/workspace/new_experiment/libero_512/task_4_initial.png",
        "grasp_prompt": "Locate the center point of the black bowl that is inside the top drawer of the wooden cabinet.",
        "place_prompt": "Locate the center point of the white plate with the red rim on the table.",
        "expected_grasp_region": "top-left (drawer area)",
        "expected_place_region": "center-right (plate on table)",
    },
    5: {
        "description": "pick up the black bowl on the ramekin and place it on the plate",
        "image": "/workspace/new_experiment/libero_512/task_5_initial.png",
        "grasp_prompt": "Locate the center point of the black bowl that is sitting on top of the white ramekin.",
        "place_prompt": "Locate the center point of the white plate with the red rim.",
        "expected_grasp_region": "center (bowl on ramekin)",
        "expected_place_region": "right (plate)",
    },
    8: {
        "description": "pick up the black bowl next to the plate and place it on the plate",
        "image": "/workspace/new_experiment/libero_512/task_8_initial.png",
        "grasp_prompt": "Locate the center point of the black bowl that is positioned next to the plate, not the other bowls in the scene.",
        "place_prompt": "Locate the center point of the white plate with the red rim.",
        "expected_grasp_region": "near the plate",
        "expected_place_region": "center (plate)",
    },
    9: {
        "description": "pick up the black bowl on the wooden cabinet and place it on the plate",
        "image": "/workspace/new_experiment/libero_512/task_9_initial.png",
        "grasp_prompt": "Locate the center point of the black bowl that is on top of the wooden cabinet.",
        "place_prompt": "Locate the center point of the white plate with the red rim on the table.",
        "expected_grasp_region": "top-left (on cabinet)",
        "expected_place_region": "center-right (plate on table)",
    },
}


class RoboPointValidator:
    """Validates RoboPoint on LIBERO tasks."""

    def __init__(self, model_path: str = "wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora"):
        """Initialize RoboPoint model."""
        print(f"Loading RoboPoint model: {model_path}")
        disable_torch_init()

        # For LoRA models, need to specify base model
        if "lora" in model_path.lower():
            if "vicuna" in model_path.lower():
                if "7b" in model_path:
                    model_base = "lmsys/vicuna-7b-v1.5"
                else:
                    model_base = "lmsys/vicuna-13b-v1.5"
            else:
                if "7b" in model_path:
                    model_base = "meta-llama/Llama-2-7b-chat-hf"
                else:
                    model_base = "meta-llama/Llama-2-13b-chat-hf"
        else:
            model_base = None

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
        self.conv_mode = "llava_v1"
        print("RoboPoint model loaded!")

    def query_keypoint(self, image: Image.Image, prompt: str) -> list:
        """Query RoboPoint for keypoint coordinates.

        Args:
            image: PIL Image
            prompt: Natural language query for keypoint location

        Returns:
            List of (x, y) tuples in normalized coordinates [0, 1]
        """
        # Build prompt with image token
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        # Add instruction for output format
        qs += "\nYour answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
                use_cache=True
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"RoboPoint output: {output}")

        # Parse output - expect format like [(0.5, 0.3), (0.6, 0.4)]
        keypoints = self._parse_keypoints(output)
        return keypoints, output

    def _parse_keypoints(self, output: str) -> list:
        """Parse keypoint coordinates from model output."""
        # Try to find tuples in the output
        pattern = r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)'
        matches = re.findall(pattern, output)

        keypoints = []
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                # Ensure in valid range
                if 0 <= x <= 1 and 0 <= y <= 1:
                    keypoints.append((x, y))
            except ValueError:
                continue

        return keypoints

    def validate_task(self, task_id: int, task_info: dict, output_dir: Path) -> dict:
        """Validate RoboPoint on a single task."""
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_info['description']}")
        print(f"{'='*60}")

        # Check if image exists
        if not os.path.exists(task_info['image']):
            print(f"WARNING: Image not found: {task_info['image']}")
            return {
                "task_id": task_id,
                "success": False,
                "error": "Image not found"
            }

        # Create output directory
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Load image (already 512x512)
        image = Image.open(task_info['image']).convert('RGB')
        print(f"Image size: {image.size}")
        image.save(task_dir / "original.png")

        # Query grasp point
        print(f"\n--- Grasp Query ---")
        print(f"Prompt: {task_info['grasp_prompt']}")
        grasp_kps, grasp_raw = self.query_keypoint(image, task_info['grasp_prompt'])

        # Query place point
        print(f"\n--- Place Query ---")
        print(f"Prompt: {task_info['place_prompt']}")
        place_kps, place_raw = self.query_keypoint(image, task_info['place_prompt'])

        # Visualize results
        self._visualize_results(
            image, grasp_kps, place_kps,
            task_info, task_dir / "robopoint_output.png"
        )

        # Save results
        result = {
            "task_id": task_id,
            "description": task_info['description'],
            "grasp_prompt": task_info['grasp_prompt'],
            "place_prompt": task_info['place_prompt'],
            "grasp_keypoints": grasp_kps,
            "place_keypoints": place_kps,
            "grasp_raw_output": grasp_raw,
            "place_raw_output": place_raw,
            "expected_grasp_region": task_info['expected_grasp_region'],
            "expected_place_region": task_info['expected_place_region'],
            "success": len(grasp_kps) > 0 and len(place_kps) > 0,
        }

        with open(task_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _visualize_results(self, image: Image.Image, grasp_kps: list, place_kps: list,
                          task_info: dict, output_path: Path):
        """Visualize keypoint predictions."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)

        w, h = image.size

        # Plot grasp keypoints in red
        for i, (x, y) in enumerate(grasp_kps):
            px, py = x * w, y * h
            ax.scatter(px, py, c='red', s=200, marker='*', edgecolors='white', linewidths=2, zorder=5)
            ax.text(px + 10, py, f'G{i}', fontsize=12, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot place keypoints in blue
        for i, (x, y) in enumerate(place_kps):
            px, py = x * w, y * h
            ax.scatter(px, py, c='blue', s=200, marker='*', edgecolors='white', linewidths=2, zorder=5)
            ax.text(px + 10, py, f'P{i}', fontsize=12, color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                      markersize=15, label=f'Grasp: {task_info["expected_grasp_region"]}'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                      markersize=15, label=f'Place: {task_info["expected_place_region"]}'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.set_title(f"RoboPoint Output\n{task_info['description'][:60]}...", fontsize=12)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def run_validation(self, output_dir: Path) -> dict:
        """Run validation on all tasks."""
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for task_id, task_info in sorted(LIBERO_SPATIAL_TASKS.items()):
            if os.path.exists(task_info['image']):
                result = self.validate_task(task_id, task_info, output_dir)
                results.append(result)
            else:
                print(f"Skipping task {task_id}: image not found")

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": "RoboPoint (7B LoRA)",
            "total_tasks": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results,
        }

        report["success_rate"] = report["successful"] / max(1, report["total_tasks"])

        with open(output_dir / "validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("ROBOPOINT VALIDATION REPORT")
        print("="*60)
        print(f"Total tasks: {report['total_tasks']}")
        print(f"Successful: {report['successful']}")
        print(f"Success rate: {report['success_rate']*100:.1f}%")

        print("\nPer-task results:")
        for r in results:
            status = "OK" if r.get("success") else "FAIL"
            n_grasp = len(r.get("grasp_keypoints", []))
            n_place = len(r.get("place_keypoints", []))
            print(f"  Task {r['task_id']}: [{status}] grasp={n_grasp} pts, place={n_place} pts")

        return report


def main():
    output_dir = Path("/workspace/new_experiment/phase0_robopoint_results")

    # Use 7B LoRA model (smaller, fits in memory)
    validator = RoboPointValidator(
        model_path="wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora"
    )

    report = validator.run_validation(output_dir)

    print(f"\nResults saved to: {output_dir}")
    return report


if __name__ == "__main__":
    main()
