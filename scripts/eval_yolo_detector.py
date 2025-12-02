#!/usr/bin/env python3
"""Evaluate trained YOLO detector on LIBERO perception data.

Computes detection metrics and visualizes results.

Usage:
    python scripts/eval_yolo_detector.py --model models/yolo_libero.pt --data data/yolo_libero/data.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def evaluate_detections(
    predictions: list,
    ground_truth: list,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate detections against ground truth.

    Args:
        predictions: List of predicted detections
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with TP, FP, FN, precision, recall, F1
    """
    # Match predictions to GT by class
    matched_gt = set()
    tp, fp = 0, 0

    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)

    for pred in predictions:
        pred_class = pred["class_name"]
        pred_box = pred["bbox"]

        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            if gt["class_name"] != pred_class:
                continue

            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx is not None:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def visualize_detection(
    image_path: Path,
    predictions: list,
    ground_truth: list,
    output_path: Path,
):
    """Visualize predictions vs ground truth on an image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    # Draw ground truth in green
    for gt in ground_truth:
        x1, y1, x2, y2 = gt["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        label = f"GT:{gt['class_name']}"
        draw.text((x1, y1 - 12), label, fill=(0, 255, 0), font=font)

    # Draw predictions in red/blue
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        conf = pred.get("confidence", 0)

        # Color based on confidence
        if conf > 0.7:
            color = (0, 0, 255)  # Blue for high confidence
        else:
            color = (255, 0, 0)  # Red for low confidence

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{pred['class_name']}:{conf:.2f}"
        draw.text((x1, y2 + 2), label, fill=color, font=font)

    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Detector")
    parser.add_argument("--model", type=str, default="models/yolo_libero.pt",
                        help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/yolo_libero/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to evaluate (train/val)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for matching")
    parser.add_argument("--visualize", type=int, default=10,
                        help="Number of images to visualize (0 to disable)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations")
    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train the model first with train_yolo_detector.py")
        return 1

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return 1

    print("=" * 70)
    print("YOLO DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("=" * 70)

    model = YOLO(str(model_path))

    # Load data config
    import yaml
    data_path = Path(args.data)
    with open(data_path) as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", {})
    data_root = Path(data_config["path"])
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    # Get all image files
    image_files = sorted(images_dir.glob("*.png"))
    print(f"\nFound {len(image_files)} images in {args.split} set")

    # Evaluate each image
    all_metrics = {
        "tp": 0, "fp": 0, "fn": 0,
        "per_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }

    vis_count = 0
    vis_dir = None
    if args.visualize > 0:
        vis_dir = Path(args.output_dir) if args.output_dir else data_root / "eval_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        # Load ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        # Convert YOLO to xyxy format
                        img_w, img_h = 256, 256  # Assumed
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h

                        gt_boxes.append({
                            "class_name": class_names.get(class_id, f"class_{class_id}"),
                            "bbox": [x1, y1, x2, y2],
                        })

        # Run detection
        results = model(str(img_path), conf=args.conf, verbose=False)
        pred_boxes = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf.item())

                pred_boxes.append({
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "bbox": bbox,
                    "confidence": confidence,
                })

        # Evaluate
        metrics = evaluate_detections(pred_boxes, gt_boxes, args.iou)

        all_metrics["tp"] += metrics["tp"]
        all_metrics["fp"] += metrics["fp"]
        all_metrics["fn"] += metrics["fn"]

        # Per-class metrics
        for gt in gt_boxes:
            cls = gt["class_name"]
            all_metrics["per_class"][cls]["fn"] += 1  # Count GT

        for pred in pred_boxes:
            cls = pred["class_name"]
            # Check if matched
            matched = False
            for gt in gt_boxes:
                if gt["class_name"] == cls:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou >= args.iou:
                        matched = True
                        break

            if matched:
                all_metrics["per_class"][cls]["tp"] += 1
                all_metrics["per_class"][cls]["fn"] -= 1  # Remove from FN
            else:
                all_metrics["per_class"][cls]["fp"] += 1

        # Visualize
        if vis_dir and vis_count < args.visualize:
            vis_path = vis_dir / f"eval_{img_path.stem}.png"
            visualize_detection(img_path, pred_boxes, gt_boxes, vis_path)
            vis_count += 1

    # Compute final metrics
    tp, fp, fn = all_metrics["tp"], all_metrics["fp"], all_metrics["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    for cls in sorted(all_metrics["per_class"].keys()):
        cls_metrics = all_metrics["per_class"][cls]
        cls_tp = cls_metrics["tp"]
        cls_fp = cls_metrics["fp"]
        cls_fn = cls_metrics["fn"]

        cls_prec = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
        cls_rec = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0

        print(f"  {cls}: P={cls_prec:.3f}, R={cls_rec:.3f} (TP={cls_tp}, FP={cls_fp}, FN={cls_fn})")

    if vis_dir:
        print(f"\nVisualizations saved to: {vis_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
