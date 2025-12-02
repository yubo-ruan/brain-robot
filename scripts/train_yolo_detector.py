#!/usr/bin/env python3
"""Train YOLO detector on LIBERO perception data.

Uses ultralytics YOLO for training object detection model.

Usage:
    python scripts/train_yolo_detector.py --data data/yolo_libero/data.yaml --epochs 100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Detector")
    parser.add_argument("--data", type=str, default="data/yolo_libero/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (yolov8n, yolov8s, yolov8m)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="Image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project directory for outputs")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: timestamp)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Check ultralytics installation
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        return 1

    # Verify data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data.yaml not found at {data_path}")
        print("Run convert_to_yolo_format.py first to create YOLO format data.")
        return 1

    # Create experiment name
    if args.name is None:
        args.name = f"libero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print("YOLO DETECTOR TRAINING")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("=" * 70)

    # Load base model
    print(f"\nLoading base model {args.model}...")
    model = YOLO(args.model)

    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(data_path.absolute()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        # Augmentation settings for tabletop robotics
        hsv_h=0.015,      # Slight hue variation
        hsv_s=0.4,        # Saturation variation
        hsv_v=0.3,        # Value variation
        degrees=10.0,     # Small rotation (objects don't rotate much)
        translate=0.1,    # Small translation
        scale=0.2,        # Some scale variation
        fliplr=0.0,       # No horizontal flip (breaks spatial semantics)
        flipud=0.0,       # No vertical flip
        mosaic=0.5,       # Mosaic augmentation
        mixup=0.1,        # Mixup augmentation
        # Optimization
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        # Saving
        save=True,
        save_period=10,
        plots=True,
    )

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Get best model path
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_model_path = Path(args.project) / args.name / "weights" / "last.pt"

    print(f"\nBest model: {best_model_path}")
    print(f"Last model: {last_model_path}")

    # Run validation
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # Load best model and validate
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(data=str(data_path.absolute()))

    # Print metrics
    if hasattr(val_results, 'box'):
        box_metrics = val_results.box
        print(f"\nmAP50: {box_metrics.map50:.4f}")
        print(f"mAP50-95: {box_metrics.map:.4f}")

        if hasattr(box_metrics, 'maps'):
            print(f"\nPer-class mAP50:")
            # Get class names from data.yaml
            import yaml
            with open(data_path) as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', {})

            for i, map_val in enumerate(box_metrics.maps):
                class_name = class_names.get(i, f"class_{i}")
                print(f"  {class_name}: {map_val:.4f}")

    # Copy best model to models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    final_model_path = models_dir / "yolo_libero.pt"

    import shutil
    shutil.copy2(best_model_path, final_model_path)
    print(f"\nBest model copied to: {final_model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
