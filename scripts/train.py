#!/usr/bin/env python3
"""
Main training script for Brain-Inspired Robot Control.
"""

import argparse
import yaml
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_robot.training.trainer import BrainRobotTrainer


def main():
    parser = argparse.ArgumentParser(description="Train brain-inspired robot control")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create trainer
    trainer = BrainRobotTrainer(config=config, device=args.device)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
