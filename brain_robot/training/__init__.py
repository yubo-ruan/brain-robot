"""
Training module for RL training with VLM guidance.
"""

from .trainer import BrainRobotTrainer
from .rewards import RewardShaper

__all__ = [
    "BrainRobotTrainer",
    "RewardShaper",
]
