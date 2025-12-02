"""Perception module for brain_robot.

Provides perception interfaces and implementations for extracting
object and robot state from the environment.

Components:
- interface: Base classes (PerceptionInterface, PerceptionResult)
- oracle: Ground-truth perception from simulator
- noisy_oracle: Oracle with added noise for robustness testing
- learned: YOLO + depth-based learned perception
- data_collection: Tools for collecting perception training data
- tracking: Object tracking for instance ID persistence
- detection: Object detection models (YOLO-based)
"""

from .interface import PerceptionInterface, PerceptionResult
from .oracle import OraclePerception
from .noisy_oracle import NoisyOraclePerception
from .learned import LearnedPerception

__all__ = [
    "PerceptionInterface",
    "PerceptionResult",
    "OraclePerception",
    "NoisyOraclePerception",
    "LearnedPerception",
]
