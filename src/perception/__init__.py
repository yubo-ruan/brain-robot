"""Perception module for src.

Provides perception interfaces and implementations for extracting
object and robot state from the environment.
"""

from .interface import PerceptionInterface, PerceptionResult
from .oracle import OraclePerception
from .noisy_oracle import NoisyOraclePerception

__all__ = [
    "PerceptionInterface",
    "PerceptionResult",
    "OraclePerception",
    "NoisyOraclePerception",
]
