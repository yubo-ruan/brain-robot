"""Logging module for brain_robot.

Provides episode logging and run tracking.
"""

from .episode_logger import EpisodeLog, EpisodeLogger

# Phase 1: HDF5 logging and replay
from .hdf5_episode_logger import (
    MOKAOutput,
    Timestep,
    HDF5EpisodeLogger,
    HDF5Episode,
    load_hdf5_episode,
    get_moka_outputs,
)
from .replay_viewer import ReplayViewer, quick_view

__all__ = [
    # Original JSON-based logger
    "EpisodeLog",
    "EpisodeLogger",
    # HDF5-based logger for training
    "MOKAOutput",
    "Timestep",
    "HDF5EpisodeLogger",
    "HDF5Episode",
    "load_hdf5_episode",
    "get_moka_outputs",
    # Replay viewer
    "ReplayViewer",
    "quick_view",
]
