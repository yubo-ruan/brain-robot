"""Object tracking for learned perception.

Bridges the gap between per-frame detections (class labels) and
persistent instance identities needed for manipulation.

Key insight: Detector outputs "bowl" but world model needs "akita_black_bowl_1_main".
Tracker maintains this mapping across frames using 3D position association.
"""

from .interface import ObjectTrack, TrackerInterface
from .nearest_neighbor import NearestNeighborTracker

__all__ = [
    "ObjectTrack",
    "TrackerInterface",
    "NearestNeighborTracker",
]
