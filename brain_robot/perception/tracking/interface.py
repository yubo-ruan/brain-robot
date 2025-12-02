"""Tracker interface for object identity persistence.

Design Philosophy:
- Detector outputs: class labels ("bowl", "plate") + 3D positions
- Tracker outputs: instance IDs ("akita_black_bowl_1_main") + persistent tracks
- World model receives: consistent instance IDs across frames

This separation allows:
1. Detector to focus purely on "what is this?" (classification)
2. Tracker to focus on "which one is this?" (association)
3. Clean interface for world model (same PerceptionResult format)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Single detection from object detector.

    This is the OUTPUT of the detector, INPUT to the tracker.
    """
    class_name: str           # e.g., "bowl", "plate"
    position: np.ndarray      # 3D position in world frame [x, y, z]
    confidence: float         # Detection confidence [0, 1]
    bbox: Optional[List[float]] = None  # 2D bbox [x1, y1, x2, y2] if available

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)


@dataclass
class ObjectTrack:
    """Tracked object with persistent identity.

    This is the OUTPUT of the tracker.
    """
    # Core identity (maps to LIBERO instance IDs)
    track_id: int             # Internal tracker ID (integer for efficiency)
    instance_id: str          # LIBERO-style ID, e.g., "akita_black_bowl_1_main"
    class_name: str           # e.g., "bowl"

    # Current state
    position: np.ndarray      # Last known 3D position [x, y, z]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Estimated velocity

    # Tracking quality
    confidence: float = 1.0   # Track confidence (decays when not detected)
    age: int = 0              # Frames since track creation
    hits: int = 0             # Number of successful associations
    misses: int = 0           # Consecutive frames without detection

    # Timestamps
    last_seen: float = 0.0    # Timestamp of last detection
    created_at: float = 0.0   # Timestamp of track creation

    # State flags
    is_active: bool = True    # Track is being actively updated
    is_held: bool = False     # Object is currently held by gripper

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)
        if isinstance(self.velocity, list):
            self.velocity = np.array(self.velocity)

    @property
    def is_reliable(self) -> bool:
        """Track has enough history to be trusted."""
        return self.hits >= 3 and self.misses < 5

    def to_pose(self, orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert to 7D pose for PerceptionResult.

        If orientation unknown, uses identity quaternion.
        """
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        return np.concatenate([self.position, orientation])


@dataclass
class TrackingResult:
    """Result from tracker for a single frame.

    Can be converted to PerceptionResult for world model compatibility.
    """
    tracks: List[ObjectTrack] = field(default_factory=list)
    timestamp: float = 0.0

    # Debug info
    n_detections: int = 0     # Input detections this frame
    n_matched: int = 0        # Detections matched to existing tracks
    n_new_tracks: int = 0     # New tracks created
    n_lost_tracks: int = 0    # Tracks marked as lost

    def get_track_by_id(self, instance_id: str) -> Optional[ObjectTrack]:
        """Find track by LIBERO instance ID."""
        for track in self.tracks:
            if track.instance_id == instance_id:
                return track
        return None

    def get_tracks_by_class(self, class_name: str) -> List[ObjectTrack]:
        """Get all tracks of a given class."""
        return [t for t in self.tracks if t.class_name == class_name]

    def get_active_tracks(self) -> List[ObjectTrack]:
        """Get only active (not lost) tracks."""
        return [t for t in self.tracks if t.is_active]

    def to_objects_dict(self) -> Dict[str, np.ndarray]:
        """Convert to PerceptionResult.objects format.

        Returns: {instance_id: 7D pose}
        """
        return {
            track.instance_id: track.to_pose()
            for track in self.tracks
            if track.is_active
        }


class TrackerInterface(ABC):
    """Abstract base class for object trackers.

    Implementations must handle:
    1. Associating detections to existing tracks (data association)
    2. Creating new tracks for unmatched detections
    3. Managing track lifecycle (age, confidence decay, deletion)
    4. Maintaining instance IDs consistent with environment

    Design for LIBERO:
    - Initialize with known object instance IDs from environment
    - First few frames: associate detections to known IDs
    - Runtime: maintain associations, handle occlusions
    """

    @abstractmethod
    def initialize(self, known_objects: Dict[str, np.ndarray]):
        """Initialize tracker with known objects from environment.

        Called at episode start with oracle knowledge of objects.
        This is the "bootstrap" phase where we establish instance IDs.

        Args:
            known_objects: {instance_id: initial_position} from env
        """
        pass

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        timestamp: float = 0.0,
    ) -> TrackingResult:
        """Process new detections and update tracks.

        Args:
            detections: List of new detections from object detector
            timestamp: Current time (for velocity estimation)

        Returns:
            TrackingResult with updated tracks
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset tracker state for new episode."""
        pass

    def get_instance_id_for_detection(
        self,
        detection: Detection,
        candidates: List[str],
    ) -> Optional[str]:
        """Match detection to one of the candidate instance IDs.

        Override for custom matching logic.
        Default: returns None (let update() handle it).
        """
        return None

    def mark_object_held(self, instance_id: str, is_held: bool):
        """Mark an object as held/released by gripper.

        Held objects follow gripper motion, not independent tracking.
        """
        pass


# Type aliases for clarity
Detections = List[Detection]
Tracks = List[ObjectTrack]
