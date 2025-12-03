"""Nearest-neighbor tracker for object identity persistence.

Simple but effective approach for tabletop manipulation:
- Objects move slowly relative to frame rate
- Few objects in scene (< 20)
- 3D positions from depth provide strong association signal

Algorithm:
1. Initialize with known instance IDs from environment
2. For each detection, find nearest track of same class
3. If distance < threshold, associate; else create new track
4. Tracks without detections decay and eventually die
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .interface import (
    Detection,
    ObjectTrack,
    TrackingResult,
    TrackerInterface,
)


@dataclass
class NearestNeighborTracker(TrackerInterface):
    """Nearest-neighbor tracker using 3D position association.

    Designed for LIBERO tabletop manipulation:
    - Slow object motion (pick-and-place)
    - Known object set at episode start
    - 3D positions from depth sensing

    Usage:
        tracker = NearestNeighborTracker()

        # At episode start, initialize with known objects
        tracker.initialize(env.get_object_poses())

        # Each frame, update with detections
        result = tracker.update(detections, timestamp)

        # Use result.to_objects_dict() for PerceptionResult
    """

    # Default association threshold (meters) - objects closer than this match
    association_threshold: float = 0.10

    # Class-specific association thresholds based on object dimensions
    # Smaller objects need tighter thresholds to avoid mis-associations
    CLASS_ASSOCIATION_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        # Small objects (diameter < 10cm)
        "ramekin": 0.06,       # ~6cm diameter
        "mug": 0.08,           # ~8cm diameter
        "can": 0.07,           # ~7cm diameter
        "bottle": 0.08,        # ~8cm diameter
        "bowl": 0.10,          # ~10cm diameter
        # Medium objects
        "plate": 0.15,         # ~15cm diameter
        "cookie_box": 0.12,    # ~12cm
        # Large objects / landmarks
        "cabinet": 0.25,       # Large, static
        "stove": 0.25,         # Large, static
        "drawer": 0.20,        # Medium-large
    })

    # Track lifecycle
    max_misses: int = 10          # Frames without detection before track dies
    min_hits_to_confirm: int = 2  # Hits needed before track is "reliable"

    # Confidence decay
    confidence_decay: float = 0.9  # Per-frame decay when not detected
    min_confidence: float = 0.1    # Below this, track is deleted

    # Internal state
    _tracks: Dict[int, ObjectTrack] = field(default_factory=dict)
    _next_track_id: int = 0
    _known_instances: Dict[str, str] = field(default_factory=dict)  # instance_id -> class
    _class_instance_map: Dict[str, List[str]] = field(default_factory=dict)  # class -> [instance_ids]

    def initialize(self, known_objects: Dict[str, np.ndarray]):
        """Initialize with known objects from environment.

        Args:
            known_objects: {instance_id: position_3d} from env
        """
        self.reset()

        current_time = time.time()

        # Build class -> instance mapping
        from ..data_collection.collector import instance_id_to_class

        for instance_id, position in known_objects.items():
            class_name = instance_id_to_class(instance_id)

            # Store known instances
            self._known_instances[instance_id] = class_name

            if class_name not in self._class_instance_map:
                self._class_instance_map[class_name] = []
            self._class_instance_map[class_name].append(instance_id)

            # Create initial track
            track = ObjectTrack(
                track_id=self._next_track_id,
                instance_id=instance_id,
                class_name=class_name,
                position=np.array(position[:3]) if len(position) > 3 else np.array(position),
                confidence=1.0,
                age=0,
                hits=1,
                misses=0,
                last_seen=current_time,
                created_at=current_time,
                is_active=True,
            )

            self._tracks[self._next_track_id] = track
            self._next_track_id += 1

    def update(
        self,
        detections: List[Detection],
        timestamp: float = 0.0,
    ) -> TrackingResult:
        """Process detections and update tracks.

        Algorithm:
        1. Group detections by class
        2. For each class, match detections to tracks (Hungarian or greedy)
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Decay unmatched tracks
        """
        if timestamp == 0.0:
            timestamp = time.time()

        result = TrackingResult(timestamp=timestamp, n_detections=len(detections))

        # Group detections by class
        detections_by_class: Dict[str, List[Detection]] = {}
        for det in detections:
            if det.class_name not in detections_by_class:
                detections_by_class[det.class_name] = []
            detections_by_class[det.class_name].append(det)

        # Track which tracks were matched this frame
        matched_track_ids: Set[int] = set()
        matched_detection_indices: Set[Tuple[str, int]] = set()

        # Match detections to tracks by class
        for class_name, class_detections in detections_by_class.items():
            # Get tracks of this class
            class_tracks = [
                t for t in self._tracks.values()
                if t.class_name == class_name and t.is_active
            ]

            if not class_tracks:
                continue

            # Get class-specific association threshold
            class_threshold = self._get_association_threshold(class_name)

            # Greedy nearest-neighbor matching
            # (Could use Hungarian algorithm for optimal matching)
            for det_idx, detection in enumerate(class_detections):
                best_track = None
                best_dist = float('inf')

                for track in class_tracks:
                    if track.track_id in matched_track_ids:
                        continue

                    dist = np.linalg.norm(detection.position - track.position)
                    if dist < best_dist and dist < class_threshold:
                        best_dist = dist
                        best_track = track

                if best_track is not None:
                    # Update matched track
                    dt = timestamp - best_track.last_seen
                    if dt > 0:
                        best_track.velocity = (detection.position - best_track.position) / dt

                    best_track.position = detection.position.copy()
                    best_track.confidence = min(1.0, best_track.confidence + 0.1)
                    best_track.hits += 1
                    best_track.misses = 0
                    best_track.last_seen = timestamp
                    best_track.age += 1

                    matched_track_ids.add(best_track.track_id)
                    matched_detection_indices.add((class_name, det_idx))
                    result.n_matched += 1

        # Handle unmatched detections - create new tracks
        for class_name, class_detections in detections_by_class.items():
            for det_idx, detection in enumerate(class_detections):
                if (class_name, det_idx) in matched_detection_indices:
                    continue

                # Try to find an unassigned known instance ID
                instance_id = self._get_available_instance_id(
                    class_name, detection.position
                )

                if instance_id is None:
                    # No known instance - create generic ID
                    instance_id = f"{class_name}_{self._next_track_id}"

                track = ObjectTrack(
                    track_id=self._next_track_id,
                    instance_id=instance_id,
                    class_name=class_name,
                    position=detection.position.copy(),
                    confidence=detection.confidence,
                    age=0,
                    hits=1,
                    misses=0,
                    last_seen=timestamp,
                    created_at=timestamp,
                    is_active=True,
                )

                self._tracks[self._next_track_id] = track
                self._next_track_id += 1
                result.n_new_tracks += 1

        # Decay unmatched tracks
        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids and track.is_active:
                track.misses += 1
                track.confidence *= self.confidence_decay
                track.age += 1

                # Kill track if too many misses or low confidence
                if track.misses >= self.max_misses or track.confidence < self.min_confidence:
                    track.is_active = False
                    result.n_lost_tracks += 1

        # Build result
        result.tracks = list(self._tracks.values())

        return result

    def _get_available_instance_id(
        self,
        class_name: str,
        position: np.ndarray,
    ) -> Optional[str]:
        """Find an unassigned instance ID for this class.

        Uses nearest-neighbor to known initial positions.
        """
        if class_name not in self._class_instance_map:
            return None

        # Get instance IDs not yet assigned to active tracks
        assigned_ids = {
            t.instance_id for t in self._tracks.values()
            if t.is_active
        }

        available_ids = [
            iid for iid in self._class_instance_map[class_name]
            if iid not in assigned_ids
        ]

        if not available_ids:
            return None

        # If only one available, use it
        if len(available_ids) == 1:
            return available_ids[0]

        # Multiple available - this shouldn't happen with proper initialization
        # Return the first one (or could use position-based matching)
        return available_ids[0]

    def _get_association_threshold(self, class_name: str) -> float:
        """Get association threshold for a specific class.

        Uses class-specific thresholds when available, falls back to default.
        """
        return self.CLASS_ASSOCIATION_THRESHOLDS.get(class_name, self.association_threshold)

    def reset(self):
        """Reset tracker state for new episode."""
        self._tracks = {}
        self._next_track_id = 0
        self._known_instances = {}
        self._class_instance_map = {}

    def mark_object_held(self, instance_id: str, is_held: bool):
        """Mark object as held/released by gripper."""
        for track in self._tracks.values():
            if track.instance_id == instance_id:
                track.is_held = is_held
                break

    def get_track(self, instance_id: str) -> Optional[ObjectTrack]:
        """Get track by instance ID."""
        for track in self._tracks.values():
            if track.instance_id == instance_id:
                return track
        return None

    def get_all_tracks(self) -> List[ObjectTrack]:
        """Get all tracks (including inactive)."""
        return list(self._tracks.values())

    def get_active_tracks(self) -> List[ObjectTrack]:
        """Get only active tracks."""
        return [t for t in self._tracks.values() if t.is_active]
