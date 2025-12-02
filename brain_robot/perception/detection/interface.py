"""Detection interface for object detection.

Note: Detection dataclass is also defined in tracking/interface.py.
This module re-exports it for convenience, but the canonical definition
is in tracking since detections flow into the tracker.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Re-export Detection from tracking (single source of truth)
from ..tracking.interface import Detection


class ObjectDetector(ABC):
    """Abstract base class for object detectors.

    Implementations:
    - YOLOObjectDetector: YOLO-based detection (trained on LIBERO)
    - SAMCLIPDetector: Zero-shot detection with SAM + CLIP

    Output flows to tracker:
        detector.detect(image) -> List[Detection] -> tracker.update(detections)
    """

    # Classes the detector can recognize
    CLASSES: List[str] = []

    @abstractmethod
    def detect(self, rgb_image: np.ndarray) -> List[Detection]:
        """Detect objects in image.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            List of Detection objects with class_name, position (if depth),
            confidence, and bbox.
        """
        pass

    @abstractmethod
    def warmup(self):
        """Run warmup inference for consistent latency."""
        pass

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        pass
