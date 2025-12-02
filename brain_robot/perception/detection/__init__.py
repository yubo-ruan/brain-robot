"""Object detection for learned perception.

Provides YOLO-based object detection for LIBERO manipulation tasks.
Outputs class labels and 2D bounding boxes - tracking converts these
to persistent instance IDs.
"""

from .interface import Detection, ObjectDetector
from .yolo_detector import YOLOObjectDetector

__all__ = [
    "Detection",
    "ObjectDetector",
    "YOLOObjectDetector",
]
