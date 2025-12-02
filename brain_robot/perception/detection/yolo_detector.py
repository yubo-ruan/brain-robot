"""YOLO-based object detector for LIBERO objects.

Fine-tuned on collected perception data for LIBERO manipulation tasks.
Outputs 2D bounding boxes + class labels that flow into tracker.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from .interface import ObjectDetector, Detection


# LIBERO object classes (must match training data)
LIBERO_CLASSES = [
    "bowl",
    "plate",
    "mug",
    "ramekin",
    "cabinet",
    "drawer",
    "cookie_box",
    "can",
    "bottle",
    "stove",
]


@dataclass
class YOLOObjectDetector(ObjectDetector):
    """YOLO-based object detector for LIBERO objects.

    Usage:
        detector = YOLOObjectDetector(model_path="models/yolo_libero.pt")
        detections = detector.detect(rgb_image)

        # Each detection has class_name, bbox, confidence
        for det in detections:
            print(f"{det.class_name}: {det.confidence:.2f}")
    """

    # Model configuration
    model_path: str = "models/yolo_libero.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45  # For NMS
    device: str = "cuda"

    # Class-specific confidence thresholds (override default for specific classes)
    # Bowl detections have lower confidence due to dark color and training data
    class_confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "bowl": 0.15,  # Lower threshold for bowls (they typically have 0.2-0.3 confidence)
        "ramekin": 0.25,  # Similar issue with small objects
    })

    # Classes this detector recognizes
    CLASSES: List[str] = field(default_factory=lambda: LIBERO_CLASSES.copy())

    # Internal state
    _model: Optional[object] = field(default=None, repr=False)
    _class_map: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        """Load YOLO model if path exists."""
        if Path(self.model_path).exists():
            self._load_model()
        else:
            print(f"Warning: Model not found at {self.model_path}")
            print("Call load_model() after training, or use default YOLO.")

    def _load_model(self):
        """Load YOLO model from checkpoint."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model.to(self.device)

            # Build class index -> name mapping
            # YOLO models have a names dict
            if hasattr(self._model, 'names'):
                self._class_map = dict(self._model.names)
            else:
                # Fallback to positional mapping
                self._class_map = {i: name for i, name in enumerate(self.CLASSES)}

            print(f"Loaded YOLO model from {self.model_path}")
            print(f"Classes: {list(self._class_map.values())}")

        except ImportError:
            print("Warning: ultralytics not installed. Run: pip install ultralytics")
            self._model = None

    def load_model(self, model_path: Optional[str] = None):
        """Explicitly load model (for use after training)."""
        if model_path:
            self.model_path = model_path
        self._load_model()

    def detect(self, rgb_image: np.ndarray) -> List[Detection]:
        """Detect objects in image.

        Args:
            rgb_image: RGB image (H, W, 3), uint8

        Returns:
            List of Detection objects
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use minimum of all thresholds to get all potential detections
        # Then filter by class-specific thresholds
        min_threshold = min(
            self.confidence_threshold,
            *self.class_confidence_thresholds.values()
        )

        # Run inference with low threshold
        results = self._model(
            rgb_image,
            conf=min_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    # Get class ID and name
                    class_id = int(box.cls.item())
                    class_name = self._class_map.get(class_id, f"class_{class_id}")

                    # Get confidence
                    confidence = float(box.conf.item())

                    # Apply class-specific threshold
                    threshold = self.class_confidence_thresholds.get(
                        class_name, self.confidence_threshold
                    )
                    if confidence < threshold:
                        continue

                    # Get bbox in xyxy format
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    # Create detection (position will be estimated by tracker/pose)
                    det = Detection(
                        class_name=class_name,
                        position=np.zeros(3),  # Placeholder, will be set by depth/pose
                        confidence=confidence,
                        bbox=bbox,
                    )
                    detections.append(det)

        return detections

    def warmup(self, image_size: tuple = (256, 256, 3)):
        """Run warmup inference for consistent latency."""
        if self._model is None:
            return

        dummy_image = np.zeros(image_size, dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy_image)

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        self.confidence_threshold = threshold

    @classmethod
    def from_pretrained(cls, model_name: str = "yolov8n") -> "YOLOObjectDetector":
        """Load a pretrained YOLO model (not fine-tuned).

        Useful for initial testing before training on LIBERO data.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, etc.)

        Returns:
            YOLOObjectDetector with pretrained model
        """
        detector = cls(model_path=f"{model_name}.pt")

        try:
            from ultralytics import YOLO
            detector._model = YOLO(model_name)
            detector._model.to(detector.device)

            # Use COCO classes for pretrained model
            if hasattr(detector._model, 'names'):
                detector._class_map = dict(detector._model.names)
                detector.CLASSES = list(detector._class_map.values())

            print(f"Loaded pretrained {model_name}")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")

        return detector
