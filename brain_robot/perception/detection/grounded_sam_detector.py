"""Grounded-SAM: Grounding-DINO + SAM2 for open-vocabulary instance segmentation.

Pipeline:
    Image + Text → Grounding-DINO → Bboxes → SAM2 → Masks + 3D positions

This combines:
- Grounding-DINO: Open-vocabulary detection with text prompts
- SAM2: Precise instance segmentation from bbox prompts

Key features:
- Task-aware detection: Query with exact target description
- Precise masks: For 3D localization and grasp planning
- No training needed: Works zero-shot on any object
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from .interface import ObjectDetector, Detection
from .grounding_dino_detector import GroundingDINODetector, LIBERO_OBJECT_PROMPTS
from .sam2_segmentor import SAM2Segmentor, SegmentationResult, mask_centroid_3d


@dataclass
class InstanceDetection(Detection):
    """Extended detection with segmentation mask."""
    mask: Optional[np.ndarray] = None  # Binary mask (H, W)
    mask_score: float = 0.0  # SAM2 mask confidence
    tight_bbox: Optional[List[float]] = None  # Bbox from mask (more accurate)
    area: int = 0  # Mask area in pixels


@dataclass
class GroundedSAMDetector(ObjectDetector):
    """Combined Grounding-DINO + SAM2 detector.

    Usage:
        detector = GroundedSAMDetector()

        # Detect and segment specific target
        detections = detector.detect_target(image, "bbq_sauce")

        # Detect and segment all objects matching prompt
        detector.set_target_objects(["bbq_sauce", "basket", "bowl"])
        detections = detector.detect(image)

        # Get 3D position from mask + depth
        position = detector.get_3d_position(detection, depth_image, camera_intrinsics)
    """

    # Model configuration
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam2_model_size: str = "tiny"  # tiny, base, large
    device: str = "cuda"

    # Detection thresholds
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    # Object prompts
    object_prompts: Dict[str, str] = field(default_factory=lambda: LIBERO_OBJECT_PROMPTS.copy())

    # Internal components
    _gdino: Optional[GroundingDINODetector] = field(default=None, repr=False)
    _sam2: Optional[SAM2Segmentor] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)
    _current_prompt: str = ""

    # Classes
    CLASSES: List[str] = field(default_factory=lambda: list(LIBERO_OBJECT_PROMPTS.keys()))

    def __post_init__(self):
        """Lazy initialization."""
        pass

    def _load_models(self):
        """Load both models."""
        if self._loaded:
            return

        print("Loading Grounded-SAM (Grounding-DINO + SAM2)...")

        # Initialize Grounding-DINO
        self._gdino = GroundingDINODetector(
            model_id=self.gdino_model_id,
            device=self.device,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            object_prompts=self.object_prompts,
        )

        # Initialize SAM2
        self._sam2 = SAM2Segmentor(
            model_size=self.sam2_model_size,
            device=self.device,
        )

        self._loaded = True
        print("Grounded-SAM loaded successfully")

    def set_target_objects(self, object_names: List[str]):
        """Set detection targets.

        Args:
            object_names: List of object names (e.g., ["bbq_sauce", "basket"])
        """
        self._load_models()
        self._gdino.set_target_objects(object_names)

    def set_prompt(self, prompt: str):
        """Set raw text prompt.

        Args:
            prompt: Period-separated, lowercase (e.g., "red bbq sauce bottle. basket.")
        """
        self._load_models()
        self._gdino.set_prompt(prompt)

    def detect(self, rgb_image: np.ndarray) -> List[InstanceDetection]:
        """Detect and segment objects matching current prompt.

        Args:
            rgb_image: RGB image (H, W, 3), uint8

        Returns:
            List of InstanceDetection objects with masks
        """
        self._load_models()

        # Step 1: Grounding-DINO detection
        gdino_dets = self._gdino.detect(rgb_image)

        if not gdino_dets:
            return []

        # Step 2: SAM2 segmentation
        bboxes = [det.bbox for det in gdino_dets]
        class_names = [det.class_name for det in gdino_dets]

        seg_results = self._sam2.segment_bboxes(rgb_image, bboxes, class_names)

        # Step 3: Combine results
        detections = []
        for gdino_det, seg_result in zip(gdino_dets, seg_results):
            det = InstanceDetection(
                class_name=gdino_det.class_name,
                position=gdino_det.position,
                confidence=gdino_det.confidence,
                bbox=gdino_det.bbox,
                mask=seg_result.mask,
                mask_score=seg_result.score,
                tight_bbox=seg_result.bbox,
                area=seg_result.area,
            )
            detections.append(det)

        return detections

    def detect_target(
        self,
        rgb_image: np.ndarray,
        target_object: str,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Dict] = None,
    ) -> List[InstanceDetection]:
        """Detect and segment specific target object.

        Args:
            rgb_image: RGB image (H, W, 3), uint8
            target_object: LIBERO object name (e.g., "bbq_sauce")
            depth_image: Optional depth image for 3D position
            camera_intrinsics: Optional camera params (fx, fy, cx, cy)

        Returns:
            List of InstanceDetection objects for target
        """
        self._load_models()

        # Set target
        self._gdino.set_target_objects([target_object])

        # Detect and segment
        detections = self.detect(rgb_image)

        # Update positions from depth if provided
        if depth_image is not None and camera_intrinsics is not None:
            for det in detections:
                if det.mask is not None:
                    det.position = mask_centroid_3d(
                        det.mask, depth_image, camera_intrinsics
                    )

        return detections

    def detect_all_groceries(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Dict] = None,
    ) -> List[InstanceDetection]:
        """Detect and segment all grocery items.

        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Optional depth for 3D positions
            camera_intrinsics: Optional camera params

        Returns:
            List of InstanceDetection objects
        """
        grocery_items = [
            "bbq_sauce", "ketchup", "tomato_sauce", "cream_cheese",
            "butter", "milk", "orange_juice", "alphabet_soup",
            "salad_dressing", "chocolate_pudding"
        ]

        self.set_target_objects(grocery_items)
        detections = self.detect(rgb_image)

        # Update positions from depth if provided
        if depth_image is not None and camera_intrinsics is not None:
            for det in detections:
                if det.mask is not None:
                    det.position = mask_centroid_3d(
                        det.mask, depth_image, camera_intrinsics
                    )

        return detections

    def get_3d_position(
        self,
        detection: InstanceDetection,
        depth_image: np.ndarray,
        camera_intrinsics: Dict,
    ) -> np.ndarray:
        """Get 3D position from detection mask and depth.

        Args:
            detection: InstanceDetection with mask
            depth_image: Depth image (H, W) in meters
            camera_intrinsics: Dict with fx, fy, cx, cy

        Returns:
            3D position (3,) in camera frame
        """
        if detection.mask is None:
            # Fallback to bbox center
            x1, y1, x2, y2 = detection.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            depth = depth_image[int(cy), int(cx)]

            fx = camera_intrinsics.get('fx', 128)
            fy = camera_intrinsics.get('fy', 128)
            px = camera_intrinsics.get('cx', 128)
            py = camera_intrinsics.get('cy', 128)

            x3d = (cx - px) * depth / fx
            y3d = (cy - py) * depth / fy
            return np.array([x3d, y3d, depth])

        return mask_centroid_3d(detection.mask, depth_image, camera_intrinsics)

    def warmup(self, image_size: tuple = (256, 256, 3)):
        """Run warmup inference."""
        self._load_models()

        dummy_image = np.zeros(image_size, dtype=np.uint8)
        self.set_target_objects(["object"])

        # Warmup both models
        self._gdino.warmup(image_size)

        # SAM2 warmup with dummy bbox
        self._sam2.warmup(image_size)

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        self.box_threshold = threshold
        if self._gdino:
            self._gdino.box_threshold = threshold


def visualize_instance_detection(
    image: np.ndarray,
    detection: InstanceDetection,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Visualize detection with mask overlay.

    Args:
        image: RGB image (H, W, 3)
        detection: InstanceDetection with mask
        color: RGB color for mask overlay

    Returns:
        Annotated image
    """
    import cv2

    vis = image.copy()

    # Draw mask overlay
    if detection.mask is not None:
        mask_overlay = np.zeros_like(vis)
        mask_overlay[detection.mask] = color
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    # Draw bbox
    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Draw tight bbox (from mask)
    if detection.tight_bbox:
        tx1, ty1, tx2, ty2 = [int(v) for v in detection.tight_bbox]
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 0, 0), 1)

    # Draw label
    label = f"{detection.class_name}: {detection.confidence:.2f}"
    cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis
