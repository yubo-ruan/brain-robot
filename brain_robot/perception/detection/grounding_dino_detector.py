"""Grounding-DINO open-vocabulary object detector.

Uses text prompts to detect objects, solving the grocery misclassification
problem by querying with task-specific descriptions like "red bbq sauce bottle"
instead of relying on learned class IDs.

Key advantages over YOLO:
- Open vocabulary: no fixed class set, can detect novel objects
- Task-aware: query with exact target object description
- No training needed: works zero-shot on any object

Trade-off: ~200ms per image vs ~10ms for YOLO
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .interface import ObjectDetector, Detection


# Descriptive text prompts for LIBERO objects
# More descriptive = better discrimination between similar objects
LIBERO_OBJECT_PROMPTS = {
    # Grocery items (the problematic ones for YOLO)
    "bbq_sauce": "red bbq sauce bottle",
    "ketchup": "red ketchup bottle",
    "tomato_sauce": "tomato sauce can",
    "cream_cheese": "white cream cheese box",
    "butter": "yellow butter box",
    "milk": "white milk carton",
    "orange_juice": "orange juice carton",
    "alphabet_soup": "alphabet soup can",
    "salad_dressing": "salad dressing bottle",
    "chocolate_pudding": "brown chocolate pudding cup",

    # Kitchen objects
    "bowl": "ceramic bowl",
    "plate": "white plate",
    "mug": "coffee mug",
    "ramekin": "small ramekin dish",
    "cabinet": "kitchen cabinet",
    "drawer": "kitchen drawer",
    "stove": "stove burner",
    "microwave": "microwave oven",
    "frying_pan": "frying pan",

    # Other objects
    "basket": "woven basket",
    "moka_pot": "moka coffee pot",
    "book": "book",
    "caddy": "caddy container",
    "white_mug": "white coffee mug",
    "yellow_white_mug": "yellow and white mug",
    "wine_bottle": "wine bottle",
    "wine_rack": "wine rack",
    "cookie_box": "cookie box",
    "can": "metal can",
    "bottle": "bottle",
}


@dataclass
class GroundingDINODetector(ObjectDetector):
    """Grounding-DINO open-vocabulary object detector.

    Uses Hugging Face transformers for simple integration.

    Usage:
        # Detect specific target object
        detector = GroundingDINODetector()
        detections = detector.detect_target(image, "bbq_sauce")

        # Detect all objects in prompt
        detector.set_prompt("bbq sauce bottle. cream cheese box. basket.")
        detections = detector.detect(image)
    """

    # Model configuration
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    # Current detection prompt (period-separated, lowercase)
    _prompt: str = ""

    # Object prompts mapping
    object_prompts: Dict[str, str] = field(default_factory=lambda: LIBERO_OBJECT_PROMPTS.copy())

    # Internal state
    _model: Optional[object] = field(default=None, repr=False)
    _processor: Optional[object] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    # Classes (for compatibility with ObjectDetector interface)
    CLASSES: List[str] = field(default_factory=lambda: list(LIBERO_OBJECT_PROMPTS.keys()))

    def __post_init__(self):
        """Initialize model lazily on first use."""
        pass

    def _load_model(self):
        """Load Grounding-DINO model from Hugging Face."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            print(f"Loading Grounding-DINO model: {self.model_id}")

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            self._loaded = True
            print(f"Grounding-DINO loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            ) from e

    def set_prompt(self, prompt: str):
        """Set detection prompt.

        Args:
            prompt: Period-separated, lowercase object descriptions.
                   Example: "red bbq sauce bottle. white cream cheese box."
        """
        self._prompt = prompt.lower()
        if not self._prompt.endswith("."):
            self._prompt += "."

    def set_target_objects(self, object_names: List[str]):
        """Set prompt from list of LIBERO object names.

        Args:
            object_names: List of object names (e.g., ["bbq_sauce", "cream_cheese"])
                         Accepts both underscore and space versions (e.g., "alphabet_soup" or "alphabet soup")
        """
        prompts = []
        for name in object_names:
            # Normalize: try both underscore and space versions
            name_underscore = name.replace(" ", "_")
            name_space = name.replace("_", " ")

            if name in self.object_prompts:
                prompts.append(self.object_prompts[name])
            elif name_underscore in self.object_prompts:
                prompts.append(self.object_prompts[name_underscore])
            elif name_space in self.object_prompts:
                prompts.append(self.object_prompts[name_space])
            else:
                # Fallback to object name with underscores replaced
                prompts.append(name_space)

        self._prompt = ". ".join(prompts) + "."

    def detect(self, rgb_image: np.ndarray) -> List[Detection]:
        """Detect objects matching current prompt.

        Args:
            rgb_image: RGB image (H, W, 3), uint8

        Returns:
            List of Detection objects
        """
        if not self._prompt:
            raise ValueError("No prompt set. Call set_prompt() or set_target_objects() first.")

        return self._detect_with_prompt(rgb_image, self._prompt)

    def detect_target(self, rgb_image: np.ndarray, target_object: str) -> List[Detection]:
        """Detect specific target object.

        Args:
            rgb_image: RGB image (H, W, 3), uint8
            target_object: LIBERO object name (e.g., "bbq_sauce")

        Returns:
            List of Detection objects for that target
        """
        # Get descriptive prompt for target
        if target_object in self.object_prompts:
            prompt = self.object_prompts[target_object] + "."
        else:
            prompt = target_object.replace("_", " ") + "."

        detections = self._detect_with_prompt(rgb_image, prompt)

        # Map back to canonical class name
        for det in detections:
            det.class_name = target_object

        return detections

    def detect_all_groceries(self, rgb_image: np.ndarray) -> List[Detection]:
        """Detect all grocery items in image.

        Useful for debugging/analysis.
        """
        grocery_items = [
            "bbq_sauce", "ketchup", "tomato_sauce", "cream_cheese",
            "butter", "milk", "orange_juice", "alphabet_soup",
            "salad_dressing", "chocolate_pudding"
        ]
        self.set_target_objects(grocery_items)
        return self.detect(rgb_image)

    def _detect_with_prompt(self, rgb_image: np.ndarray, prompt: str) -> List[Detection]:
        """Run detection with specific prompt.

        Args:
            rgb_image: RGB image (H, W, 3), uint8
            prompt: Text prompt (period-separated, lowercase)

        Returns:
            List of Detection objects
        """
        self._load_model()

        import torch
        from PIL import Image

        # Convert to PIL Image
        if isinstance(rgb_image, np.ndarray):
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = rgb_image

        # Prepare inputs
        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]]  # (height, width)
        )

        # Convert to Detection objects
        detections = []
        if results and len(results) > 0:
            result = results[0]

            for box, score, label in zip(
                result["boxes"],
                result["scores"],
                result["labels"]
            ):
                # Map label back to class name
                class_name = self._label_to_class_name(label)

                det = Detection(
                    class_name=class_name,
                    position=np.zeros(3),  # Placeholder
                    confidence=float(score),
                    bbox=box.cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                )
                detections.append(det)

        return detections

    def _label_to_class_name(self, label: str) -> str:
        """Map detected label back to canonical class name.

        Args:
            label: Detected label text (e.g., "red bbq sauce bottle")

        Returns:
            Canonical class name (e.g., "bbq_sauce")
        """
        label_lower = label.lower().strip()

        # Check if label matches any prompt
        for class_name, prompt in self.object_prompts.items():
            if prompt.lower() in label_lower or label_lower in prompt.lower():
                return class_name

        # Fallback: convert to snake_case
        return label_lower.replace(" ", "_")

    def warmup(self, image_size: tuple = (256, 256, 3)):
        """Run warmup inference."""
        self._load_model()

        dummy_image = np.zeros(image_size, dtype=np.uint8)
        self.set_prompt("object.")

        for _ in range(2):
            self._detect_with_prompt(dummy_image, "object.")

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold."""
        self.box_threshold = threshold


def compare_detectors(
    image: np.ndarray,
    target_object: str,
    yolo_detector: Optional[ObjectDetector] = None,
    gdino_detector: Optional[GroundingDINODetector] = None,
) -> Dict:
    """Compare YOLO vs Grounding-DINO on same image.

    Args:
        image: RGB image
        target_object: Target object to detect
        yolo_detector: Optional YOLO detector
        gdino_detector: Optional Grounding-DINO detector

    Returns:
        Dict with comparison results
    """
    results = {
        "target": target_object,
        "yolo": {"detected": False, "confidence": 0.0, "class": None},
        "gdino": {"detected": False, "confidence": 0.0, "class": None},
    }

    if yolo_detector:
        yolo_dets = yolo_detector.detect(image)
        for det in yolo_dets:
            if det.class_name == target_object:
                results["yolo"]["detected"] = True
                results["yolo"]["confidence"] = max(
                    results["yolo"]["confidence"], det.confidence
                )
                results["yolo"]["class"] = det.class_name

        # Check for misclassification
        if not results["yolo"]["detected"] and yolo_dets:
            best = max(yolo_dets, key=lambda d: d.confidence)
            results["yolo"]["class"] = best.class_name
            results["yolo"]["confidence"] = best.confidence

    if gdino_detector:
        gdino_dets = gdino_detector.detect_target(image, target_object)
        for det in gdino_dets:
            results["gdino"]["detected"] = True
            results["gdino"]["confidence"] = max(
                results["gdino"]["confidence"], det.confidence
            )
            results["gdino"]["class"] = det.class_name

    return results
