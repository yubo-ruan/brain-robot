"""SAM2 (Segment Anything Model 2) wrapper for instance segmentation.

Provides precise segmentation masks from bounding box prompts.
Designed to work with Grounding-DINO detections for full instance segmentation.

Key features:
- Bbox-prompted segmentation (from Grounding-DINO)
- Precise object masks for 3D localization
- Multiple model sizes (tiny, base, large)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class SegmentationResult:
    """Result from SAM2 segmentation."""
    mask: np.ndarray  # Binary mask (H, W), bool
    score: float  # Confidence score
    bbox: List[float]  # [x1, y1, x2, y2] tight bbox from mask
    class_name: str  # From detection
    area: int  # Number of pixels in mask


@dataclass
class SAM2Segmentor:
    """SAM2 wrapper for instance segmentation.

    Usage:
        segmentor = SAM2Segmentor()

        # Segment from bbox
        results = segmentor.segment_bbox(image, bbox=[100, 100, 200, 200])

        # Segment multiple objects
        results = segmentor.segment_bboxes(image, bboxes=[[100,100,200,200], [300,150,450,350]])
    """

    # Model configuration
    model_size: str = "tiny"  # tiny, base, large
    device: str = "cuda"

    # Internal state
    _model: Optional[object] = field(default=None, repr=False)
    _predictor: Optional[object] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)
    _current_image: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Lazy loading - model loaded on first use."""
        pass

    def _load_model(self):
        """Load SAM2 model."""
        if self._loaded:
            return

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Model configs
            model_configs = {
                "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
                "base": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
                "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
            }

            checkpoint, config = model_configs.get(self.model_size, model_configs["tiny"])

            print(f"Loading SAM2 model: {self.model_size}")

            # Try to load from checkpoints directory
            checkpoint_path = f"checkpoints/{checkpoint}"

            # Build model
            sam2_model = build_sam2(config, checkpoint_path, device=self.device)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._model = sam2_model

            self._loaded = True
            print(f"SAM2 {self.model_size} loaded on {self.device}")

        except ImportError as e:
            print(f"SAM2 not installed. Using HuggingFace fallback...")
            print(f"(Error: {e})")
            self._load_model_hf()
        except Exception as e:
            print(f"Failed to load SAM2: {e}")
            print("Trying HuggingFace fallback...")
            self._load_model_hf()

    def _load_model_hf(self):
        """Load SAM2 from HuggingFace (fallback)."""
        try:
            import torch
            from transformers import SamModel, SamProcessor

            model_ids = {
                "tiny": "facebook/sam2-hiera-tiny",
                "base": "facebook/sam2-hiera-base-plus",
                "large": "facebook/sam2-hiera-large",
            }

            model_id = model_ids.get(self.model_size, model_ids["tiny"])
            print(f"Loading SAM2 from HuggingFace: {model_id}")

            self._processor = SamProcessor.from_pretrained(model_id)
            self._model = SamModel.from_pretrained(model_id).to(self.device)
            self._model.eval()

            self._loaded = True
            self._use_hf = True
            print(f"SAM2 (HuggingFace) loaded on {self.device}")

        except Exception as e:
            print(f"Failed to load SAM2 from HuggingFace: {e}")
            raise

    def set_image(self, image: np.ndarray):
        """Set image for segmentation (enables batched bbox queries).

        Args:
            image: RGB image (H, W, 3), uint8
        """
        self._load_model()

        # Ensure contiguous array (SAM2 requires positive strides)
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        if hasattr(self, '_use_hf') and self._use_hf:
            self._current_image = image
        else:
            import torch
            with torch.inference_mode():
                self._predictor.set_image(image)
            self._current_image = image

    def segment_bbox(
        self,
        image: Optional[np.ndarray] = None,
        bbox: List[float] = None,
        class_name: str = "object",
    ) -> Optional[SegmentationResult]:
        """Segment object from bounding box.

        Args:
            image: RGB image (H, W, 3). If None, uses previously set image.
            bbox: [x1, y1, x2, y2] bounding box
            class_name: Class name for the detection

        Returns:
            SegmentationResult with mask and metadata
        """
        if bbox is None:
            raise ValueError("bbox is required")

        results = self.segment_bboxes(image, [bbox], [class_name])
        return results[0] if results else None

    def segment_bboxes(
        self,
        image: Optional[np.ndarray] = None,
        bboxes: List[List[float]] = None,
        class_names: Optional[List[str]] = None,
    ) -> List[SegmentationResult]:
        """Segment multiple objects from bounding boxes.

        Args:
            image: RGB image (H, W, 3). If None, uses previously set image.
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            class_names: List of class names for each bbox

        Returns:
            List of SegmentationResult objects
        """
        self._load_model()

        if bboxes is None or len(bboxes) == 0:
            return []

        if class_names is None:
            class_names = ["object"] * len(bboxes)

        # Set image if provided
        if image is not None:
            self.set_image(image)

        if self._current_image is None:
            raise ValueError("No image set. Call set_image() or provide image.")

        import torch

        results = []

        if hasattr(self, '_use_hf') and self._use_hf:
            # HuggingFace implementation
            results = self._segment_bboxes_hf(bboxes, class_names)
        else:
            # Native SAM2 implementation
            bbox_array = np.array(bboxes)

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks, scores, _ = self._predictor.predict(
                    box=bbox_array,
                    multimask_output=False,
                )

            # Process results
            for i, (mask, score, bbox, class_name) in enumerate(
                zip(masks, scores, bboxes, class_names)
            ):
                # Handle different output shapes
                if mask.ndim == 3:
                    mask = mask[0]  # Take first mask if multiple

                # Get tight bbox from mask
                tight_bbox = self._mask_to_bbox(mask)

                result = SegmentationResult(
                    mask=mask.astype(bool),
                    score=float(score) if np.isscalar(score) else float(score[0]),
                    bbox=tight_bbox,
                    class_name=class_name,
                    area=int(mask.sum()),
                )
                results.append(result)

        return results

    def _segment_bboxes_hf(
        self,
        bboxes: List[List[float]],
        class_names: List[str],
    ) -> List[SegmentationResult]:
        """Segment using HuggingFace SAM2."""
        import torch
        from PIL import Image

        results = []
        pil_image = Image.fromarray(self._current_image)

        for bbox, class_name in zip(bboxes, class_names):
            # Prepare inputs
            inputs = self._processor(
                pil_image,
                input_boxes=[[[bbox]]],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process
            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            mask = masks[0][0][0].numpy()
            score = float(outputs.iou_scores[0][0][0])

            tight_bbox = self._mask_to_bbox(mask)

            result = SegmentationResult(
                mask=mask.astype(bool),
                score=score,
                bbox=tight_bbox,
                class_name=class_name,
                area=int(mask.sum()),
            )
            results.append(result)

        return results

    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        """Get tight bounding box from mask."""
        if not mask.any():
            return [0, 0, 0, 0]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return [float(x1), float(y1), float(x2), float(y2)]

    def warmup(self, image_size: tuple = (256, 256, 3)):
        """Run warmup inference."""
        self._load_model()

        dummy_image = np.zeros(image_size, dtype=np.uint8)
        dummy_bbox = [50, 50, 150, 150]

        self.segment_bbox(dummy_image, dummy_bbox)


def mask_to_point_cloud(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: Dict,
) -> np.ndarray:
    """Convert mask + depth to 3D point cloud.

    Args:
        mask: Binary mask (H, W)
        depth: Depth image (H, W) in meters
        camera_intrinsics: Dict with fx, fy, cx, cy

    Returns:
        Point cloud (N, 3) in camera frame
    """
    # Get mask pixels
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros((0, 3))

    # Get depth values
    depths = depth[ys, xs]

    # Filter invalid depths
    valid = (depths > 0) & (depths < 10)  # 0-10m range
    xs, ys, depths = xs[valid], ys[valid], depths[valid]

    if len(xs) == 0:
        return np.zeros((0, 3))

    # Back-project to 3D
    fx = camera_intrinsics.get('fx', 128)
    fy = camera_intrinsics.get('fy', 128)
    cx = camera_intrinsics.get('cx', 128)
    cy = camera_intrinsics.get('cy', 128)

    x3d = (xs - cx) * depths / fx
    y3d = (ys - cy) * depths / fy
    z3d = depths

    return np.stack([x3d, y3d, z3d], axis=1)


def mask_centroid_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: Dict,
) -> np.ndarray:
    """Get 3D centroid of masked object.

    Args:
        mask: Binary mask (H, W)
        depth: Depth image (H, W) in meters
        camera_intrinsics: Dict with fx, fy, cx, cy

    Returns:
        3D centroid (3,) in camera frame
    """
    points = mask_to_point_cloud(mask, depth, camera_intrinsics)

    if len(points) == 0:
        return np.zeros(3)

    return points.mean(axis=0)
