# V5 Model Improvement Plan

## Problem Summary

V4 model has high mAP (0.995) on training data but **misclassifies visually similar grocery items** at inference time.

### V4 Training Data Imbalance Analysis

```
Class Distribution (79,200 total annotations):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVER-REPRESENTED (>5000 samples, 8% each):
  cream_cheese   : 5700 (7.2%)  ← Dominates grocery detection
  bowl           : 6300 (8.0%)
  plate          : 6600 (8.3%)
  cabinet        : 6300 (8.0%)
  drawer         : 6300 (8.0%)
  stove          : 6600 (8.3%)

UNDER-REPRESENTED (<2000 samples):
  bbq_sauce      : 1800 (2.3%)  ← Gets confused with cream_cheese
  salad_dressing : 1800 (2.3%)
  chocolate_pudding: 2100 (2.7%)
  moka_pot       :  600 (0.8%)
  book           :  300 (0.4%)
  caddy          :  300 (0.4%)
  microwave      :  300 (0.4%)

MISSING (0 samples):
  can, bottle, frying_pan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Root Cause:** cream_cheese has 3x more samples than bbq_sauce/salad_dressing.
When the model sees a small colored box, it defaults to the most frequent class (cream_cheese).

### Misclassification Pattern

| Target Object | What V4 Detects Instead |
|---------------|------------------------|
| bbq_sauce | cream_cheese, alphabet_soup |
| ketchup | tomato_sauce |
| tomato_sauce | alphabet_soup, cream_cheese |
| butter | cream_cheese, alphabet_soup |
| milk | alphabet_soup, cream_cheese |
| chocolate_pudding | cream_cheese, salad_dressing |
| orange_juice | cream_cheese, salad_dressing |

**Root Causes:**
1. **Identical bounding boxes**: All grocery items have 0.234 × 0.234 boxes (zero variance)
2. **Visual similarity**: Small colorful boxes/bottles look nearly identical at 256×256
3. **No shape discrimination**: Model can't learn object shapes with uniform boxes
4. **Training-inference gap**: High mAP on synthetic train/val, poor generalization

---

## Phase 0: Class Balancing (CRITICAL)

### 0.1 Target Distribution

Balance all grocery classes to ~3000 samples each:

```python
TARGET_SAMPLES = {
    # Over-represented - REDUCE or cap
    "cream_cheese": 3000,      # Was 5700
    "alphabet_soup": 3000,     # Was 2700 (slight increase)

    # Under-represented - INCREASE
    "bbq_sauce": 3000,         # Was 1800 (+67%)
    "salad_dressing": 3000,    # Was 1800 (+67%)
    "ketchup": 3000,           # Was 2700 (+11%)
    "tomato_sauce": 3000,      # Was 2700 (+11%)
    "butter": 3000,            # Was 2400 (+25%)
    "milk": 3000,              # Was 2400 (+25%)
    "chocolate_pudding": 3000, # Was 2100 (+43%)
    "orange_juice": 3000,      # Was 2400 (+25%)
}
```

### 0.2 Data Collection Strategy

Collect additional data specifically from tasks containing under-represented classes:

```python
# Tasks containing bbq_sauce (class 13)
bbq_sauce_tasks = [
    ("libero_object", 3),  # pick_up_the_bbq_sauce_and_place_it_in_the_basket
    ("libero_object", 4),  # Contains bbq_sauce in scene
    ("libero_object", 5),  # Contains bbq_sauce in scene
    ("libero_object", 6),  # Contains bbq_sauce in scene
    ("libero_object", 8),  # Contains bbq_sauce in scene
    ("libero_object", 9),  # Contains bbq_sauce in scene
]

# Collect 50 episodes × 20 frames = 1000 additional images per task
# This should add ~1200 bbq_sauce annotations
```

### 0.3 Downsampling Over-represented Classes

When generating training data, skip frames that only contain over-represented classes:

```python
def should_keep_frame(labels, class_counts):
    """Decide whether to keep a frame based on class balance."""
    over_represented = {"cream_cheese", "bowl", "plate", "cabinet", "drawer", "stove"}

    frame_classes = {get_class_name(label) for label in labels}

    # Keep if frame has ANY under-represented class
    if frame_classes - over_represented:
        return True

    # Skip if ONLY over-represented classes
    return False
```

### 0.4 Class Weights in Training

Use class weights to penalize errors on minority classes:

```python
# YOLOv8 training with class weights
# Higher weight = model penalized more for missing this class
class_weights = {
    "bbq_sauce": 2.0,
    "salad_dressing": 2.0,
    "chocolate_pudding": 1.5,
    "moka_pot": 3.0,
    "book": 3.0,
    "caddy": 3.0,
    "microwave": 3.0,
}
```

---

## Phase 1: Precise Bounding Box Annotations

### 1.1 Implement Segmentation-Based Bounding Boxes

**Current approach:** Fixed-size boxes based on distance from camera
```python
box_size = max(20, min(60, int(180 / dist)))  # Same for all objects
```

**New approach:** Use MuJoCo geom boundaries for precise boxes

```python
def get_precise_bbox(sim, body_name, cam_name, img_size=256):
    """Get tight bounding box using all geoms attached to body."""
    body_id = sim.model.body_name2id(body_name)

    # Get camera matrices
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = sim.model.cam_fovy[cam_id]

    # Project all geom vertices to image
    all_points = []
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == body_id:
            geom_pos = sim.data.geom_xpos[geom_id]
            geom_size = sim.model.geom_size[geom_id]
            geom_type = sim.model.geom_type[geom_id]

            # Get 8 corners of geom bounding box
            corners = get_geom_corners(geom_pos, geom_size, geom_type)

            for corner in corners:
                # Project to image coordinates
                u, v = project_to_image(corner, cam_pos, cam_mat, fovy, img_size)
                if 0 <= u < img_size and 0 <= v < img_size:
                    all_points.append((u, v))

    if not all_points:
        return None

    # Compute tight bounding box
    us, vs = zip(*all_points)
    x1, x2 = max(0, min(us)), min(img_size, max(us))
    y1, y2 = max(0, min(vs)), min(img_size, max(vs))

    return [x1, y1, x2, y2]
```

**Expected outcome:** Different objects will have different aspect ratios:
- `butter`: wide, flat box (e.g., 0.15 × 0.08)
- `milk`: tall, narrow box (e.g., 0.08 × 0.20)
- `alphabet_soup`: square can (e.g., 0.10 × 0.12)

### 1.2 Object-Specific Size Database

Create a lookup table with actual object dimensions:

```python
OBJECT_DIMENSIONS = {
    # name: (width, height, depth) in meters
    "butter": (0.12, 0.03, 0.06),      # Flat rectangle
    "milk": (0.07, 0.20, 0.07),        # Tall carton
    "alphabet_soup": (0.08, 0.10, 0.08), # Squat can
    "cream_cheese": (0.10, 0.05, 0.05),  # Small box
    "ketchup": (0.06, 0.18, 0.06),      # Tall bottle
    "bbq_sauce": (0.07, 0.15, 0.07),    # Medium bottle
    "orange_juice": (0.08, 0.22, 0.08), # Tall carton
    "chocolate_pudding": (0.08, 0.06, 0.08), # Small cup
    # ... etc
}
```

---

## Phase 2: Higher Resolution Training

### 2.1 Increase Image Resolution

**Current:** 256 × 256 pixels
**Target:** 512 × 512 pixels (or 640 × 640 for YOLOv8 native)

```python
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 512,  # Was 256
    "camera_widths": 512,   # Was 256
}
```

**Trade-offs:**
- (+) 4× more pixels = better texture discrimination
- (+) Model can see label text, color gradients
- (-) 4× more training time
- (-) 4× more storage (~48k images at 512×512)

### 2.2 Multi-Scale Training

Train with images at multiple scales to learn scale-invariant features:

```python
# In training config
imgsz: [512, 640]  # Random scale during training
scale: 0.5         # Scale augmentation range
```

---

## Phase 3: Data Augmentation for Discrimination

### 3.1 Class-Aware Augmentation

Apply different augmentations to similar classes to force discrimination:

```python
# Color jitter with class-specific parameters
augmentations = {
    "ketchup": {"hue": 0.1, "saturation": 0.3},  # Red tones
    "bbq_sauce": {"hue": 0.05, "saturation": 0.2}, # Brown tones
    "tomato_sauce": {"hue": 0.08, "saturation": 0.25}, # Orange-red
}
```

### 3.2 Mosaic with Confusing Classes

Create mosaic images that include confusing class pairs:

```python
confusing_pairs = [
    ("ketchup", "tomato_sauce", "bbq_sauce"),
    ("butter", "cream_cheese"),
    ("milk", "orange_juice"),
    ("chocolate_pudding", "cream_cheese"),
]

# Ensure training batches contain these pairs for hard negative mining
```

### 3.3 Copy-Paste Augmentation

Paste objects from one scene into another to increase diversity:

```python
# YOLOv8 built-in copy-paste
copy_paste: 0.3  # 30% probability
```

---

## Phase 4: Model Architecture Changes

### 4.1 Use Larger Model

**Current:** YOLOv8s (11.15M parameters)
**Option A:** YOLOv8m (25.9M parameters) - 2× capacity
**Option B:** YOLOv8l (43.7M parameters) - 4× capacity

```bash
python scripts/train_yolo_detector.py \
    --model yolov8m.pt \
    --data data/yolo_v5/data.yaml \
    --epochs 150
```

### 4.2 Fine-Grained Classification Head

Add auxiliary classification head focused on confusing classes:

```python
# Custom YOLO head with grocery-specific classifier
class GroceryClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),  # 10 grocery classes
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

---

## Phase 5: Improved Evaluation Pipeline

### 5.1 Target-Aware Evaluation Script

Create evaluation that checks if the **correct target object** is detected:

```python
def evaluate_task_detection(model_path, suite, task_id):
    """Evaluate detection with ground truth target object."""
    env, task = make_libero_env(suite, task_id)

    # Extract target object from task name
    # e.g., "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    target_object = extract_target_from_task_name(task.name)

    detector = YOLOObjectDetector(model_path=model_path)

    results = {
        "task_name": task.name,
        "target_object": target_object,
        "target_detected": False,
        "target_confidence": 0.0,
        "misclassifications": [],
        "all_detections": []
    }

    env.reset()
    for step in range(50):
        img = get_observation(env)
        detections = detector.detect(img)

        for det in detections:
            results["all_detections"].append({
                "class": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox
            })

            if det.class_name == target_object:
                results["target_detected"] = True
                results["target_confidence"] = max(
                    results["target_confidence"],
                    det.confidence
                )
            elif is_similar_class(det.class_name, target_object):
                results["misclassifications"].append({
                    "detected": det.class_name,
                    "expected": target_object,
                    "confidence": det.confidence
                })

        # Random action
        env.step(random_action())

    env.close()
    return results


def is_similar_class(detected, expected):
    """Check if detected class is a known confusion with expected."""
    confusion_groups = [
        {"ketchup", "tomato_sauce", "bbq_sauce"},
        {"butter", "cream_cheese", "chocolate_pudding"},
        {"milk", "orange_juice"},
        {"alphabet_soup", "tomato_sauce"},
    ]

    for group in confusion_groups:
        if detected in group and expected in group:
            return True
    return False
```

### 5.2 Confusion Matrix Visualization

Generate per-class confusion matrix during evaluation:

```python
def generate_confusion_report(results):
    """Generate confusion matrix for grocery items."""
    grocery_classes = [
        "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
        "ketchup", "tomato_sauce", "butter", "milk",
        "chocolate_pudding", "orange_juice"
    ]

    # Build confusion matrix
    confusion = np.zeros((len(grocery_classes), len(grocery_classes)))

    for r in results:
        expected_idx = grocery_classes.index(r["target_object"])
        for det in r["all_detections"]:
            if det["class"] in grocery_classes:
                detected_idx = grocery_classes.index(det["class"])
                confusion[expected_idx, detected_idx] += 1

    # Visualize
    plot_confusion_matrix(confusion, grocery_classes)
```

---

## Phase 6: Fallback Strategies

### 6.1 Two-Stage Detection

If single-stage YOLO fails, use two-stage approach:

1. **Stage 1:** Detect "grocery_item" (all 10 classes merged)
2. **Stage 2:** Classify the cropped region with specialized classifier

```python
class TwoStageGroceryDetector:
    def __init__(self):
        self.detector = YOLOObjectDetector(...)  # Detects generic "grocery"
        self.classifier = GroceryClassifier(...)  # 10-class classifier

    def detect(self, img):
        # Stage 1: Find grocery regions
        grocery_regions = self.detector.detect(img)

        results = []
        for region in grocery_regions:
            # Stage 2: Classify each region
            crop = img[region.y1:region.y2, region.x1:region.x2]
            crop_resized = cv2.resize(crop, (64, 64))
            class_probs = self.classifier(crop_resized)
            class_name = GROCERY_CLASSES[class_probs.argmax()]

            results.append(Detection(
                class_name=class_name,
                confidence=class_probs.max(),
                bbox=region.bbox
            ))

        return results
```

### 6.2 Spatial Priors

Use task context to disambiguate:

```python
def apply_spatial_priors(detections, task_name):
    """Use task context to resolve ambiguous detections."""
    target = extract_target_from_task_name(task_name)

    # If we detect something in the target's expected location
    # but with wrong class, consider reclassifying
    for det in detections:
        if is_confusable_with(det.class_name, target):
            # Boost confidence if it's in expected location
            # (e.g., target is usually in center-bottom of frame)
            if is_in_expected_region(det.bbox, target):
                det.class_name = target
                det.confidence *= 0.8  # Mark as uncertain

    return detections
```

---

## Implementation Timeline

### Step 1: Target-Aware Evaluation Script (FIRST)
1. Create `scripts/evaluate_target_detection.py`
2. Extract target object from task name
3. Check if target is detected (not just any object)
4. Track misclassifications and generate confusion matrix
5. **Run on V4 to establish baseline**

### Step 2: Class-Balanced Data Collection
1. Collect additional data for under-represented classes:
   - bbq_sauce: +1200 samples (from 1800 → 3000)
   - salad_dressing: +1200 samples
   - chocolate_pudding: +900 samples
2. Downsample over-represented classes (cream_cheese, etc.)
3. Target: All grocery classes at ~3000 samples

### Step 3: Precise Bounding Boxes
1. Implement geom-based bounding box extraction
2. Re-collect data with accurate box sizes
3. Verify variance per class (target: std > 0.05)

### Step 4: Training V5
1. Train YOLOv8s with balanced data
2. If still failing, try YOLOv8m
3. Use class weights for remaining minority classes

### Step 5: Evaluation & Iteration
1. Run target-aware evaluation on V5
2. Check misclassification rate dropped from ~70% to <10%
3. If issues persist, implement two-stage detection

---

## Success Criteria

| Metric | V4 Current | V5 Target |
|--------|------------|-----------|
| libero_object target detection | 3/10 | 10/10 |
| Grocery misclassification rate | ~70% | <10% |
| Bounding box size variance | 0.000 | >0.05 |
| Average target confidence | 0.00 | >0.5 |
| Class imbalance ratio (max/min) | 3.2x | <1.5x |
| bbq_sauce samples | 1800 | 3000 |
| salad_dressing samples | 1800 | 3000 |

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `scripts/collect_v5_data.py` | New data collection with precise bboxes |
| `scripts/evaluate_target_detection.py` | Target-aware evaluation |
| `scripts/visualize_confusion.py` | Confusion matrix plotting |
| `brain_robot/perception/detection/two_stage_detector.py` | Fallback detector |
| `data/yolo_v5/` | New training data at 512×512 |
