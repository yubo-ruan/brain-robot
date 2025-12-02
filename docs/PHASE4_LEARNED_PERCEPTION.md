# Phase 4: Learned Perception Implementation Plan

## Goal

Replace oracle perception with learned perception that:
1. Takes RGB images as input (no privileged simulator access)
2. Outputs object poses, spatial relations, and gripper state
3. Maintains the same `PerceptionResult` API as oracle
4. Achieves ≥80% task success rate (vs 100% with oracle)

---

## Critical Design Decisions

### 1. Class vs Instance ID Separation

The detector outputs **object classes** (bowl, plate, cabinet), not instance IDs.
A separate **tracking layer** maintains stable instance IDs across frames.

```
Detection: class="bowl", bbox=[100,200,150,250], confidence=0.95
     ↓
Tracking: matches to existing track → instance_id="akita_black_bowl_1_main"
     ↓
PerceptionResult.objects["akita_black_bowl_1_main"] = pose
```

**Rationale**: Training YOLO to distinguish `bowl_1` vs `bowl_2` is brittle and unnecessary - they're visually identical. The tracker handles identity.

### 2. Object Tracking (Required Component)

Simple nearest-neighbor tracking in 3D:
- For each detection, match to existing track with same class and closest position
- If no match within threshold, spawn new track with new instance ID
- Tracks persist across frames, providing stable IDs to WorldState

```python
@dataclass
class ObjectTrack:
    instance_id: str           # e.g., "akita_black_bowl_1_main"
    class_name: str            # e.g., "bowl"
    last_pose: np.ndarray      # Last known 3D position
    last_seen: float           # Timestamp
    confidence: float          # Detection confidence
```

### 3. Episode-Level Data Splits

**Critical**: Split train/val by `(task_id, episode_idx)`, NOT by individual frames.

Frames within an episode are highly correlated. Frame-level splits cause data leakage and inflated metrics.

```python
# WRONG: Frame-level split (data leakage)
random.shuffle(all_frames)
train = all_frames[:split]

# CORRECT: Episode-level split
episodes = group_by(all_frames, key=lambda f: (f.task_id, f.episode_idx))
random.shuffle(episodes)
train_episodes = episodes[:split]
```

### 4. Train/Val Split Policy (Phase 4)

**Policy**: Within-task generalization with episode-level splits.

- **What we're testing**: Can learned perception generalize to new episodes of the same 6 LIBERO spatial tasks?
- **Split method**: Mix episodes from all tasks, split 90/10 by episode (not frame)
- **What this measures**: Robustness to viewpoint/configuration variation within known task types

**What this does NOT measure**:
- Cross-task generalization (novel tasks, unseen objects)
- Sim-to-real transfer

**Future work**: For cross-task generalization, use task-held-out splits (e.g., train on tasks 0-4, test on task 5).

---

## Architecture Overview

```
RGB Image(s)
     ↓
┌────────────────────────────────────────────┐
│       Learned Perception Pipeline          │
│                                            │
│  ┌─────────────┐    ┌──────────────┐      │
│  │ Object      │    │ Pose         │      │
│  │ Detection   │───▶│ Estimation   │      │
│  │ (YOLO)      │    │ (Centroid)   │      │
│  └─────────────┘    └──────────────┘      │
│         │                  │              │
│         ▼                  ▼              │
│  ┌─────────────┐    ┌──────────────┐      │
│  │ Object      │    │ Spatial      │      │
│  │ Tracker     │───▶│ Relations    │      │
│  │ (NN in 3D)  │    │ (Geometric)  │      │
│  └─────────────┘    └──────────────┘      │
│                            │              │
│                     ┌──────────────┐      │
│                     │ Gripper      │      │
│                     │ State        │      │
│                     │ (Proprio)    │      │
│                     └──────────────┘      │
│                                            │
└────────────────────────────────────────────┘
     ↓
PerceptionResult (same API as oracle)
```

---

## Phase 4.1: Data Collection Infrastructure

**Duration**: 2-3 days

### Deliverables

#### 1. Image Capture During Phase 1-3 Execution

```python
@dataclass
class PerceptionDataPoint:
    """Single training example for learned perception."""

    # Input
    rgb_image: np.ndarray  # (H, W, 3) uint8
    depth_image: Optional[np.ndarray]  # (H, W) float32, optional

    # Ground truth from oracle
    object_poses: Dict[str, np.ndarray]  # {name: 7D pose}
    object_bboxes: Dict[str, np.ndarray]  # {name: [x1, y1, x2, y2]}
    gripper_pose: np.ndarray  # 7D pose
    gripper_width: float

    # Spatial relations (from oracle)
    on_relations: Dict[str, str]
    inside_relations: Dict[str, str]

    # Metadata
    task_id: int
    episode_idx: int
    step_idx: int
    camera_name: str  # e.g., "agentview", "robot0_eye_in_hand"
```

#### 2. Data Collection Hook

```python
class PerceptionDataCollector:
    """Collect perception training data during episode execution."""

    def __init__(self, output_dir: str, cameras: List[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = cameras or ["agentview"]
        self.data_points: List[PerceptionDataPoint] = []

    def collect_frame(
        self,
        env,
        oracle_result: PerceptionResult,
        task_id: int,
        episode_idx: int,
        step_idx: int
    ):
        """Collect one training frame."""
        for camera in self.cameras:
            # Get image from robosuite
            rgb = env.render(mode="rgb_array", camera_name=camera)

            # Get bounding boxes (project 3D poses to 2D)
            bboxes = self._compute_bboxes(env, oracle_result.objects, camera)

            data_point = PerceptionDataPoint(
                rgb_image=rgb,
                depth_image=None,  # Optional, add later if needed
                object_poses=oracle_result.objects,
                object_bboxes=bboxes,
                gripper_pose=oracle_result.gripper_pose,
                gripper_width=oracle_result.gripper_width,
                on_relations=dict(oracle_result.on),
                inside_relations=dict(oracle_result.inside),
                task_id=task_id,
                episode_idx=episode_idx,
                step_idx=step_idx,
                camera_name=camera,
            )
            self.data_points.append(data_point)

    def _compute_bboxes(self, env, objects, camera) -> Dict[str, np.ndarray]:
        """Project 3D poses to 2D bounding boxes."""
        # Use MuJoCo camera intrinsics
        # Project object center + estimate box from object size
        # ...
        pass

    def save_dataset(self, split_ratio: float = 0.9):
        """Save collected data as train/val split (episode-level)."""
        # Group frames by episode (CRITICAL: avoid data leakage)
        episodes = defaultdict(list)
        for dp in self.data_points:
            key = (dp.task_id, dp.episode_idx)
            episodes[key].append(dp)

        # Shuffle and split at episode level
        episode_keys = list(episodes.keys())
        random.shuffle(episode_keys)
        split = int(len(episode_keys) * split_ratio)

        train_keys = episode_keys[:split]
        val_keys = episode_keys[split:]

        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"

        for key in train_keys:
            for dp in episodes[key]:
                self._save_data_point(dp, train_dir)
        for key in val_keys:
            for dp in episodes[key]:
                self._save_data_point(dp, val_dir)

        print(f"Saved {len(train_keys)} train episodes, {len(val_keys)} val episodes")
```

#### 3. Keyframe Selection Strategy

Not every frame is equally useful. Collect at:
- **Pre-grasp**: Robot approaching object (varied viewpoints)
- **Grasp contact**: Just before/after gripper closes
- **Transport**: Object in gripper, moving
- **Pre-place**: Above target location
- **Post-place**: After release (verify placement)

```python
class KeyframeSelector:
    """Select informative frames for training."""

    def __init__(self, interval: int = 20):
        self.interval = interval  # Minimum steps between keyframes
        self.last_keyframe = -interval

    def should_collect(self, step: int, world_state: WorldState) -> bool:
        """Decide if current frame should be collected."""
        if step - self.last_keyframe < self.interval:
            return False

        # Always collect at state transitions
        # (holding changed, object placed, etc.)
        # Implementation tracks state changes

        self.last_keyframe = step
        return True
```

---

## Phase 4.2: Object Detection Module

**Duration**: 3-4 days

### Option A: Fine-tune Existing Detector (Recommended Start)

Use YOLO or GroundingDINO with LIBERO object classes.

```python
class YOLOObjectDetector:
    """YOLO-based object detector for LIBERO objects."""

    # LIBERO object classes
    CLASSES = [
        "bowl", "plate", "mug", "ramekin", "cabinet",
        "drawer", "cookie_box", "can", "bottle"
    ]

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold

    def detect(self, rgb_image: np.ndarray) -> List[Detection]:
        """Detect objects in image."""
        results = self.model(rgb_image, conf=self.conf_threshold)

        detections = []
        for box in results[0].boxes:
            det = Detection(
                class_name=self.CLASSES[int(box.cls)],
                bbox=box.xyxy[0].cpu().numpy(),
                confidence=float(box.conf),
            )
            detections.append(det)
        return detections
```

### Option B: Zero-Shot Detection with SAM + CLIP

For more flexibility without training:

```python
class SAMCLIPDetector:
    """Segment Anything + CLIP for zero-shot detection."""

    def __init__(self):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import clip

        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")

    def detect(self, rgb_image: np.ndarray, object_queries: List[str]) -> List[Detection]:
        """Detect objects matching text queries."""
        # 1. Generate all masks
        masks = self.mask_generator.generate(rgb_image)

        # 2. Crop each mask region and classify with CLIP
        detections = []
        for mask in masks:
            crop = self._crop_mask_region(rgb_image, mask)
            clip_input = self.clip_preprocess(crop).unsqueeze(0)

            # Classify with CLIP
            text_inputs = clip.tokenize(object_queries)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(clip_input)
                text_features = self.clip_model.encode_text(text_inputs)
                similarity = (image_features @ text_features.T).softmax(dim=-1)

            best_idx = similarity.argmax().item()
            if similarity[0, best_idx] > 0.5:
                detections.append(Detection(
                    class_name=object_queries[best_idx],
                    mask=mask['segmentation'],
                    confidence=float(similarity[0, best_idx]),
                ))

        return detections
```

---

## Phase 4.3: Pose Estimation Module

**Duration**: 4-5 days

### Approach: Keypoint-Based 6DoF Pose

For each detected object, estimate 6DoF pose from keypoints.

```python
@dataclass
class KeypointPoseEstimator:
    """Estimate 6DoF pose from detected keypoints."""

    # Define keypoint templates for each object type
    KEYPOINT_TEMPLATES = {
        "bowl": {
            "center": [0, 0, 0],
            "rim_front": [0.05, 0, 0.03],
            "rim_back": [-0.05, 0, 0.03],
            "rim_left": [0, 0.05, 0.03],
            "rim_right": [0, -0.05, 0.03],
        },
        "plate": {
            "center": [0, 0, 0],
            "edge_front": [0.08, 0, 0],
            "edge_back": [-0.08, 0, 0],
        },
        # ...
    }

    def __init__(self, model_path: str):
        """Load keypoint detection model."""
        # Could be a simple CNN or transformer
        self.model = self._load_model(model_path)

    def estimate_pose(
        self,
        rgb_crop: np.ndarray,
        object_type: str,
        camera_intrinsics: np.ndarray
    ) -> np.ndarray:
        """Estimate 7D pose from image crop."""

        # 1. Detect keypoints in image
        keypoints_2d = self.model(rgb_crop)  # [(u, v), ...]

        # 2. Get 3D template
        template_3d = self.KEYPOINT_TEMPLATES[object_type]

        # 3. Solve PnP for pose
        success, rvec, tvec = cv2.solvePnP(
            np.array(list(template_3d.values())),
            keypoints_2d,
            camera_intrinsics,
            distCoeffs=None
        )

        if not success:
            return None

        # Convert to 7D pose
        R, _ = cv2.Rodrigues(rvec)
        quat = rotation_matrix_to_quaternion(R)
        pose = np.concatenate([tvec.flatten(), quat])

        return pose
```

### Alternative: Direct Pose Regression

Simpler but may be less accurate:

```python
class DirectPoseRegressor:
    """Directly regress pose from image features."""

    def __init__(self, backbone: str = "resnet18"):
        self.backbone = torchvision.models.resnet18(pretrained=True)
        # Replace final layer for pose output
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # xyz + quaternion
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        """Predict 7D pose from object crop."""
        pose = self.backbone(crop)
        # Normalize quaternion
        pose[:, 3:7] = F.normalize(pose[:, 3:7], dim=-1)
        return pose
```

---

## Phase 4.4: Spatial Relation Inference

**Duration**: 1-2 days

### Option A: Geometric Heuristics (Reuse from Oracle)

```python
class GeometricSpatialRelations:
    """Compute spatial relations from estimated poses."""

    def compute(self, object_poses: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """Same logic as oracle._extract_spatial_relations."""
        # Reuse the existing geometric heuristics
        # Just operates on learned poses instead of GT poses

        on = {}
        inside = {}

        # ... (copy logic from oracle.py)

        return on, inside
```

### Option B: Learned Spatial Relations

Train a classifier on object pair features:

```python
class LearnedSpatialRelations:
    """Classify spatial relations from visual features."""

    RELATIONS = ["on", "inside", "next_to", "none"]

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def classify_pair(
        self,
        obj_a_features: np.ndarray,
        obj_b_features: np.ndarray,
        obj_a_pose: np.ndarray,
        obj_b_pose: np.ndarray
    ) -> str:
        """Classify spatial relation between two objects."""

        # Combine features
        combined = np.concatenate([
            obj_a_features,
            obj_b_features,
            obj_a_pose - obj_b_pose,  # Relative pose
        ])

        logits = self.model(combined)
        relation_idx = logits.argmax()
        return self.RELATIONS[relation_idx]
```

---

## Phase 4.5: Gripper State Estimation

**Duration**: 1 day

### Proprioception (No Vision Needed)

```python
class GripperStateEstimator:
    """Estimate gripper state from proprioception."""

    def estimate(self, obs: dict) -> Tuple[np.ndarray, float]:
        """Get gripper pose and width from observation."""

        # These are typically in the observation dict directly
        gripper_pose = np.concatenate([
            obs['robot0_eef_pos'],
            obs['robot0_eef_quat']
        ])

        gripper_width = np.sum(np.abs(obs['robot0_gripper_qpos']))

        return gripper_pose, gripper_width
```

Note: Gripper state comes from proprioception, not vision. This remains accurate even with learned perception.

---

## Phase 4.6: Integration & Testing

**Duration**: 2-3 days

### Unified Learned Perception Class

```python
class LearnedPerception(PerceptionInterface):
    """Learned perception following oracle API."""

    def __init__(
        self,
        detector: ObjectDetector,
        pose_estimator: PoseEstimator,
        camera_name: str = "agentview"
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.camera_name = camera_name
        self.spatial_computer = GeometricSpatialRelations()
        self.gripper_estimator = GripperStateEstimator()

    def perceive(self, env) -> PerceptionResult:
        """Learned perception - same API as oracle."""

        result = PerceptionResult(timestamp=time.time())

        # 1. Get image
        rgb = env.render(mode="rgb_array", camera_name=self.camera_name)
        camera_matrix = self._get_camera_matrix(env)

        # 2. Detect objects
        detections = self.detector.detect(rgb)

        # 3. Estimate poses
        for det in detections:
            crop = self._crop_detection(rgb, det.bbox)
            pose = self.pose_estimator.estimate_pose(
                crop, det.class_name, camera_matrix
            )
            if pose is not None:
                # Generate unique ID for this detection
                obj_id = self._generate_object_id(det)
                result.objects[obj_id] = pose
                result.object_names.append(obj_id)

        # 4. Compute spatial relations
        result.on, result.inside = self.spatial_computer.compute(result.objects)

        # 5. Get gripper state (from proprioception)
        obs = env._get_observations()
        result.gripper_pose, result.gripper_width = self.gripper_estimator.estimate(obs)

        return result

    def _generate_object_id(self, detection: Detection) -> str:
        """Generate consistent object ID from detection."""
        # Need to track objects across frames
        # Could use simple position-based matching or learned re-ID
        # For now, use class + instance counter
        pass
```

### Evaluation Protocol

```python
def evaluate_learned_perception(
    oracle: OraclePerception,
    learned: LearnedPerception,
    env,
    n_episodes: int = 50
) -> Dict[str, float]:
    """Compare learned vs oracle perception."""

    metrics = {
        "position_error_mean": [],
        "position_error_std": [],
        "orientation_error_mean": [],
        "detection_recall": [],
        "detection_precision": [],
        "spatial_on_accuracy": [],
        "spatial_inside_accuracy": [],
    }

    for episode in range(n_episodes):
        env.reset()

        oracle_result = oracle.perceive(env)
        learned_result = learned.perceive(env)

        # Match objects between oracle and learned
        matches = match_objects(oracle_result.objects, learned_result.objects)

        # Compute position errors
        for oracle_id, learned_id in matches:
            oracle_pos = oracle_result.objects[oracle_id][:3]
            learned_pos = learned_result.objects[learned_id][:3]
            error = np.linalg.norm(oracle_pos - learned_pos)
            metrics["position_error_mean"].append(error)

        # Compute spatial relation accuracy
        # ...

    return {k: np.mean(v) for k, v in metrics.items()}
```

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Position error | <3 cm | Mean L2 distance |
| Orientation error | <15° | Angular error |
| Detection recall | >90% | Miss rate for key objects |
| Detection precision | >85% | False positive rate |
| Spatial ON accuracy | >85% | Matches oracle |
| Spatial INSIDE accuracy | >80% | Harder due to occlusion |
| Task success rate | ≥80% | vs 100% with oracle |
| Inference latency | <50 ms | Per frame, GPU |

---

## Recommended Implementation Order

1. **4.1 Data Collection** (2-3 days)
   - Add image capture to existing Phase 1-3 execution
   - Collect ~5000 frames across all 6 tasks
   - Verify data format and quality

2. **4.5 Gripper State** (1 day)
   - Simplest component - use proprioception
   - Test integration with PerceptionResult

3. **4.2 Object Detection** (3-4 days)
   - Start with fine-tuned YOLO on collected data
   - Measure detection recall/precision
   - Iterate on augmentation if needed

4. **4.3 Pose Estimation** (4-5 days)
   - Start with direct regression (simpler)
   - Measure pose error
   - Try keypoint method if accuracy insufficient

5. **4.4 Spatial Relations** (1-2 days)
   - Reuse geometric heuristics from oracle
   - Verify accuracy matches oracle on GT poses
   - Test with learned poses

6. **4.6 Integration** (2-3 days)
   - Combine all modules into LearnedPerception
   - Run full evaluation protocol
   - Measure task success rate

---

## Implementation Status

### Phase 4.1: Data Collection ✅ COMPLETE
- Implemented `PerceptionDataCollector` with episode-level train/val splits
- Implemented `KeyframeSelector` for efficient sampling
- Collected 2040 frames across 6 LIBERO spatial tasks (120 episodes)
- Dataset saved to `data/perception_v1/` with `train.json`, `val.json`, and `images/`

### Phase 4.2: Object Detection 🔄 IN PROGRESS
- Implemented tracker interface (`brain_robot/perception/tracking/`)
  - `Detection` dataclass for detector outputs
  - `ObjectTrack` dataclass for persistent tracks
  - `NearestNeighborTracker` for 3D position-based association
- Implemented `YOLOObjectDetector` class
- Created YOLO format dataset converter (`scripts/convert_to_yolo_format.py`)
- Created training script (`scripts/train_yolo_detector.py`)
- Created evaluation script (`scripts/eval_yolo_detector.py`)

**Environment Note**: YOLO training requires compatible torch/torchvision versions.
If you encounter NMS errors, ensure matching versions:
```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install ultralytics
```

### Phase 4.3-4.6: TODO
- Pose estimation module
- Spatial relation inference
- Gripper state estimation
- Full integration and testing

---

## Risk Mitigation

### Risk: Detection misses key objects

**Mitigation**:
- Aggressive data augmentation
- Multiple detection thresholds
- Fallback to oracle if confidence low

### Risk: Pose estimation too noisy

**Mitigation**:
- Temporal smoothing across frames
- Use multiple camera views
- Ensemble of pose estimators

### Risk: Object tracking drift

**Mitigation**:
- Re-initialize tracking at state transitions
- Use appearance + position for matching
- Accept occasional re-detection as new object

### Risk: Latency too high

**Mitigation**:
- Profile each component
- Use TensorRT/ONNX optimization
- Reduce input resolution if needed
- Skip frames during transport (less critical)

---

## Hardware Requirements

- GPU with ≥8GB VRAM (detection + pose models)
- ~10GB disk for training data
- ~2GB for trained models

Current setup should be sufficient if detection and pose models are reasonably sized.

---

## Timeline Summary

| Phase | Task | Duration |
|-------|------|----------|
| 4.1 | Data collection infrastructure | 2-3 days |
| 4.2 | Object detection module | 3-4 days |
| 4.3 | Pose estimation module | 4-5 days |
| 4.4 | Spatial relation inference | 1-2 days |
| 4.5 | Gripper state estimation | 1 day |
| 4.6 | Integration & testing | 2-3 days |
| **Total** | | **~2-2.5 weeks** |
