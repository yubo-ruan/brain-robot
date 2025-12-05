# MOKA-Guided Learned Skill Policies — Implementation Log

## Implementation Status

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Phase 0 | ✅ COMPLETE | 2025-12-04 | 100% success on LIBERO tasks |
| Phase 1 | ✅ COMPLETE | 2025-12-04 | HDF5 logger + replay viewer |
| Phase 2 | 🔄 IN PROGRESS | - | Scripted Expert + Data |
| Phase 3 | ⏳ PENDING | - | Tracking + Geometry |
| Phase 4 | ⏳ PENDING | - | Pick-and-Place Policy |
| Phase 5 | ⏳ PENDING | - | Contact Skills |
| Phase 6 | ⏳ PENDING | - | DAgger |

---

## Phase 0: MOKA Validation — COMPLETE ✅

### Results Summary
- **Date**: 2025-12-04
- **Total tasks tested**: 4 (LIBERO tasks 4, 5, 8, 9)
- **Success rate**: 100% (4/4)
- **Exit criteria**: PASS (≥80% required)

### Per-Task Results

| Task | Description | grasp_kp | target_kp | Status |
|------|-------------|----------|-----------|--------|
| 4 | pick black bowl from drawer, place on plate | (30.5, 103.5) | (166.5, 205.5) | ✅ |
| 5 | pick black bowl on ramekin, place on plate | (312.5, 298.5) | (371.5, 310.5) | ✅ |
| 8 | pick black bowl next to plate, place on plate | (466.6, 303.4) | (400.1, 341.0) | ✅ |
| 9 | pick black bowl on cabinet, place on plate | (39.5, 150.5) | (0.5, 112.5) | ✅ |

### MOKA Output Schema (Validated)
```python
@dataclass
class MOKAOutput:
    grasp_kp: Tuple[float, float]     # (u, v) in pixels
    function_kp: Tuple[float, float]  # contact point on held object
    target_kp: Tuple[float, float]    # placement target
    pre_tile: str                     # e.g., "d3"
    target_tile: str
    post_tile: str
    pre_height: str                   # "same" or "above"
    post_height: str
    target_angle: str                 # e.g., "downward"
```

### Observations
1. **GroundingDINO** reliably detects "black bowl" and "plate" (logits 0.38-0.77)
2. **SAM** produces good segmentation masks
3. **Qwen2.5-VL-7B** successfully selects appropriate keypoints from visual marks
4. All outputs pass schema validation

### Files
- Validation script: `/workspace/new_experiment/phase0_moka_validation.py`
- Results: `/workspace/new_experiment/phase0_results/`
- Per-task visualizations in `task_*/` subdirectories

---

## Phase 1: Logging & Replay Infrastructure — COMPLETE ✅

### Components Implemented

1. **HDF5 Episode Logger** (`brain_robot/logging/hdf5_episode_logger.py`)
   - Efficient storage for training data
   - Chunked, compressed image datasets
   - MOKAOutput schema with 16D array encoding
   - Full proprioception: ee_pos, ee_quat, gripper_state, joint_pos
   - Actions (10D: 3 pos + 6D rot + gripper)
   - Skill/phase IDs and timestamps

2. **MOKAOutput Schema**
   ```python
   @dataclass
   class MOKAOutput:
       grasp_kp: Tuple[float, float]     # normalized (0-1)
       function_kp: Tuple[float, float]
       target_kp: Tuple[float, float]
       pre_tile, target_tile, post_tile: str  # e.g., "b2"
       pre_height, post_height: str      # "same" or "above"
       target_angle: str                 # "downward", "forward", etc.
       conf_grasp, conf_target: float    # confidence scores
   ```

3. **Replay Viewer** (`brain_robot/logging/replay_viewer.py`)
   - Interactive matplotlib GUI
   - Frame-by-frame scrubbing
   - MOKA keypoint overlays (G=grasp, F=function, T=target)
   - Skill/phase timeline bar
   - Proprioception info panel
   - Export to video

### Usage

```python
# Recording
from brain_robot.logging.hdf5_episode_logger import HDF5EpisodeLogger, Timestep, MOKAOutput

logger = HDF5EpisodeLogger(log_dir='./episodes')
with logger.start_episode(task='pick_bowl', skill_id=0) as ep:
    for obs, action in rollout:
        timestep = Timestep(
            rgb=obs['rgb'],
            depth=obs['depth'],
            ee_pos=obs['ee_pos'],
            moka_output=MOKAOutput(grasp_kp=..., target_kp=...),
            action=action,
            skill_id=0,
            phase_id=1,
        )
        ep.add_timestep(timestep)
    ep.mark_success(True)

# Replay
python -m brain_robot.logging.replay_viewer /path/to/episode.hdf5
```

### Tests
- All tests pass: `python test_phase1_standalone.py`
- MOKAOutput serialization roundtrip
- HDF5 read/write with all fields
- Empty episode handling

### Files
- HDF5 logger: `brain_robot/logging/hdf5_episode_logger.py`
- Replay viewer: `brain_robot/logging/replay_viewer.py`
- Tests: `brain_robot/test_phase1_standalone.py`

---

## Implementation Plan

### Phase 0: Validate MOKA First (Before Building Anything) — COMPLETE
Why: The entire system assumes MOKA gives good keypoints. If it doesn't, you're building on sand.
Day 1-2: Run MOKA on LIBERO tasks 4, 5, 8, 9
- Manually inspect: Are grasp_kp, target_kp reasonable?
- Measure: What % of keypoints are usable?
- Document failure modes

Day 3: Run MOKA on drawer/knob tasks (if you have images)
- Same analysis
- Identify if MOKA needs prompt tuning for your specific objects
Exit criteria: MOKA produces usable keypoints >80% of the time on your task distribution. If not, fix MOKA first or choose different tasks.
Phase 1: Logging & Replay Infrastructure
Why: You cannot debug what you cannot see. This is non-negotiable.
Week 1:
├── Episode logger (HDF5 or zarr)
│   ├── RGB images (256x256)
│   ├── Depth images
│   ├── Proprio: ee_pos, ee_quat, gripper_state, joint_pos
│   ├── Forces (if available)
│   ├── MOKAOutput (validated)
│   ├── Actions (from expert or policy)
│   ├── Skill ID, Phase ID
│   └── Timestamps
│
└── Replay viewer
    ├── Scrub through episode
    ├── Overlay keypoints on image
    ├── Show 3D subgoals projected to 2D
    └── Phase timeline bar
Exit criteria: Can record an episode, replay it, see all keypoints and phases visually.
Phase 2: Scripted Expert + Data Collection (Sim Only)
Why: You need data before you can train. Scripted expert in sim is cheap and controllable.
Week 2-3:
├── Scripted controllers for each skill
│   ├── ApproachSkill: Move to pre-grasp pose (use ground truth)
│   ├── GraspSkill: Descend, close gripper
│   ├── MoveSkill: Interpolate to target
│   ├── PlaceSkill: Descend, open gripper
│   ├── OpenDrawerSkill: Grasp handle, pull back
│   ├── CloseDrawerSkill: Push forward
│   └── StoveSkills: Grasp knob, rotate
│
├── Data collection
│   ├── 500-1000 episodes per skill
│   ├── Randomize object positions, orientations
│   ├── Include some "recovery" trajectories (start from slightly wrong pose)
│   └── Record MOKA output at each step (or every N steps)
│
└── Data validation
    ├── Visualize 10-20 episodes per skill
    ├── Check action distributions (no weird outliers)
    └── Verify MOKA keypoints align with expert behavior
Exit criteria: Have clean dataset with ~5000 total episodes across skills. Can visualize any episode and it looks reasonable.
Phase 3: Tracking + 2D→3D Grounding
Why: Policy needs 3D subgoals, not 2D keypoints. This is the bridge.
Week 4:
├── Correlation tracker
│   ├── NCC template matching (32x32 patch, 64x64 search)
│   ├── Track grasp_kp, target_kp across frames
│   ├── Log correlation scores
│   └── Detect tracking failure (score < 0.5)
│
├── 2D→3D lift
│   ├── Sample depth in 5x5 patch around keypoint
│   ├── Reject invalid depths
│   ├── Compute mean + std as uncertainty
│   └── Project to 3D using camera intrinsics
│
└── Surface normal estimation
    ├── Table plane: hardcoded or single RANSAC at start
    ├── Object surfaces: local PCA on depth patch
    └── Fallback: assume vertical normal
Exit criteria: Can run tracker on recorded episodes, 3D subgoals are within 2cm of ground truth 80% of time.
Phase 4: Train Pick-and-Place Policy First
Why: ApproachSkill + GraspSkill + MoveSkill + PlaceSkill are the core. Get these working before contact skills.
Week 5-6:
├── Policy architecture
│   ├── ResNet18 encoder (frozen)
│   ├── Subgoal encoder: MLP(3D pos + uncertainty) → 64-dim
│   ├── Proprio encoder: MLP(8) → 32-dim
│   ├── Skill/phase embedding: learned, 32-dim
│   ├── Transformer: 4 layers, d=256, 4 heads, L=12
│   └── Action head: MLP → 7-dim (dx, dy, dz, drot, gripper)
│
├── Training
│   ├── BC loss: MSE on actions
│   ├── Batch size: 64
│   ├── LR: 1e-4 with cosine decay
│   ├── Train for 50-100 epochs
│   └── Early stopping on val loss
│
└── Evaluation
    ├── Run in sim with MOKA (closed-loop)
    ├── Measure success rate per skill
    ├── Log failure modes
    └── Compare to scripted expert baseline
Exit criteria: Pick-and-place success rate >70% in sim. Can identify specific failure modes.
Phase 5: Add Contact Skills (Drawer, Knob)
Why: These are harder due to force requirements and progress estimation.
Week 7-8:
├── Drawer skills
│   ├── Progress estimator: track handle displacement in image
│   ├── Force threshold for contact detection
│   ├── Termination: d_open > threshold OR force spike
│   └── Train on drawer demos
│
├── Knob skills
│   ├── Angle estimator: track 2-3 points on knob edge
│   ├── (Skip optical flow for now - too complex)
│   ├── Termination: angle > target OR timeout
│   └── Train on knob demos
│
└── Unified training
    ├── Mix all skill data
    ├── Skill embedding selects behavior
    └── Evaluate each skill separately
Exit criteria: Drawer open >60%, knob turn >50% in sim.
Phase 6: DAgger + Iteration
Why: BC alone will have compounding errors. Need to collect failure data and retrain.
Week 9+:
├── Run policy in sim
│   ├── Identify failure episodes
│   ├── Cluster failure modes (tracking loss, wrong approach, etc.)
│   └── Generate "hard" initial conditions
│
├── Expert relabeling
│   ├── For each failure state, run scripted expert
│   ├── Add corrective trajectories to dataset
│   └── Weight recent failures higher in training
│
└── Iterate
    ├── Retrain with augmented dataset
    ├── Re-evaluate
    ├── Repeat until plateau
    └── Then consider sim-to-real
Summary: What's Different from the Spec
Spec Says	I Recommend
Week 1: Logging	Phase 0: Validate MOKA first
Week 2: MOKA Evaluation	Move to Phase 0, before any infrastructure
Week 4: Policy training	Need data first - scripted expert comes before policy
Optical flow for knob	Skip for v2.0 - just track edge points
L=12 sequence length	Start with L=16 for contact skills
The Critical Question
Before implementing anything: Have you validated MOKA on your actual task images? If not, that's the first thing to do. Want me to run MOKA on the LIBERO task images right now and show you the outputs?
what do you think:

3. Where I Slightly Disagree / Suggest Refinements

These are minor but important.

(A) Phase 3: Surface Normal via Local PCA — I’d Still Adjust

You currently propose:

Object surfaces: local PCA on depth patch
Fallback: assume vertical normal

This is still fragile near:

Occlusions

Edges

Thin handles

I strongly recommend one extra check, even in v2.0:

If PCA eigenvalue ratios do not show a clear planar structure
(e.g., λ₁ ≫ λ₂ ≫ λ₃ not satisfied):
→ Treat as unreliable normal, don’t trust it.

This avoids injecting garbage orientation signals into the policy.

(B) Phase 4: Policy Output Dimensionality Is Slightly Underspecified

You currently say:

Action head → 7-dim (dx, dy, dz, drot, gripper)

You should make this explicit:

Either:

3 pos + 3 rot (6D) + gripper = 7D

Or:

3 pos + 6D rotation representation + gripper = 10D

You earlier argued strongly against ambiguous rotation parameterization — so you should lock this now to avoid inconsistent training targets.

(C) Phase 5: Knob — Skipping Optical Flow Is Fine, But Track Geometry

You changed to:

Track 2–3 points on knob edge

Good simplification. But I recommend:

Track at least 3 non-collinear points on the circumference.

Estimate angle by:

Fitting circle center once.

Computing angular displacement of each tracked point.

Taking median Δθ.

This gives you:

Built-in outlier rejection

No need for flow

Much more stable than single-edge tracking

(D) Phase 4 Exit Criteria: 70% May Be Too Low for Policy Stability

You say:

Exit criteria: Pick-and-place success rate >70% in sim

70% is OK as a debug gate, but not as a control baseline. At 70%:

DAgger will be flooded with low-quality recovery data.

You’ll spend weeks stabilizing instead of refining.

I建议 splitting the gate:

Proceed to Phase 5 at 70% (for experimentation).

Require ≥85% before freezing architecture and moving toward real-world tests.

4. Minor Adjustments to Your Phase List (Still Your Plan)

Here is your exact plan with only minimal strengthening (no structural changes):

✅ Phase 0: Validate MOKA First

No changes. This is perfect.

Only add:

Log pixel error distribution vs GT.

Log failure mode taxonomy (occlusion, wrong object, wrong affordance).

✅ Phase 1: Logging & Replay

No changes.

Only add:

Phase duration histograms.

Correlation scores (when you add tracking later).

✅ Phase 2: Scripted Expert + Data

No changes.

Add:

At least 10–15% recovery-initialized episodes per skill (slightly wrong initial pose).

This trains the policy’s error dynamics early.

✅ Phase 3: Tracking + Geometry

Refine:

PCA normal only if eigenvalue condition satisfied.

Record:

Subgoal bias (mean error)

Subgoal variance

Outlier rate

Exit criterion strengthened to:

80% within 2 cm AND <10% catastrophic outliers (>5 cm)

✅ Phase 4: Pick-and-Place Policy

Refine:

Clarify rotation representation (6D or quat).

Track:

Action magnitude histogram

Attention heatmaps

Subgoal-policy error coupling

Exit gate split:

≥70% → proceed to Phase 5

≥85% → architecture considered stable

✅ Phase 5: Contact Skills

Refine knob:

Track ≥3 edge points

Median angular displacement

No optical flow

✅ Phase 6: DAgger

No changes. This is exactly right.
Good feedback. Let me address each point:
(A) PCA Normal Sanity Check — Agree
You're right. Adding eigenvalue ratio check is cheap and prevents garbage:
def estimate_surface_normal(depth_patch, min_planarity=0.8):
    """Estimate normal only if patch is clearly planar."""
    points_3d = deproject_patch(depth_patch)
    
    # PCA
    cov = np.cov(points_3d.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    λ1, λ2, λ3 = sorted(eigenvalues, reverse=True)
    
    # Planarity check: λ3 should be much smaller than λ1, λ2
    planarity = 1.0 - (λ3 / (λ1 + 1e-8))
    
    if planarity < min_planarity:
        return None, False  # Unreliable, use fallback
    
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal, True
Policy sees normal_reliable: bool flag. If False, policy learns to be more cautious or use prior (vertical).
(B) Rotation Representation — Agree, Lock It Now
You're right, I was vague. Here's the decision: Use 6D rotation representation (Zhou et al., "On the Continuity of Rotation Representations"):
Action = [dx, dy, dz, r1, r2, r3, r4, r5, r6, gripper]
         └─ 3D pos ─┘ └────── 6D rotation ──────┘  └─ 1D
         
Total: 10-dim action
Why 6D over quaternion:
Continuous (no discontinuity at antipodal points)
No normalization constraint during training
Easier for network to learn
Why 6D over axis-angle:
Axis-angle has singularity at zero rotation
6D is just first two columns of rotation matrix, trivially recoverable
(C) Knob Tracking — Agree, 3-Point Circle Fit
Your suggestion is better than my "2-3 edge points":
def estimate_knob_angle(tracked_points, center):
    """
    tracked_points: list of (u, v) for 3+ points on knob edge
    center: (u_c, v_c) knob center (from MOKA or fixed)
    """
    angles = []
    for (u, v) in tracked_points:
        θ = np.arctan2(v - center[1], u - center[0])
        angles.append(θ)
    
    # Compute angular displacement from initial
    Δθ = [angle - initial_angle for angle, initial_angle 
          in zip(angles, self.initial_angles)]
    
    # Median for robustness
    return np.median(Δθ)
Key points:
Track ≥3 non-collinear points on circumference
Fit circle center once at skill start (or use MOKA grasp_kp as center)
Median Δθ rejects outliers from occlusion
(D) Exit Criteria Split — Agree
Your split makes sense:
Gate	Threshold	What It Means
Proceed Gate	≥70%	Good enough to start Phase 5, continue iterating
Stability Gate	≥85%	Architecture is sound, ready for real-world
Deployment Gate	≥95%	Ready for actual use (Phase 6+)
This prevents:
Premature architecture changes when at 70%
DAgger drowning in garbage data
False confidence from lucky runs
Updated Phase Summary
Here's the refined plan with your additions:
Phase 0: MOKA Validation
├── Run on task images
├── Log pixel error vs GT
├── Taxonomy: {occlusion, wrong_object, wrong_affordance, good}
└── Exit: >80% usable keypoints

Phase 1: Logging & Replay
├── Episode logger (HDF5)
├── Replay viewer with overlays
├── Phase duration histograms
└── Exit: Can record + replay + visualize

Phase 2: Scripted Expert + Data
├── 500-1000 episodes per skill
├── 10-15% recovery-initialized episodes
├── Randomize positions/orientations
└── Exit: Clean dataset, visualized samples look good

Phase 3: Tracking + Geometry
├── NCC tracker (32x32 patch)
├── 2D→3D lift with uncertainty
├── PCA normal WITH eigenvalue check
├── Log: bias, variance, outlier rate
└── Exit: 80% within 2cm AND <10% outliers >5cm

Phase 4: Pick-and-Place Policy
├── ResNet18 (frozen) + Transformer
├── 10D action: 3 pos + 6D rot + gripper
├── Log: action histograms, attention maps
├── Proceed gate: ≥70%
└── Stability gate: ≥85%

Phase 5: Contact Skills
├── Drawer: track handle displacement
├── Knob: 3+ edge points, median Δθ
├── Force thresholds for contact
└── Exit: Drawer >60%, Knob >50%

Phase 6: DAgger
├── Failure clustering
├── Expert relabeling
├── Weighted retraining
└── Iterate until plateau
One Question Back to You
On recovery-initialized episodes (10-15%): How wrong should "slightly wrong initial pose" be? My suggestion:
Position: ±2-3cm from ideal
Orientation: ±10-15° from ideal
Not so wrong that scripted expert can't recover
This teaches the policy "how to correct" without being impossible demonstrations. Does that match your intent?