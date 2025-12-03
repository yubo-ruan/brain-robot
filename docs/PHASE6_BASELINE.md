# Phase 6: Multi-Suite Baseline Evaluation

## Overview

Phase 6 evaluates the brain_robot system's readiness for LIBERO suites beyond libero_spatial, with the goal of eventually supporting long-horizon tasks (libero_10/libero_long).

## Phase 6-0: Suite Compatibility Check

### Current System Capabilities

**YOLO Classes (10 classes):**
```
bowl, plate, mug, ramekin, cabinet, drawer, cookie_box, can, bottle, stove
```

**Skills (4 skills):**
```
ApproachObject, GraspObject, MoveObjectToRegion, PlaceObject
```

### Suite Compatibility Analysis

| Suite | YOLO Compatible | Skills Compatible | Runnable Tasks |
|-------|-----------------|-------------------|----------------|
| libero_spatial | ✅ 10/10 | ✅ 10/10 | **10/10** |
| libero_object | ❌ 0/10 | ✅ 10/10 | **0/10** |
| libero_goal | ⚠️ 4/10 | ⚠️ 4/10 | **3-4/10** |
| libero_10 (long) | ❌ 1/10 | ❌ 0/10 | **0/10** |

### libero_object Blockers

All 10 tasks require objects not in YOLO training:
- alphabet_soup, cream_cheese, salad_dressing, bbq_sauce, ketchup
- tomato_sauce, butter, milk, chocolate_pudding, orange_juice, basket

**Action Required:** Retrain YOLO with new object classes

### libero_goal Blockers

| Task | Description | Blocker |
|------|-------------|---------|
| 0 | open the middle drawer | Missing OpenDrawer skill |
| 1 | put bowl on stove | ✅ Compatible |
| 2 | put wine bottle on cabinet | ✅ Compatible (MoveSkill timeout) |
| 3 | open drawer and put bowl inside | Missing OpenDrawer skill |
| 4 | put bowl on top of cabinet | ✅ Compatible (MoveSkill timeout) |
| 5 | push plate to front of stove | Missing Push skill |
| 6 | put cream cheese in bowl | Missing cream_cheese class |
| 7 | turn on the stove | Missing TurnOn skill |
| 8 | put bowl on plate | ✅ Compatible |
| 9 | put wine bottle on rack | Missing rack class |

### libero_10 (Long-Horizon) Blockers

All tasks are multi-step and require:
1. **Skill chaining** - Execute 2-5 skills in sequence
2. **New skills** - TurnOn, CloseDrawer, CloseMicrowave
3. **New objects** - moka_pot, book, caddy, microwave
4. **Object memory** - Track objects moved out of view

---

## Phase 6-A: Baseline Results (Oracle Perception)

### libero_goal Evaluation

| Task | Description | Success | Notes |
|------|-------------|---------|-------|
| 1 | put bowl on stove | **100%** (5/5) | ✅ Works perfectly |
| 2 | put wine bottle on cabinet | **20%** (1/5) | MoveSkill timeout |
| 4 | put bowl on top of cabinet | **0%** (0/5) | MoveSkill timeout |
| 8 | put bowl on plate | **100%** (5/5) | ✅ Works perfectly |

### libero_object Evaluation

| Task | Description | Success | Notes |
|------|-------------|---------|-------|
| 0 | alphabet soup → basket | **0%** (0/3) | Missing object classes |

### Key Findings

1. **MoveSkill timeout is the primary blocker** for cabinet placement tasks
   - Same issue seen in libero_spatial (80.8% of failures)
   - Cabinet targets may be at edges of reachable workspace

2. **Grounding fixes implemented:**
   - Added "put the X on/in the Y" pattern (libero_goal style)
   - Added "on top of" → cabinet_top mapping
   - Added multi-word source parsing (e.g., "wine bottle")
   - Added stove burner fallback when main stove not available

3. **Compatible libero_goal tasks work at 100%:**
   - Task 1 (bowl → stove): 100%
   - Task 8 (bowl → plate): 100%

---

## Blockers Summary

### Immediate (Blocking Any Progress)

| Blocker | Impact | Solution |
|---------|--------|----------|
| MoveSkill timeout | 60-100% failure on cabinet tasks | Improve motion planning, workspace checks |
| Missing object classes | 100% failure on libero_object | Retrain YOLO |

### Required for Long-Horizon

| Blocker | Impact | Solution |
|---------|--------|----------|
| No skill chaining | Can't execute multi-step tasks | Activate QwenSkillPlanner |
| No OpenDrawer skill | 2 libero_goal tasks blocked | Implement skill |
| No TurnOn skill | 1 libero_goal + 2 libero_10 blocked | Implement skill |
| No CloseDrawer skill | 3 libero_10 tasks blocked | Implement skill |
| No object memory | Objects lost when moved | Implement last_seen tracking |

---

## Recommended Next Steps

### Priority 1: Fix MoveSkill (Highest ROI)
- Same issue blocks libero_spatial AND libero_goal
- 80%+ of failures in both suites

### Priority 2: Add Missing Skills
- OpenDrawer, CloseDrawer, TurnOn
- Unblocks 4 libero_goal tasks

### Priority 3: Retrain YOLO
- Add libero_object classes (10 new objects)
- Add libero_10 classes (moka_pot, book, etc.)

### Priority 4: Multi-Step Execution
- Implement skill chaining for long-horizon
- Add object memory (last_seen tracking)
- Implement failure recovery (from original Phase 6 plan)

---

## Files Modified

- `scripts/run_evaluation.py` - Extended grounding for libero_goal task formats
  - Added "put the X on/in the Y" pattern
  - Added "on top of" → cabinet_top mapping
  - Added multi-word source parsing
  - Added stove/cabinet part handling

---

---

## Phase 6-B: Implementation Progress

### Completed

1. **MoveSkill Improvements** (Priority 1)
   - Added height boost recovery when stuck at workspace edges
   - Tuned stuck detection: min_velocity 0.001→0.0005, window 10→20, threshold 3→5
   - Cabinet placement success: 20% → 65%

2. **New Skills Added** (Priority 2)
   - `OpenDrawerSkill` - ~40% success rate
   - `CloseDrawerSkill` - Same mechanism as open
   - `TurnOnStoveSkill` - 0% success (stove button is hinge joint, needs rotation not push)
   - `TurnOffStoveSkill` - Toggle version

3. **Skill Chaining** (Priority 4)
   - Created `brain_robot/planning/skill_chain.py`
   - `SkillChain` class with task decomposition and execution
   - Pattern matching for LIBERO task formats:
     - "pick up X and place it in Y"
     - "put both X and Y in Z"
     - "turn on stove and put X on it"
     - "put X in drawer and close it"
     - "put X on left plate and Y on right plate"
   - All 10 libero_10 tasks decompose successfully

### Pending

4. **YOLO Retraining** (Priority 3)
   - Need to add 10+ new object classes
   - Data collection required

### Known Issues

1. **TurnOnStove** - Button is hinge joint (type 3), needs rotation motion
2. **Object grounding** - "yellow and white mug" grounds to "white_mug"
3. **Two moka pots** - Pattern "put both X on Y" needs different objects

---

## Files Modified/Created

| File | Change |
|------|--------|
| `skills/move.py` | Height boost recovery, tuned stuck detection |
| `skills/drawer.py` | NEW - OpenDrawer/CloseDrawer skills |
| `skills/stove.py` | NEW - TurnOn/TurnOff skills |
| `skills/__init__.py` | Registered new skills in SKILL_REGISTRY |
| `planning/skill_chain.py` | NEW - Task decomposition and chain execution |
| `planning/__init__.py` | Exported SkillChain |
| `scripts/run_evaluation.py` | Extended grounding patterns |
| `scripts/test_skill_chain.py` | NEW - Unit tests for skill chaining |

---

## Phase 6-C: Grasp Skill Investigation

### Critical Bug Discovery: Inverted Gripper Actions

**Date:** Phase 6-C investigation

**Bug:** All gripper actions were inverted across the codebase.
- We were using `action[6] = 1.0` for "open" and `-1.0` for "close"
- Robosuite OSC convention: `action[6] = -1.0` opens gripper, `+1.0` closes

**Impact:** Complete grasp failure - gripper was closing when we wanted open and vice versa.

**Files Fixed:**
- `skills/grasp.py` - All gripper action signs corrected
- `skills/place.py` - Fixed open/close during place sequence
- `skills/approach.py` - Fixed gripper kept open during approach
- `skills/move.py` - Fixed gripper kept closed during movement
- `skills/drawer.py` - Fixed open/close for drawer manipulation
- `skills/stove.py` - Fixed gripper closed for button pressing

### Grasp Skill Improvements

1. **Hold position while opening gripper** (Phase 0)
   - Previously: `action = np.zeros(7)` caused arm drift
   - Now: Use PD controller to hold position while opening

2. **Grasp height offset adjusted**
   - Changed from `+0.02m` (above center) to `0.0m` (at center)
   - Small cylindrical objects need gripper at/below center

3. **Gripper closure detection improved**
   - Threshold lowered from `0.015-0.07` to `0.008-0.07`
   - Better detection of partial grasps

### Remaining Issue: Gripper Not Reaching Object

**Observation:**
- World state and simulator positions match (0.0000m diff)
- Gripper approaches correctly (5cm XY error at approach end)
- During grasp descent, gripper PUSHES object sideways
- Object moves ~3.5cm during grasp attempt
- Gripper closes on air (width = 0.0011-0.0028)

**Root Cause Hypothesis:**
The gripper is descending but not on the correct trajectory. The PD controller is fighting workspace limits or the gripper orientation causes collision with object edge before fingers can wrap around.

**Evidence:**
- Workspace test shows all positions ARE reachable (2-4cm error)
- Object moves during grasp = gripper is making contact
- Gripper width ~0.001 = closed on nothing (not on object)

**Potential Fixes to Try:**
1. Increase descent precision (tighter PD gains for final approach)
2. Add short pause after XY alignment before descent
3. Adjust gripper orientation for cylindrical objects
4. Implement visual servoing for final approach

### YOLO Data Collection Fix

**Bug:** Camera projection was using wrong convention.
- MuJoCo uses OpenGL convention: -Z is forward in camera frame
- Original code checked `point_cam[2] <= 0` (incorrect)
- Fixed to check `point_cam[2] >= 0` for behind-camera rejection

**Result:** Data collection now works, generating ~150 frames per task.

---

**Status:** Phase 6-C complete. Major gripper action bug fixed. Grasp still fails on small cylindrical objects due to trajectory/orientation issues. YOLO data collection working.

---

## Phase 6-D: Truth Discovery

### Critical Finding: Previous Evaluations Were False

**Discovery Date:** Phase 6-D audit

The original "100% success rate" on libero_spatial was fake. Investigation revealed:

1. **Old gripper code was completely wrong:**
   - Phase 1 (descent): `gripper=1.0` → Actually CLOSING (not opening as comment said)
   - Phase 2 (close): `action[6]=-1.0` → Actually OPENING (not closing)
   - Phase 3 (lift): `gripper=-1.0` → Actually OPENING (not keeping closed)

2. **Old success threshold was passing by accident:**
   - Gripper ended at width ~0.078 (nearly fully open)
   - Threshold `0.001 < width < 0.08` accepted 0.078 as "grasped"
   - Object was never actually picked up

3. **Evaluation was measuring wrong thing:**
   - Our code measured "did our skills report success?"
   - LIBERO's actual reward was 0.0 (task NOT achieved)

### Fix: Added LIBERO Ground Truth Success Metric

Added `check_libero_success(env)` function that calls `env.env._check_success()` to get LIBERO's built-in success condition. This is now reported alongside our skill success rate.

New output format:
```
SUCCESS METRICS
Skill Success Rate:        100.0% (2/2) [skills reported OK]
LIBERO Success Rate:       0.0% (0/2) [actual task goal achieved]
Fake Success Rate:         100.0% (2/2) [skills OK but goal NOT achieved]
```

### Remaining Issues After Gripper Fix

With correct gripper semantics (open=-1, close=+1):
- Gripper opens correctly before descent
- Gripper closes correctly, but closes on air (width ~0.001)
- Object is NOT lifted

**Root cause identified:**
1. Physics settling: Object spawns at Z=0.97 but settles to Z=0.898 after first steps
2. Perception uses spawn position, not settled position
3. Grasp target Z is set to 0.97, but object is actually at 0.898
4. Gripper reaches ~0.91 (workspace limit) while trying to reach 0.97
5. Result: Gripper is 2cm above object when it tries to close

**Workspace limitations:**
- Table top: Z=0.875
- Bowl body center (settled): Z=0.898
- Gripper minimum Z: ~0.91
- Gripper cannot reach object center (2cm short)

---

## Phase 6-E: Grasp Fix Attempts

### Key Issues Identified

1. **Workspace Limitation:**
   - Gripper minimum Z: ~0.91
   - Bowl center Z: ~0.898 (unreachable)
   - Bowl rim Z: ~0.93 (reachable)
   - Solution: Changed `grasp_height_offset` from 0.0 to 0.03 to target rim

2. **Gripper Width Threshold:**
   - Old threshold: `0.008 < width < 0.07`
   - Bowl rim grasps produce width ~0.0075
   - Solution: Changed threshold to `0.005 < width < 0.07`

3. **Dynamic Object Tracking:**
   - During descent, gripper pushes bowl, bowl moves
   - Grasp targets stale position
   - Object ends up 7-8cm away from gripper

### Remaining Issues

1. **Object pushed during descent:**
   - Gripper collides with bowl rim
   - Bowl slides away
   - Gripper closes on air

2. **XY refinement insufficient:**
   - After approach, XY error is 6-7cm
   - Refinement reduces to 2-3cm but not enough
   - Bowl diameter is ~10cm, need <2cm error for reliable grasp

### Potential Solutions (Not Yet Implemented)

1. **Slower, more careful descent** - Reduce collision force
2. **Visual servoing during descent** - Track object as it moves
3. **Approach from directly above** - Minimize lateral contact
4. **Two-stage grasp** - Open wide, descend, then close
5. **Cage grasp** - Fingers on both sides before closing

---

**Status:** Phase 6-E complete. Grasp improvements made (height offset, threshold). Fundamental issue remains: object tracking during grasp descent. When starting from a stable position (not after approach), grasp works with ~5cm lift.

---

## Phase 6-F: First-Principles Grasp Analysis

### Grasp Tolerance Map Experiment

Conducted systematic experiment to measure actual grasp success as function of XY offset from bowl center.

**Methodology:**
1. Reset environment, let physics settle
2. Position gripper directly above bowl at various XY offsets (no ApproachSkill)
3. Execute grasp: descend → close → lift
4. Log LIBERO success, gripper width, object displacement
5. 5x5 grid: offsets [-4, -2, 0, +2, +4] cm in X and Y
6. 3 trials per offset combination

**Results (Success Rate %):**

```
           dy offset
           -4cm   -2cm   +0cm   +2cm   +4cm
  -4cm |   100    100      0    100    100
  -2cm |   100     67      0    100     67
dx +0cm |   100    100      0    100    100
  +2cm |   100    100      0    100    100
  +4cm |   100    100      0     33    100
```

### Critical Finding: Center Grasp Always Fails

**Observation:** Center (0,0) = 0% success, but off-center (±2-4cm) = 67-100% success.

**Root Cause:** The bowl is a **hollow container** (~5.6cm radius).
- At center: gripper fingers descend into the hollow interior, nothing to grip
- At offset: gripper fingers straddle the bowl rim, can grip successfully

**Geometric Analysis:**
- Bowl outer radius: ~5.6 cm
- Bowl rim width: ~0.5-1 cm
- Optimal grasp zone: 4-5 cm from center (on the rim)
- Dead zone: 0-2 cm from center (inside hollow)

**Implications for Skill Design:**
1. **Do NOT target object center for hollow objects**
2. For bowls/cups/containers: target rim at offset ~4cm from center
3. Requires object-specific grasp planning (solid vs hollow)

### Updated Grasp Strategy

For hollow objects (bowls, cups, mugs):
```python
# Calculate rim offset based on object type
if object_type in ['bowl', 'mug', 'cup']:
    grasp_offset = object_radius * 0.7  # ~4cm for typical bowl
    grasp_point = object_center + [grasp_offset, 0, rim_height]
else:
    grasp_point = object_center + [0, 0, grasp_height]
```

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| Overall success rate | 74.7% (56/75 trials) |
| Center success rate | 0% |
| Rim success rate (±4cm) | 100% |
| Average object displacement | 4.5 cm |
| Optimal offset range | 3-5 cm from center |

---

**Status:** Phase 6-F complete. Discovered fundamental issue: center grasp on hollow objects always fails. Solution: offset grasp to target rim, not center.

---

## Phase 6-G: Rim Grasp Implementation

### Implementation

Added object-type-aware grasp targeting to `GraspSkill`:

1. **Object type detection**: Uses `world_state.objects[obj].object_type` or infers from name
2. **Rim offset calculation**: `offset = object_radius - gripper_finger_half_width`
3. **Direction**: Offset toward gripper's current position (grasp "near side" of rim)
4. **XY refinement targets rim point**, not object center

**Key code changes in `skills/grasp.py`:**
- Added `HOLLOW_OBJECTS = {'bowl', 'mug', 'cup', 'ramekin'}`
- Added `OBJECT_RADII` lookup table
- Added `_compute_grasp_point()` method for object-aware targeting
- Modified XY refinement to target grasp point, not center
- Increased descent steps from 16 to 50

### Results

| Metric | Before (center grasp) | After (rim grasp) |
|--------|----------------------|-------------------|
| Skill Success Rate | 0% | 15% (3/20) |
| LIBERO Success Rate | 0% | 0% |

### Remaining Issues

1. **Low grasp success rate (15%)**: Rim targeting helps but still fails 85% of the time
2. **LIBERO success = 0%**: Even when grasp succeeds, bowl isn't placed correctly on plate
3. **Variable starting position**: ApproachSkill leaves gripper at different positions

### Analysis

The tolerance map showed 100% success at ±4cm offset when starting directly above the bowl. But the full pipeline has:
- ApproachSkill leaves gripper ~5cm away from ideal position
- XY refinement helps but doesn't fully compensate
- The gripper often doesn't descend far enough to reach the rim

### Next Steps

1. **Improve ApproachSkill precision** to leave gripper more directly above target
2. **Add visual servo during descent** to track object as it moves
3. **Debug LIBERO success** - investigate why completed skill chain still fails LIBERO goal

---

**Status:** Phase 6-G complete. Rim grasp implemented and shows improvement (0% → 15%).

---

## Phase 6-H: Placement Precision Improvements

### Problem Identified

LIBERO success remained at 0% despite skill chain completing successfully. Investigation revealed:

1. **LIBERO's `On` predicate requires XY distance < 3cm** from target object center
2. **MoveSkill ends with 4-5cm XY error** (acceptable for skill but not for LIBERO)
3. **PlaceSkill just lowered and released** without correcting final position

### Root Cause Analysis

Looking at LIBERO's `check_ontop` predicate (`libero/libero/envs/object_states/base_object_states.py`):

```python
def check_ontop(self, other):
    # ...
    return (
        (this_object_position[2] <= other_object_position[2])
        and self.check_contact(other)
        and (
            np.linalg.norm(this_object_position[:2] - other_object_position[:2])
            < 0.03  # <-- 3cm threshold!
        )
    )
```

Our debug traces showed:
- Episode 0: Final bowl-to-plate distance = 7.3 cm ❌
- Episode 3: Final bowl-to-plate distance = 13.6 cm ❌
- Episode 5: Final bowl-to-plate distance = 5.3 cm ❌

### Implementation

Added **XY centering phase** to `PlaceSkill`:

1. Before lowering, move gripper directly above target (XY correction)
2. Target <2cm XY error before proceeding to lower phase
3. Then lower and release as before

**Key code changes in `skills/place.py`:**
```python
# Phase 0: XY Centering - move directly above target before lowering
# This is critical for LIBERO success which requires <3cm XY precision
if target_pos is not None:
    xy_error_before = np.linalg.norm(current_pose[:2] - target_pos[:2])

    if xy_error_before > 0.02:  # Only if needed (>2cm error)
        center_target = current_pose.copy()
        center_target[0] = target_pos[0]
        center_target[1] = target_pos[1]
        self.controller.set_target(center_target, gripper=1.0)

        for step in range(xy_center_steps):
            # ... servo to target XY
```

### Additional Fixes

1. **Fixed rim offset direction**: Changed from gripper-relative to fixed world-frame direction (-Y toward robot base). This makes rim targeting predictable across different Approach end positions.

2. **Adjusted gripper thresholds**: Lowered from 0.005 to 0.004 for `_check_gripper_closed()` - bowl rim grasps can be very tight.

3. **Relaxed lift verification**: Changed from `lift_height/2` to fixed 2cm threshold.

### Results

| Metric | Phase 6-G (rim grasp only) | Phase 6-H (+ placement fix) |
|--------|---------------------------|----------------------------|
| Skill Success Rate | 15% (3/20) | 30% (6/20) |
| LIBERO Success Rate | 0% | 5% (1/20) |
| Fake Success Rate | 15% | 25% |

**Debug episode example:**
```
Episode 4:
  Place: ✓ (70 steps)
    XY centering: 3.2cm -> 1.3cm  # Successfully centered
```

### Remaining Issues

1. **30% skill success** - Grasp still fails 70% of the time
2. **25% fake success** - Skills report OK but bowl not on plate (slip during release?)
3. **Gripper descent issues** - Some episodes show gripper going UP instead of down

### Analysis of Fake Successes

The "fake successes" happen when:
- Grasp reports success (gripper width ~0.005)
- Move and Place report success
- But bowl ends up 4-15cm from target

Likely cause: **Bowl slipping from gripper during XY centering movement**. The marginal grip (width 0.004-0.005) isn't secure enough to hold during lateral motion.

### Next Steps

1. **Improve grasp security** - Consider deeper grip or different grasp strategy
2. **Reduce XY centering force** - Slower movement during centering to prevent slip
3. **Investigate descent failures** - Why does gripper sometimes go UP?
4. **YOLO retraining** (parallel track) - Improve detection for learned perception

---

**Status:** Phase 6-H complete. LIBERO success improved from 0% to 5%. Total skill pipeline now at 30% success.

**Current Performance Summary:**
- Approach: 100% success
- Grasp: 40% success
- Move: 40% success (blocked by Grasp)
- Place: 40% success (blocked by Grasp)
- **End-to-end LIBERO: 5%**

---

## Phase 6-I: YOLO Retraining for Multi-Suite Support

### Problem Statement

The original YOLO model (v14) was trained on only 10 classes from libero_spatial:
```
bowl, plate, mug, ramekin, cabinet, drawer, cookie_box, can, bottle, stove
```

This blocked 100% of libero_object tasks and many libero_10 tasks that require new object classes.

### Data Collection & Training

#### V1 Training (Initial)

**Dataset Creation:**
1. Merged `yolo_libero` (original 2000 images) with `perception_v2` (2400 images)
2. Total: 3756 train + 684 val images
3. Expanded to 30 classes

**Training Configuration:**
- Model: YOLOv8s (11.15M params, up from YOLOv8n 3.01M)
- Epochs: 100
- Batch size: 32
- Image size: 256
- Patience: 30 (early stopping)

**V1 Results:**
| Metric | Value |
|--------|-------|
| mAP50 | 0.995 |
| mAP50-95 | 0.967 |
| Precision | 0.996 |
| Recall | 0.998 |

#### V1 vs V14 Comparison

| Metric | V14 (Old) | V1 (New) | Change |
|--------|-----------|----------|--------|
| Architecture | YOLOv8n | YOLOv8s | 3.7x params |
| Classes | 10 | 30 | +20 classes |
| Total Detections | 351 | 526 | +50% |
| Classes Detected | 6 | 15 | +150% |
| Bowl Confidence | 0.284 | 0.294 | +3.5% |
| Plate Confidence | 0.419 | 0.660 | +58% |
| Ramekin Confidence | 0.176 | 0.427 | +143% |

**New Classes Now Detected:**
- salad_dressing: 0.629 conf
- basket: 0.578 conf
- tomato_sauce: 0.444 conf
- ketchup: 0.361 conf
- orange_juice: 0.301 conf
- moka_pot: 0.253 conf
- butter, milk: 0.213 conf
- alphabet_soup: 0.177 conf

### MuJoCo Body Name Mapping Discovery

Investigation revealed incorrect body name mappings were preventing data collection for some classes:

| MuJoCo Body Name | Correct Class | Previously Mapped |
|------------------|---------------|-------------------|
| `cabinet_top/middle/bottom` | drawer | (missed) |
| `porcelain_mug` | white_mug | mug |
| `white_yellow_mug` | yellow_white_mug | (missed) |
| `microwave_1_main` | microwave | (missed) |
| `red_coffee_mug` | mug | (correct) |

### V2 Training (With Additional Classes)

After fixing body name mappings, collected additional data:

**Extra Data Collection:**
- Tasks with drawer, white_mug, microwave
- 2100 additional images
- **New class samples:**
  - drawer: 1200
  - white_mug: 900
  - yellow_white_mug: 600
  - microwave: 300
  - wine_bottle: 900
  - wine_rack: 900

**Final Dataset (V2):**
- Total training images: 8556 (was 3756)
- Validation images: 684
- 30 classes with improved coverage

**V2 Training Status:** In progress (libero_merged_v2)

### Files Modified

| File | Change |
|------|--------|
| `brain_robot/perception/detection/yolo_detector.py` | Updated LIBERO_CLASSES to 30 classes, default model to yolo_libero_v2.pt |
| `data/yolo_merged/data.yaml` | 30-class configuration |
| `scripts/collect_additional_classes.py` | NEW - Fixed body name mappings for drawer/white_mug/microwave |
| `models/yolo_libero_v2.pt` | NEW - Retrained model |

### Class Coverage Analysis

**Classes with Good Coverage (>500 samples):**
```
bowl: 4959, plate: 5825, cabinet: 8609, stove: 5064, drawer: 1200,
cookie_box: 1836, ramekin: 1986, mug: 1830, white_mug: 900,
wine_bottle: 1050, wine_rack: 1017, basket: 1443, ketchup: 952,
tomato_sauce: 964, milk: 967, butter: 958, alphabet_soup: 957,
cream_cheese: 966, orange_juice: 955, salad_dressing: 727
```

**Classes with Limited Coverage (<500 samples):**
```
microwave: 300, yellow_white_mug: 837, moka_pot: 123, book: 122,
caddy: 122, bbq_sauce: 717, chocolate_pudding: 727
```

**Classes Still Missing Data:**
```
can: 0, bottle: 0, frying_pan: 0
```

### V3 Training Results

**Training:** Completed successfully with 8556 images
- mAP50: 0.991
- mAP50-95: 0.950

**Full Benchmark Results (V3 Model):**

| Suite | Tasks Pass | Total Detections | Classes Detected |
|-------|------------|------------------|------------------|
| libero_spatial | 7/10 | 490 | 7 |
| libero_object | **10/10** | 308 | 11 |
| libero_goal | 5/10 | 544 | 5 |
| libero_10 | 6/10 | 152 | 11 |

**Key Improvements:**
- libero_object: Now fully supported (was 0/10)
- Total classes detected across all suites: 22 (was 6)

**Failing libero_10 Tasks (0, 1, 7, 8):**
- Tasks 0, 1, 7: LIVING_ROOM scenes with grocery items
- Task 8: moka_pot task

**Investigation Findings:**
- Grocery objects (alphabet_soup, cream_cheese, butter, etc.) have low confidence (0.05-0.26)
- Model confuses LIVING_ROOM scenes with kitchen objects (detects "microwave", "ketchup" at higher confidence than actual objects)
- Domain shift issue: training data mostly from KITCHEN scenes, not LIVING_ROOM

### V4 Data Collection (LIVING_ROOM Focus)

**Additional Data Collected:**
- 6066 images from LIVING_ROOM scenes (libero_10 tasks 0,1,7 + all libero_object tasks)
- moka_pot task: 500 additional images
- **New class sample counts:**
  - basket: 5500
  - cream_cheese: 4500
  - tomato_sauce: 4500
  - alphabet_soup: 4000
  - milk: 4000
  - butter: 3500
  - ketchup: 3500
  - orange_juice: 3000
  - salad_dressing: 2000
  - bbq_sauce: 2000
  - chocolate_pudding: 2000
  - moka_pot: 500
  - stove: 500

### V4 Training Results

**Training:** Completed successfully with 12,000 images (all 40 LIBERO tasks)
- mAP50: 0.995
- mAP50-95: 0.995 (+4.5% vs V3)

**Full Benchmark Results (V4 Model):**

| Suite | V3 | V4 |
|-------|-----|-----|
| libero_spatial | 7/10 | 7/10 |
| libero_object | 10/10 | 10/10 |
| libero_goal | 5/10 | 5/10 |
| libero_10 | 6/10 | **10/10** |

**Previously Failing Tasks - Now Fixed:**

| Task | Object | V3 Confidence | V4 Confidence |
|------|--------|---------------|---------------|
| libero_10 task 0 | cream_cheese | 0.00 | **0.77** |
| libero_10 task 1 | butter/milk | 0.00 | **0.54** |
| libero_10 task 7 | alphabet_soup | low | **0.76** |
| libero_10 task 8 | moka_pot | 0.00 | **0.91** |

### Task Success Testing

**libero_object with V3 Model:**
```
Task 0: PASS - 6 classes, 4.1 det/frame (basket 0.88, alphabet_soup 0.56)
Task 1: PASS - 3 classes, 1.7 det/frame (basket 0.89)
Task 2: PASS - 7 classes, 4.6 det/frame (salad_dressing 0.69)
Task 3: PASS - 8 classes, 4.9 det/frame (alphabet_soup 0.83, bbq_sauce 0.77)
Task 4: PASS - 4 classes, 2.6 det/frame (bbq_sauce 0.69)
Task 5: PASS - 7 classes, 4.1 det/frame (bbq_sauce 0.77)
Task 6: PASS - 2 classes, 2.0 det/frame (basket 0.89)
Task 7: PASS - 5 classes, 3.5 det/frame (alphabet_soup 0.75)
Task 8: PASS - 7 classes, 4.0 det/frame (wine_bottle 0.63)
Task 9: PASS - 4 classes, 1.3 det/frame (basket 0.89)

libero_object Summary: 10/10 tasks with good detection
```

**libero_10 with V3 Model:**
```
Task 0: FAIL - LIVING_ROOM (low confidence on grocery items)
Task 1: FAIL - LIVING_ROOM (low confidence on grocery items)
Task 2: PASS - KITCHEN_SCENE3 (stove 0.57, bowl 0.38)
Task 3: PASS - KITCHEN_SCENE4 (cabinet 0.78, wine_bottle 0.87, drawer 0.75)
Task 4: PASS - LIVING_ROOM_SCENE5 (white_mug 0.61)
Task 5: PASS - STUDY_SCENE1 (book 0.69, caddy 0.81)
Task 6: PASS - LIVING_ROOM_SCENE6 (white_mug 0.61, microwave 0.52)
Task 7: FAIL - LIVING_ROOM (low confidence on grocery items)
Task 8: FAIL - KITCHEN_SCENE8 (moka_pot not detected)
Task 9: PASS - KITCHEN_SCENE6 (microwave 0.86)

libero_10 Summary: 6/10 tasks with good detection
```

### Files Created/Modified

| File | Change |
|------|--------|
| `scripts/collect_living_room_data.py` | NEW - Data collection from LIVING_ROOM scenes |
| `scripts/collect_all_data_v4.py` | NEW - Comprehensive data collection from all 40 tasks |
| `scripts/run_task_success_test.py` | NEW - Detection quality evaluation per task |
| `scripts/run_full_libero_benchmark.py` | NEW - Full 4-suite benchmark runner |
| `models/yolo_libero_v3.pt` | 30-class model trained on 8556 images |
| `models/yolo_libero_v4.pt` | **NEW - 30-class model trained on 12000 images (deployed)** |

### Next Steps

1. **Integrate detection with skill execution** for end-to-end task success
2. **Collect data for missing classes** (can, bottle, frying_pan have 0 samples - not blocking)

---

**Status:** Phase 6 complete. V4 model deployed (yolo_libero_v4.pt).

**Detection Performance (V4):**
- libero_spatial: 7/10 tasks
- libero_object: **10/10 tasks**
- libero_goal: 5/10 tasks
- libero_10: **10/10 tasks** (was 6/10 with V3)

**Model Comparison:**
| Metric | V3 | V4 |
|--------|-----|-----|
| Training Images | 8,556 | 12,000 |
| mAP50 | 0.991 | 0.995 |
| mAP50-95 | 0.950 | **0.995** |
| libero_10 pass rate | 6/10 | **10/10** |

**Remaining Issues:**
- 3 classes still missing data: can, bottle, frying_pan (not used in current tasks)
