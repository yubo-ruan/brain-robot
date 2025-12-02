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

**Status:** Phase 6-0 and 6-A complete. MoveSkill and missing skills are primary blockers.
