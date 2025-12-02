# Future Work and Deferred Tests

This document tracks planned improvements and tests that are deferred for later phases.

## Deferred Tests

### Object ID Randomization Test

**Status**: Deferred to Phase 4 or later

**Purpose**: Verify that the semantic grounder works correctly regardless of object ID ordering or naming variations (e.g., `akita_black_bowl_1_main` vs `akita_black_bowl_2_main`).

**Background**:
During Phase 3 development, GPT suggested adding tests to ensure grounding accuracy isn't dependent on:
1. Object ordering in the list presented to Qwen
2. Specific instance numbers in object IDs
3. Random seed variations affecting object placement

**Why Deferred**:
- Current grounding accuracy is 100% on the 6 LIBERO spatial tasks (18/18 episodes)
- Spatial context differentiation (`spatial_text`) handles most disambiguation cases
- The remaining edge cases (e.g., two identical bowls with identical spatial context) require either:
  - BDDL-level ground truth for which specific instance the task refers to
  - Enhanced spatial reasoning (left/right, near/far)

**Implementation Notes for Later**:
```python
def test_id_randomization():
    """Test grounding robustness to ID ordering and naming."""
    # 1. Run same task with shuffled object list order
    # 2. Verify grounding picks semantically correct object
    # 3. For ambiguous cases, verify any matching object is acceptable

    # Key: need to define what "correct" means when task is ambiguous
    # e.g., "pick up a bowl" with two identical bowls
```

**Dependencies**:
- Enhanced spatial relations (Phase 4 learned perception may provide better features)
- BDDL goal parsing to extract which specific instance is targeted

---

## Phase 4 Preparation Notes

### Data Collection for Learned Perception

During Phase 1-3 execution, we should collect:
- RGB images at key frames (pre-grasp, grasp, post-grasp, place)
- Ground truth object poses from oracle
- Ground truth gripper pose
- Task success/failure labels

This data enables training perception models in Phase 4.

### Metrics to Establish Before Phase 4

1. **Pose estimation error**: Mean L2 distance between predicted and GT position
2. **Orientation error**: Angular error in quaternion space
3. **Spatial relation accuracy**: Does learned perception reproduce ON/INSIDE correctly?
4. **Latency**: Must maintain <50ms per frame for real-time execution

---

## Known Limitations

### LIBERO-Specific

1. **Drawer detection**: Uses geometric heuristics based on `cabinet_top` naming convention. May not generalize to other drawer designs.

2. **ON relation surface types**: Currently only detects objects ON `cookies`, `ramekin`, `plate`, etc. based on hardcoded patterns. Learned perception could generalize this.

3. **INSIDE relation**: Limited to drawer detection. Full container reasoning (boxes, bins) deferred.

### Qwen Grounding

1. **Paraphrase robustness**: Tested on limited paraphrases ("move onto", "put on", "grab and place"). More extensive paraphrase testing needed.

2. **Multi-object disambiguation**: When task says "the bowl" but multiple bowls exist with identical spatial context, grounding picks arbitrarily. This is acceptable per current architecture but could be improved.

---

## Suggested Improvements (Not Blocking)

1. **Confidence thresholds**: Add confidence-based fallback when Qwen grounding confidence is low

2. **Interactive clarification**: When grounding is ambiguous, could potentially ask user for clarification (out of scope for autonomous operation)

3. **Spatial relation caching**: Currently recomputed every perception cycle; could cache and update incrementally

4. **Visualization dashboard**: Real-time display of spatial relations, grounding decisions, and execution state
