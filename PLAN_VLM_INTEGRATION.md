# Plan: Real VLM Integration (Option C)

## Overview

Integrate Qwen2.5-VL-7B as the high-level planner for the brain-inspired robot control system. The VLM will process robot camera images and task descriptions to generate structured motion plans that condition the brain-inspired policy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIBERO Environment                            │
│  - Camera image (128x128 RGB)                                    │
│  - Task description (e.g., "pick up the black bowl...")         │
│  - Proprioception (15 dims)                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Qwen2.5-VL-7B Planner (1 Hz)                       │
│  Input: Image + Task description + Previous phase               │
│  Output: JSON plan {phase, movements, gripper, confidence}      │
│  - Runs asynchronously every N steps or on phase change         │
│  - Cached plan used between VLM calls                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ JSON Plan
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Plan Encoder (existing)                             │
│  - Encodes JSON plan to 128-dim embedding                       │
│  - Phase, direction, speed, gripper embeddings                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ plan_embed (128 dims)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Brain-Inspired Policy (20+ Hz)                          │
│  - Primitive Selector: plan_embed → primitive weights           │
│  - Motion Primitives: weighted blend                            │
│  - Modulator: amplitude, speed, offset                          │
│  - (Optional) Forward model for error correction                │
└────────────────────────┬────────────────────────────────────────┘
                         │ action (7 dims)
                         ▼
                    Robot Control
```

## Implementation Steps

### Step 1: Create VLM-Conditioned Policy Wrapper
**File:** `brain_robot/policy/vlm_policy.py`

```python
class VLMConditionedPolicy:
    """
    Wrapper that combines:
    1. Qwen2.5-VL for high-level planning
    2. Brain-inspired policy for low-level control
    """

    def __init__(self, vlm, brain_policy, plan_interval=10):
        self.vlm = vlm  # QwenVLPlanner
        self.brain_policy = brain_policy  # BrainInspiredActionGenerator
        self.plan_interval = plan_interval  # Steps between VLM queries
        self.cached_plan = None
        self.steps_since_plan = 0

    def get_action(self, image, task_desc, proprio):
        # Query VLM if needed
        if self.should_replan():
            self.cached_plan = self.vlm.plan(image, task_desc)
            self.steps_since_plan = 0

        # Get action from brain policy using cached plan
        action = self.brain_policy(self.cached_plan, proprio)
        self.steps_since_plan += 1

        return action, self.cached_plan
```

### Step 2: Add Asynchronous VLM Planning
**File:** `brain_robot/vlm/async_planner.py`

- Run VLM in background thread
- Use queue for image/task input
- Cache results for policy to consume
- Handle VLM failures gracefully (use previous plan)

### Step 3: Create Training Script with VLM
**File:** `scripts/train_vlm_conditioned.py`

Two training modes:
1. **Offline (BC with VLM labels)**: Pre-compute VLM plans for demo trajectories
2. **Online (VLM in loop)**: VLM generates plans during rollouts

### Step 4: Create Evaluation Script with Visualization
**File:** `scripts/eval_vlm_libero.py`

- Run VLM-conditioned policy on LIBERO tasks
- Generate GIFs with VLM output overlay:
  - Current phase
  - Planned direction
  - Primitive weights
  - Confidence score
- Compare against baseline (no VLM)

### Step 5: Optimize for Speed
- Use smaller VLM (Qwen2.5-VL-2B) if 7B is too slow
- Implement plan caching and smart re-planning
- Use quantization (4-bit) if memory is an issue

## Key Design Decisions

### 1. When to Re-Query VLM
Options:
- **Fixed interval**: Every N steps (e.g., N=20)
- **Phase change detection**: When phase likely changed
- **Uncertainty-based**: When confidence drops below threshold
- **Combination**: Fixed interval + phase change trigger

**Recommendation**: Start with fixed interval (N=20), add phase change detection later.

### 2. VLM Prompt Design for LIBERO
Need to adapt prompts for LIBERO-specific:
- Object names (black bowl, plate, ramekin, etc.)
- Scene understanding (table layout)
- Task descriptions (pick and place)

### 3. Training Strategy
**Phase 1: Pre-compute VLM plans for demos**
- Run VLM on all demo images (offline)
- Save (image, vlm_plan, action) tuples
- Train brain policy with BC on these

**Phase 2: Online fine-tuning**
- Run VLM during rollouts
- Fine-tune policy with PPO or DAgger

### 4. Visualization
Each GIF frame should show:
```
┌─────────────────────────────────────────────┐
│  Robot Camera View                          │
│  [128x128 RGB image]                        │
├─────────────────────────────────────────────┤
│  VLM Output:                                │
│  Phase: approach → grasp                    │
│  Direction: forward, down                   │
│  Gripper: open → close                      │
│  Confidence: 0.85                           │
├─────────────────────────────────────────────┤
│  Policy Info:                               │
│  Primitives: [0.1, 0.0, 0.7, 0.0, 0.2, 0.0]│
│  Phase bar: [=====>     ] 50%              │
└─────────────────────────────────────────────┘
```

## Files to Create/Modify

### New Files:
1. `brain_robot/policy/vlm_policy.py` - VLM-conditioned policy wrapper
2. `brain_robot/vlm/async_planner.py` - Async VLM planning
3. `scripts/train_vlm_conditioned.py` - Training with VLM
4. `scripts/eval_vlm_libero.py` - Evaluation with visualization
5. `scripts/precompute_vlm_plans.py` - Pre-compute VLM plans for demos

### Modify:
1. `brain_robot/vlm/prompts.py` - Add LIBERO-specific prompts
2. `brain_robot/vlm/qwen_planner.py` - Add batch inference, caching

## Estimated Effort

| Task | Complexity | Est. Time |
|------|------------|-----------|
| VLM-conditioned policy wrapper | Medium | 1 hour |
| Async VLM planning | Medium | 1 hour |
| Pre-compute VLM plans | Low | 30 min |
| Training script | Medium | 1 hour |
| Evaluation with visualization | Medium | 1 hour |
| LIBERO prompt tuning | Low | 30 min |
| Testing & debugging | High | 2 hours |
| **Total** | | **~7 hours** |

## Success Criteria

1. VLM correctly identifies task phases from images
2. Brain policy responds to VLM plans appropriately
3. GIF visualizations clearly show VLM reasoning
4. Performance comparable to or better than mock VLM baseline
5. System runs at acceptable speed (>5 FPS with VLM, >20 FPS policy-only)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| VLM too slow | Use plan caching, reduce image resolution |
| VLM hallucinations | Validate outputs, use conservative fallback |
| GPU OOM | Use quantization, reduce batch size |
| Poor VLM grounding | Fine-tune prompts, add few-shot examples |

## Next Steps After Approval

1. Create `vlm_policy.py` with basic wrapper
2. Test VLM on LIBERO images (no policy yet)
3. Implement pre-computation of VLM plans
4. Train brain policy with VLM plans
5. Evaluate and generate visualization GIFs
