# Zero-Shot LIBERO Implementation Plan

## Goal
Zero-shot task execution on LIBERO using:
- **Qwen2.5-VL-7B** (semantic grounding + planning)
- **Visual Servo (B3)** (local geometric stabilizer)
- **Skill Library** (composable motor primitives)

## System Architecture

```
Language Task
     ↓
Qwen Semantic Grounding (Layer 2.5)
     ↓
Symbolic World State (Layer 4)
     ↓
Qwen Skill Planner (Layer 5)
     ↓
Skill Library (Layer 3)
     ↓
Visual Servo + Motor Primitives (Layer 1)
     ↓
PD / Impedance Controller (Layer 0)
     ↓
Robot Executes (Closed Loop)
```

## Layer Reference

| Layer | Name | Frequency | Role |
|-------|------|-----------|------|
| 0 | Motor Controller (PD/Impedance) | 100-1000 Hz | Joint-space control, safety |
| 1 | Visual Servo + Motor Primitives | 50-100 Hz | Image-space to joint-space mapping |
| 2 | Geometric Perception | 10-30 Hz | Segmentation, keypoints, depth |
| 2.5 | Qwen Semantic Grounding | 1-2 Hz | Object→role mapping |
| 3 | Skill Library | On-demand | Composable motor primitives |
| 3.5 | Forward Model (Cerebellum) | 50-100 Hz | Predict outcomes, safety |
| 4 | Symbolic World Model | On-demand | State tracking (holding, on, in) |
| 5 | Qwen Task Planner | 0.5-2 Hz | Skill sequencing |
| 6 | Execution Monitor | Continuous | Failure detection, replanning |

---

## Phase 1: Oracle Perception + Classical Skills

**Goal**: Prove the architecture works with perfect perception.

**Layers Activated**: 0, 2 (oracle), 3, 4

### Deliverables

#### 1. Oracle Perception API (Layer 2 as Ground Truth)

```python
def oracle_perception(env):
    return {
        "objects": env.get_object_poses(),      # {name: pose}
        "gripper_pose": env.get_gripper_pose(), # 7D pose
        "gripper_width": env.get_gripper_width(),
        "object_names": env.get_object_names()
    }
```

No learning. Uses robosuite/LIBERO privileged access.

#### 2. Symbolic World Model (Layer 4)

```python
world_state = {
    "holding": None,           # Currently held object
    "on": {},                  # obj → surface
    "in": {},                  # obj → container
    "open": {},                # container → bool
}
```

Update rules:
- After `GraspObject`: `holding = obj`
- After `PlaceObject`: `in[obj] = container`
- After `OpenDrawer`: `open[drawer] = True`

#### 3. Classical Skills (Layer 3)

Cartesian PD control with position + orientation error:

```python
def pd_control(current_pose, target_pose, kp_pos=5.0, kp_ori=2.0):
    pos_error = target_pose[:3] - current_pose[:3]
    ori_error = quat_error(target_pose[3:7], current_pose[3:7])
    action = np.concatenate([kp_pos * pos_error, kp_ori * ori_error, [gripper]])
    return np.clip(action, -1, 1)
```

**Core Skills**:
- `ApproachObject(obj)` - Move to pre-grasp pose
- `GraspObject(obj)` - Close gripper on object
- `MoveObjectToRegion(obj, region)` - Transport held object
- `PlaceObject(obj, region)` - Lower and release

Each skill has:
- **Preconditions**: What must be true before execution
- **Postconditions**: What should be true after success
- **World state updates**: How state changes

#### 4. Project Structure

```
brain_robot/
├── skills/
│   ├── __init__.py
│   ├── base_skill.py          # Skill base class
│   ├── approach.py            # ApproachObject
│   ├── grasp.py               # GraspObject
│   ├── move.py                # MoveObjectToRegion
│   └── place.py               # PlaceObject
├── perception/
│   ├── __init__.py
│   └── oracle.py              # Ground truth from simulator
├── world_model/
│   ├── __init__.py
│   └── symbolic_state.py      # World state tracking
└── control/
    ├── __init__.py
    └── cartesian_pd.py        # Pose controller
```

### Success Criteria

- [ ] Pick & place works without Qwen (hardcoded skill sequence)
- [ ] All skills have pre/postconditions
- [ ] World state updates correctly after each skill
- [ ] GIF recording showing execution
- [ ] Works on at least 1 LIBERO task

### Estimated Time: 4-6 days

---

## Phase 2: Qwen Skill Planning (Layer 5)

**Goal**: Make Qwen sequence skills using world state.

**Layers Activated**: 4, 5, 3, 0

### Deliverables

#### 1. Skill Schema Definition

```python
SKILLS = {
    "ApproachObject(obj)": {
        "description": "Move gripper to pre-grasp pose above object",
        "preconditions": ["holding == None", "obj is reachable"],
        "postconditions": ["gripper above obj"]
    },
    "GraspObject(obj)": {
        "description": "Close gripper on object and lift slightly",
        "preconditions": ["gripper above obj", "holding == None"],
        "postconditions": ["holding == obj"]
    },
    "MoveObjectToRegion(obj, region)": {
        "description": "Move held object to target region",
        "preconditions": ["holding == obj"],
        "postconditions": ["obj above region"]
    },
    "PlaceObject(obj, region)": {
        "description": "Lower object and release",
        "preconditions": ["holding == obj", "obj above region"],
        "postconditions": ["holding == None", "in[obj] == region"]
    }
}
```

#### 2. Planner Prompt

```
You are a robot task planner.

Task: {task_description}
World State: {world_state_json}
Available Skills: {skill_schemas}

Output a JSON list of skill calls that accomplish the task.
Only use skills from the available list.
Ensure preconditions are satisfied before each skill.

Output format:
[
  {"skill": "SkillName", "args": {"arg1": "value1"}},
  ...
]
```

#### 3. Execution Loop

```python
def execute_plan(plan, env, world_state):
    for skill_call in plan:
        # Validate preconditions
        if not check_preconditions(skill_call, world_state):
            return False, f"Precondition failed: {skill_call}"

        # Execute skill
        success = execute_skill(skill_call, env)

        # Update world state
        if success:
            update_world_state(skill_call, world_state)
        else:
            return False, f"Skill failed: {skill_call}"

    return True, "Plan completed"
```

#### 4. Validation Layer

```python
def validate_plan(plan, available_skills, world_state):
    for step in plan:
        # Check skill exists
        if step['skill'] not in available_skills:
            return False, f"Unknown skill: {step['skill']}"

        # Check required args present
        required_args = get_required_args(step['skill'])
        for arg in required_args:
            if arg not in step.get('args', {}):
                return False, f"Missing arg {arg} for {step['skill']}"

    return True, None
```

### Success Criteria

- [ ] Qwen outputs valid, parseable skill programs
- [ ] Validation catches malformed plans
- [ ] Skills execute sequentially with world state updates
- [ ] Works on 3-5 different LIBERO tasks
- [ ] Handles retry on invalid Qwen output

### Estimated Time: 3-4 days

---

## Phase 3: Qwen Semantic Grounding (Layer 2.5)

**Goal**: Qwen maps detected objects → task roles (no hardcoded object selection).

**Layers Activated**: 2 (oracle), 2.5, 4, 5

### Deliverables

#### 1. Grounding Prompt

```
You are a robot perception assistant.

Task: {task_description}

Detected objects in the scene:
{detected_objects_with_descriptions}

Identify which objects are relevant for this task and their roles.

Output JSON:
{
  "source_object": "object_id for the object to manipulate",
  "target_location": "object_id or region for placement",
  "other_relevant": ["any other relevant object_ids"]
}
```

#### 2. Object Description Enrichment

```python
def enrich_detected_objects(object_names):
    """Add human-readable descriptions for Qwen."""
    enriched = []
    for name in object_names:
        # Parse LIBERO naming convention
        # e.g., "akita_black_bowl_1" → "black bowl"
        description = parse_object_description(name)
        enriched.append({
            "id": name,
            "description": description
        })
    return enriched
```

#### 3. Integration with Planning

```python
def full_pipeline(task_description, env):
    # Layer 2: Oracle perception
    perception = oracle_perception(env)

    # Layer 2.5: Qwen grounding
    objects = enrich_detected_objects(perception['object_names'])
    grounding = qwen_ground_objects(task_description, objects)

    # Layer 4: Initialize world state
    world_state = init_world_state(perception, grounding)

    # Layer 5: Qwen planning
    plan = qwen_plan(task_description, world_state, SKILLS)

    # Layer 3: Execute skills
    success, msg = execute_plan(plan, env, world_state)

    return success, msg
```

### Success Criteria

- [ ] No hardcoded object selection anywhere
- [ ] Qwen correctly identifies source/target for various tasks
- [ ] Handles ambiguous cases (multiple valid choices)
- [ ] End-to-end execution without manual object specification
- [ ] Works on 5+ LIBERO tasks

### Estimated Time: 3-4 days

---

## Future Phases (Post-MVP)

### Phase 4: Learned Geometric Perception
- Replace oracle with SAM/DINO/depth estimation
- Target: <30% failure rate on short-horizon tasks

### Phase 5: Visual Servo (B3)
- Replace Cartesian PD with image-based visual servoing
- Critical for real robot deployment

### Phase 6: Monitoring & Replanning
- Postcondition verification
- Qwen-based failure recovery

---

## Key Design Principles

1. **Qwen does semantics, not geometry** - Never ask Qwen for pixel coordinates or distances in meters

2. **Oracle first, learned later** - Validate architecture with perfect perception before adding noise

3. **Skills are generic** - `GraspObject(id)` not `GraspBlueMug()` enables reuse

4. **Explicit state tracking** - Symbolic world model enables multi-step reasoning

5. **Validation at every layer** - Check preconditions, validate Qwen output, verify postconditions

---

## Timeline Summary

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Oracle + Classical Skills | 4-6 days |
| 2 | Qwen Planning | 3-4 days |
| 3 | Qwen Grounding | 3-4 days |
| **MVP Total** | **End-to-end zero-shot** | **~2 weeks** |
| 4 | Learned Perception | 1-2 weeks |
| 5 | Visual Servo | 1 week |
| 6 | Monitoring | 3-5 days |
