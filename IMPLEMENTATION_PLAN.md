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

#### 2. Coordinate Frame Sanity (CRITICAL)

Before writing any skills, establish and verify coordinate frames:

```python
def test_coordinate_frames(env):
    """Verify all poses are in consistent world frame."""
    # 1. Spawn one object
    env.reset()

    # 2. Get poses from different sources
    obj_pose_sim = env.sim.data.body_xpos['object_name']
    obj_pose_api = oracle_perception(env)['objects']['object_name']
    gripper_pose = oracle_perception(env)['gripper_pose']

    # 3. Verify they're in same frame
    assert np.allclose(obj_pose_sim[:3], obj_pose_api[:3], atol=1e-3)

    # 4. Compute expected action direction
    direction = obj_pose_api[:3] - gripper_pose[:3]

    # 5. Verify action makes sense (e.g., positive X = forward)
    print(f"Direction to object: {direction}")
    # Manual sanity check: does this match visual observation?
```

**Canonical frame**: All poses in world frame (robosuite default).

#### 3. Symbolic World Model (Layer 4)

**Schema Design** (not just a simple dict):

```python
@dataclass
class WorldState:
    """Symbolic world state with explicit uncertainty handling."""

    # Core relations
    holding: Optional[str] = None          # Currently held object
    on: Dict[str, str] = field(default_factory=dict)    # obj → surface
    inside: Dict[str, str] = field(default_factory=dict) # obj → container
    open_state: Dict[str, bool] = field(default_factory=dict)  # container → bool

    # Object tracking
    objects: Dict[str, ObjectState] = field(default_factory=dict)

    # Uncertainty flags
    perception_stale: bool = False         # Perception hasn't updated recently
    last_perception_time: float = 0.0

    def update_from_perception(self, perception: dict):
        """Reconcile symbolic state with perception."""
        # Check for disappeared objects
        perceived_objects = set(perception['object_names'])
        known_objects = set(self.objects.keys())

        disappeared = known_objects - perceived_objects
        for obj in disappeared:
            # Object disappeared - mark relations as uncertain
            if obj in self.on:
                del self.on[obj]  # Or mark as uncertain?
            if obj in self.inside:
                del self.inside[obj]

        self.last_perception_time = time.time()
        self.perception_stale = False

    def to_dict(self) -> dict:
        """Serialize for Qwen prompt."""
        return {
            "holding": self.holding,
            "on": self.on,
            "inside": self.inside,
            "open": self.open_state,
        }

@dataclass
class ObjectState:
    """Per-object state."""
    name: str
    pose: np.ndarray              # 7D pose (pos + quat)
    last_seen: float              # Timestamp
    confidence: float = 1.0       # For learned perception later
```

**Update rules**:
- After `GraspObject(obj)`: `holding = obj`, remove `on[obj]`
- After `PlaceObject(obj, region)`: `holding = None`, `inside[obj] = region`
- After `OpenDrawer(drawer)`: `open_state[drawer] = True`

#### 4. Skill Base Class with Structured Results

```python
class SkillResult(NamedTuple):
    """Structured result from skill execution."""
    success: bool
    info: dict  # Skill-specific details

    # Common info keys:
    # - "reached_target": bool
    # - "final_pose": np.ndarray
    # - "error_msg": str
    # - "steps_taken": int

class Skill(ABC):
    """Base class for all skills."""

    name: str

    @abstractmethod
    def preconditions(self, world_state: WorldState, args: dict) -> Tuple[bool, str]:
        """Check if skill can execute. Returns (ok, reason)."""
        pass

    @abstractmethod
    def postconditions(self, world_state: WorldState, args: dict) -> Tuple[bool, str]:
        """Check if skill succeeded. Returns (ok, reason)."""
        pass

    @abstractmethod
    def execute(self, env, world_state: WorldState, args: dict) -> SkillResult:
        """Execute the skill. Returns structured result."""
        pass

    @abstractmethod
    def update_world_state(self, world_state: WorldState, args: dict, result: SkillResult):
        """Update world state after successful execution."""
        pass

    def run(self, env, world_state: WorldState, args: dict) -> SkillResult:
        """Full skill execution with pre/post checks."""
        # Check preconditions
        ok, reason = self.preconditions(world_state, args)
        if not ok:
            return SkillResult(success=False, info={"error": f"Precondition failed: {reason}"})

        # Execute
        result = self.execute(env, world_state, args)

        # Update world state if successful
        if result.success:
            self.update_world_state(world_state, args, result)

        return result
```

#### 5. Classical Skills (Layer 3)

Cartesian PD control with position + orientation error:

```python
def pd_control(current_pose, target_pose, kp_pos=5.0, kp_ori=2.0):
    pos_error = target_pose[:3] - current_pose[:3]
    ori_error = quat_error(target_pose[3:7], current_pose[3:7])
    action = np.concatenate([kp_pos * pos_error, kp_ori * ori_error, [gripper]])
    return np.clip(action, -1, 1)
```

**Core Skills**:
- `ApproachObject(obj)` - Move to pre-grasp pose (10cm above object)
- `GraspObject(obj)` - Lower, close gripper, lift slightly
- `MoveObjectToRegion(obj, region)` - Transport held object
- `PlaceObject(obj, region)` - Lower and release

#### 6. Logging Pipeline (CRITICAL for Future Iteration)

```python
@dataclass
class EpisodeLog:
    """Complete episode record for debugging and training."""

    task: str
    task_id: int
    timestamp: str

    # Execution trace
    skill_sequence: List[dict]           # [{skill, args, result}, ...]
    world_state_trace: List[dict]        # State after each skill

    # Observations (for BC/RL later)
    observations: List[dict]             # Key frames: {image, proprio, step}
    actions: List[np.ndarray]            # All actions taken

    # Outcome
    success: bool
    failure_reason: Optional[str]
    total_steps: int
    total_time: float

    # Qwen interactions (for Phase 2+)
    qwen_prompts: List[str] = field(default_factory=list)
    qwen_responses: List[str] = field(default_factory=list)

class Logger:
    """Episode logger."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode: Optional[EpisodeLog] = None

    def start_episode(self, task: str, task_id: int):
        self.current_episode = EpisodeLog(
            task=task,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            skill_sequence=[],
            world_state_trace=[],
            observations=[],
            actions=[],
            success=False,
            failure_reason=None,
            total_steps=0,
            total_time=0.0,
        )

    def log_skill(self, skill_name: str, args: dict, result: SkillResult):
        self.current_episode.skill_sequence.append({
            "skill": skill_name,
            "args": args,
            "success": result.success,
            "info": result.info,
        })

    def log_world_state(self, world_state: WorldState):
        self.current_episode.world_state_trace.append(world_state.to_dict())

    def log_observation(self, obs: dict, step: int):
        # Store key frames (not every step - too large)
        if step % 10 == 0:
            self.current_episode.observations.append({
                "step": step,
                "proprio": obs['robot0_eef_pos'].tolist(),
                "gripper": obs['robot0_gripper_qpos'].tolist(),
                # Image stored separately as PNG to save space
            })

    def end_episode(self, success: bool, failure_reason: Optional[str] = None):
        self.current_episode.success = success
        self.current_episode.failure_reason = failure_reason

        # Save to disk
        filename = f"episode_{self.current_episode.timestamp}.json"
        with open(self.output_dir / filename, 'w') as f:
            json.dump(asdict(self.current_episode), f, indent=2, default=str)
```

#### 7. Determinism & Testing

```python
def set_deterministic(seed: int = 42):
    """Fix all RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # LIBERO/robosuite seeds set via env.reset(seed=seed)

# Unit-like tests for skills
def test_approach_skill():
    """ApproachObject brings gripper within threshold 95% of time."""
    env = make_libero_env(task_suite="libero_spatial", task_id=0)
    skill = ApproachSkill()

    successes = 0
    for _ in range(20):
        env.reset()
        perception = oracle_perception(env)
        obj_name = list(perception['objects'].keys())[0]

        world_state = WorldState()
        result = skill.run(env, world_state, {"obj": obj_name})

        if result.success:
            # Verify gripper is actually close
            final_gripper = oracle_perception(env)['gripper_pose'][:3]
            obj_pos = perception['objects'][obj_name][:3]
            dist = np.linalg.norm(final_gripper - obj_pos - [0, 0, 0.1])  # Pre-grasp offset
            if dist < 0.03:  # 3cm threshold
                successes += 1

    assert successes >= 19, f"ApproachSkill only succeeded {successes}/20 times"
```

#### 8. Project Structure

```
brain_robot/
├── skills/
│   ├── __init__.py
│   ├── base.py                # Skill, SkillResult base classes
│   ├── approach.py            # ApproachObject
│   ├── grasp.py               # GraspObject
│   ├── move.py                # MoveObjectToRegion
│   └── place.py               # PlaceObject
├── perception/
│   ├── __init__.py
│   └── oracle.py              # Ground truth from simulator
├── world_model/
│   ├── __init__.py
│   ├── state.py               # WorldState, ObjectState classes
│   └── relations.py           # Relation update logic
├── control/
│   ├── __init__.py
│   └── cartesian_pd.py        # Pose controller
├── logging/
│   ├── __init__.py
│   └── episode_logger.py      # EpisodeLog, Logger classes
└── tests/
    ├── __init__.py
    ├── test_frames.py         # Coordinate frame tests
    └── test_skills.py         # Skill unit tests
```

### Success Criteria

- [ ] Coordinate frame test passes
- [ ] Pick & place works without Qwen (hardcoded skill sequence)
- [ ] All skills return structured SkillResult
- [ ] World state updates correctly after each skill
- [ ] Logging saves episode traces to disk
- [ ] Skill unit tests pass (>95% success with oracle)
- [ ] GIF recording showing execution
- [ ] Works on at least 1 LIBERO task

### Estimated Time: 5-7 days (realistic)

---

## Phase 2: Qwen Skill Planning (Layer 5)

**Goal**: Make Qwen sequence skills using world state.

**Layers Activated**: 4, 5, 3, 0

### Deliverables

#### 1. Skill Schema Definition

```python
SKILL_SCHEMA = {
    "ApproachObject": {
        "description": "Move gripper to pre-grasp pose above object",
        "args": {"obj": "object_id"},
        "preconditions": ["holding == None", "obj exists"],
        "postconditions": ["gripper above obj"]
    },
    "GraspObject": {
        "description": "Close gripper on object and lift slightly",
        "args": {"obj": "object_id"},
        "preconditions": ["gripper above obj", "holding == None"],
        "postconditions": ["holding == obj"]
    },
    "MoveObjectToRegion": {
        "description": "Move held object to target region",
        "args": {"obj": "object_id", "region": "region_id"},
        "preconditions": ["holding == obj"],
        "postconditions": ["obj above region"]
    },
    "PlaceObject": {
        "description": "Lower object and release",
        "args": {"obj": "object_id", "region": "region_id"},
        "preconditions": ["holding == obj", "obj above region"],
        "postconditions": ["holding == None", "inside[obj] == region"]
    }
}
```

#### 2. Planner Prompt with JSON Schema

```python
PLANNER_PROMPT = """You are a robot task planner.

Task: {task_description}

Current World State:
{world_state_json}

Available Skills:
{skill_schemas}

Output a JSON list of skill calls to accomplish the task.
Rules:
- Only use skills from the available list
- Ensure preconditions are met before each skill
- Maximum 10 skills per plan
- Output ONLY valid JSON, no explanations

Output format:
[
  {{"skill": "SkillName", "args": {{"arg1": "value1"}}}},
  ...
]"""
```

#### 3. Validation Layer with Guardrails

```python
def validate_plan(plan: List[dict], world_state: WorldState) -> Tuple[bool, str]:
    """Validate Qwen's plan before execution."""

    # 1. Check plan is a list
    if not isinstance(plan, list):
        return False, "Plan must be a list"

    # 2. Check length
    if len(plan) > 10:
        return False, f"Plan too long: {len(plan)} > 10"

    # 3. Check each skill
    seen_skills = []
    for i, step in enumerate(plan):
        skill_name = step.get('skill')
        args = step.get('args', {})

        # Check skill exists
        if skill_name not in SKILL_SCHEMA:
            return False, f"Step {i}: Unknown skill '{skill_name}'"

        # Check required args
        schema = SKILL_SCHEMA[skill_name]
        for arg_name in schema['args']:
            if arg_name not in args:
                return False, f"Step {i}: Missing arg '{arg_name}' for {skill_name}"

        # Detect loops (same skill with same args 3+ times in a row)
        seen_skills.append((skill_name, tuple(sorted(args.items()))))
        if len(seen_skills) >= 3:
            last_three = seen_skills[-3:]
            if last_three[0] == last_three[1] == last_three[2]:
                return False, f"Step {i}: Detected loop - {skill_name} repeated 3 times"

    return True, "Plan valid"

def parse_qwen_output(raw_output: str) -> Tuple[List[dict], str]:
    """Parse Qwen output, attempting to fix common JSON issues."""

    # Try direct parse first
    try:
        plan = json.loads(raw_output)
        return plan, None
    except json.JSONDecodeError as e:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
    if match:
        try:
            plan = json.loads(match.group(1))
            return plan, None
        except:
            pass

    # Try to find JSON array in text
    match = re.search(r'\[[\s\S]*\]', raw_output)
    if match:
        try:
            plan = json.loads(match.group(0))
            return plan, None
        except:
            pass

    return None, f"Could not parse JSON from: {raw_output[:200]}..."
```

#### 4. Execution Loop with Retry

```python
def plan_and_execute(task: str, env, world_state: WorldState, logger: Logger, max_retries: int = 3) -> bool:
    """Full planning and execution loop."""

    for attempt in range(max_retries):
        # Get plan from Qwen
        prompt = PLANNER_PROMPT.format(
            task_description=task,
            world_state_json=json.dumps(world_state.to_dict(), indent=2),
            skill_schemas=json.dumps(SKILL_SCHEMA, indent=2)
        )

        raw_output = qwen_query(prompt)
        logger.log_qwen(prompt, raw_output)

        # Parse output
        plan, parse_error = parse_qwen_output(raw_output)
        if parse_error:
            print(f"Attempt {attempt+1}: Parse error - {parse_error}")
            continue

        # Validate plan
        valid, validation_error = validate_plan(plan, world_state)
        if not valid:
            print(f"Attempt {attempt+1}: Validation error - {validation_error}")
            continue

        # Execute plan
        for skill_call in plan:
            skill = get_skill(skill_call['skill'])
            result = skill.run(env, world_state, skill_call['args'])
            logger.log_skill(skill_call['skill'], skill_call['args'], result)
            logger.log_world_state(world_state)

            if not result.success:
                print(f"Skill failed: {skill_call['skill']} - {result.info}")
                return False

        return True  # All skills succeeded

    return False  # All retries exhausted
```

#### 5. Prompt Revision Tracking

```python
@dataclass
class PromptFailure:
    """Record of a failed Qwen interaction for prompt improvement."""
    prompt: str
    raw_output: str
    error_type: str  # "parse_error", "validation_error", "execution_error"
    error_message: str
    task: str
    timestamp: str

class PromptTracker:
    """Track prompt failures for systematic improvement."""

    def __init__(self, output_dir: str):
        self.failures: List[PromptFailure] = []
        self.output_dir = Path(output_dir)

    def log_failure(self, prompt: str, output: str, error_type: str, error_msg: str, task: str):
        self.failures.append(PromptFailure(
            prompt=prompt,
            raw_output=output,
            error_type=error_type,
            error_message=error_msg,
            task=task,
            timestamp=datetime.now().isoformat()
        ))

    def save(self):
        with open(self.output_dir / "prompt_failures.json", 'w') as f:
            json.dump([asdict(f) for f in self.failures], f, indent=2)

    def analyze(self) -> dict:
        """Analyze failure patterns."""
        by_type = defaultdict(list)
        for f in self.failures:
            by_type[f.error_type].append(f)

        return {
            "total_failures": len(self.failures),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "common_errors": self._find_common_errors()
        }
```

### Success Criteria

- [ ] Qwen outputs valid, parseable skill programs (>80% first try)
- [ ] Validation catches malformed plans
- [ ] Loop detection works
- [ ] Retry logic recovers from parse errors
- [ ] Skills execute sequentially with world state updates
- [ ] Prompt failures are logged for analysis
- [ ] Works on 3-5 different LIBERO tasks
- [ ] All Qwen interactions logged

### Estimated Time: 4-5 days (realistic)

---

## Phase 3: Qwen Semantic Grounding (Layer 2.5)

**Goal**: Qwen maps detected objects → task roles (no hardcoded object selection).

**Layers Activated**: 2 (oracle), 2.5, 4, 5

### Deliverables

#### 1. Object Description Enrichment

```python
def parse_libero_object_name(name: str) -> dict:
    """Parse LIBERO naming convention into human-readable description.

    Examples:
        "akita_black_bowl_1" → {"id": "akita_black_bowl_1", "type": "bowl", "color": "black", "description": "black bowl"}
        "plate_1" → {"id": "plate_1", "type": "plate", "color": None, "description": "plate"}
        "wooden_cabinet_1" → {"id": "wooden_cabinet_1", "type": "cabinet", "material": "wooden", "description": "wooden cabinet"}
    """
    parts = name.lower().split('_')

    # Remove trailing number
    if parts[-1].isdigit():
        instance_id = parts[-1]
        parts = parts[:-1]
    else:
        instance_id = "1"

    # Known object types
    OBJECT_TYPES = ['bowl', 'plate', 'mug', 'cup', 'drawer', 'cabinet', 'box', 'can', 'bottle', 'ramekin']
    COLORS = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'brown']
    MATERIALS = ['wooden', 'metal', 'plastic', 'glass', 'ceramic']

    obj_type = None
    color = None
    material = None

    for part in parts:
        if part in OBJECT_TYPES:
            obj_type = part
        elif part in COLORS:
            color = part
        elif part in MATERIALS:
            material = part

    # Build description
    desc_parts = []
    if color:
        desc_parts.append(color)
    if material:
        desc_parts.append(material)
    if obj_type:
        desc_parts.append(obj_type)

    description = ' '.join(desc_parts) if desc_parts else name

    return {
        "id": name,
        "type": obj_type,
        "color": color,
        "material": material,
        "description": description,
        "instance": instance_id
    }

def enrich_detected_objects(object_names: List[str], object_poses: dict) -> List[dict]:
    """Create rich object descriptions for Qwen."""
    enriched = []
    for name in object_names:
        info = parse_libero_object_name(name)
        pose = object_poses.get(name)
        if pose is not None:
            info["position"] = f"at ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})"
        enriched.append(info)
    return enriched
```

#### 2. Grounding Prompt

```python
GROUNDING_PROMPT = """You are a robot perception assistant.

Task: {task_description}

Objects detected in the scene:
{objects_json}

Identify which objects are relevant for this task and their roles.

Rules:
- "source_object" is the object to pick up / manipulate
- "target_location" is where to place it (can be another object like a bowl, or a surface)
- If multiple objects could work (e.g., "a bowl"), pick any one
- Output ONLY valid JSON

Output format:
{{
  "source_object": "object_id",
  "target_location": "object_id or region name",
  "reasoning": "brief explanation"
}}"""
```

#### 3. Grounding Test Bench

```python
class GroundingTestBench:
    """Test semantic grounding without running full episodes."""

    def __init__(self):
        self.test_cases = []

    def add_test(self, task: str, objects: List[dict], expected_source: str, expected_target: str):
        self.test_cases.append({
            "task": task,
            "objects": objects,
            "expected_source": expected_source,
            "expected_target": expected_target
        })

    def run_tests(self, grounding_fn) -> dict:
        """Run all grounding tests."""
        results = {"passed": 0, "failed": 0, "failures": []}

        for tc in self.test_cases:
            result = grounding_fn(tc["task"], tc["objects"])

            source_ok = result.get("source_object") == tc["expected_source"]
            target_ok = result.get("target_location") == tc["expected_target"]

            if source_ok and target_ok:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "task": tc["task"],
                    "expected": {"source": tc["expected_source"], "target": tc["expected_target"]},
                    "got": {"source": result.get("source_object"), "target": result.get("target_location")}
                })

        return results

# Example test cases
def create_grounding_tests():
    bench = GroundingTestBench()

    bench.add_test(
        task="pick up the black bowl and place it on the plate",
        objects=[
            {"id": "akita_black_bowl_1", "description": "black bowl"},
            {"id": "plate_1", "description": "plate"},
            {"id": "ramekin_1", "description": "ramekin"}
        ],
        expected_source="akita_black_bowl_1",
        expected_target="plate_1"
    )

    bench.add_test(
        task="put the mug in the drawer",
        objects=[
            {"id": "mug_1", "description": "mug"},
            {"id": "wooden_drawer_1", "description": "wooden drawer"},
            {"id": "plate_1", "description": "plate"}
        ],
        expected_source="mug_1",
        expected_target="wooden_drawer_1"
    )

    return bench
```

#### 4. Ambiguity Handling

```python
@dataclass
class GroundingResult:
    """Result of semantic grounding with ambiguity info."""
    source_object: str
    target_location: str
    reasoning: str

    # Ambiguity tracking
    ambiguous_source: bool = False      # Multiple valid sources?
    ambiguous_target: bool = False      # Multiple valid targets?
    alternative_sources: List[str] = field(default_factory=list)
    alternative_targets: List[str] = field(default_factory=list)

def ground_with_ambiguity(task: str, objects: List[dict]) -> GroundingResult:
    """Ground objects with explicit ambiguity detection."""

    # Get Qwen's grounding
    prompt = GROUNDING_PROMPT.format(
        task_description=task,
        objects_json=json.dumps(objects, indent=2)
    )
    raw = qwen_query(prompt)
    result = json.loads(raw)

    # Detect ambiguity: are there other valid choices?
    task_lower = task.lower()

    # Check for ambiguous source
    source_type = None
    for obj in objects:
        if obj["id"] == result["source_object"]:
            source_type = obj.get("type")
            break

    alternative_sources = [
        obj["id"] for obj in objects
        if obj.get("type") == source_type and obj["id"] != result["source_object"]
    ]

    return GroundingResult(
        source_object=result["source_object"],
        target_location=result["target_location"],
        reasoning=result.get("reasoning", ""),
        ambiguous_source=len(alternative_sources) > 0,
        alternative_sources=alternative_sources
    )
```

#### 5. Full Pipeline Integration

```python
def full_pipeline(task_description: str, env, logger: Logger) -> bool:
    """End-to-end execution: perception → grounding → planning → execution."""

    logger.start_episode(task_description, env.task_id)

    # Layer 2: Oracle perception
    perception = oracle_perception(env)

    # Layer 2.5: Qwen grounding
    objects = enrich_detected_objects(
        perception['object_names'],
        perception['objects']
    )
    grounding = ground_with_ambiguity(task_description, objects)
    logger.log_grounding(grounding)

    if grounding.ambiguous_source:
        print(f"Note: Ambiguous source, chose {grounding.source_object} from {grounding.alternative_sources}")

    # Layer 4: Initialize world state
    world_state = WorldState()
    world_state.update_from_perception(perception)

    # Inject grounding into world state
    world_state.task_source = grounding.source_object
    world_state.task_target = grounding.target_location

    # Layer 5 + 3: Plan and execute
    success = plan_and_execute(task_description, env, world_state, logger)

    logger.end_episode(success)
    return success
```

### Success Criteria

- [ ] Object parsing works for all LIBERO naming conventions
- [ ] Grounding test bench passes >90% of cases
- [ ] No hardcoded object selection anywhere in pipeline
- [ ] Qwen correctly identifies source/target for various tasks
- [ ] Ambiguous cases logged and handled gracefully
- [ ] End-to-end execution without manual object specification
- [ ] Works on 5+ LIBERO tasks

### Estimated Time: 4-5 days (realistic)

---

## Future Phases (Post-MVP)

### Phase 4: Learned Geometric Perception

**Priority**: Required for real-world, optional for LIBERO demo.

**Key additions from feedback**:
- Data pipeline: collect rendered images + GT from Phases 1-3
- Clear metrics before training (keypoint error, mask IoU, pose error)
- Same API as oracle: `learned_perception()` returns same dict format
- Runtime/latency checks: must fit on GPU with other components
- Debugging visualizers: overlay masks/keypoints on images

### Phase 5: Visual Servo (B3)

**Priority**: Research-critical, not MVP-blocking.

- Start with simplified servo: `error_pixel → action`
- Full IBVS (Jacobian) only if needed
- Forward model for safety constraints

### Phase 6: Monitoring & Replanning

- Postcondition verification using perception
- Simple heuristic repairs first (re-approach, re-grasp)
- Qwen-based repair only for complex failures

---

## Key Design Principles

1. **Qwen does semantics, not geometry** - Never ask Qwen for pixel coordinates or distances in meters

2. **Oracle first, learned later** - Validate architecture with perfect perception before adding noise

3. **Skills are generic** - `GraspObject(id)` not `GraspBlueMug()` enables reuse

4. **Explicit state tracking** - Symbolic world model enables multi-step reasoning

5. **Validation at every layer** - Check preconditions, validate Qwen output, verify postconditions

6. **Log everything from day 1** - Episode traces, Qwen interactions, failures for future training

7. **Coordinate frames are sacred** - Establish and test frame consistency before writing skills

8. **Structured results** - Every skill returns `SkillResult`, not just bool

---

## Timeline Summary

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Oracle + Classical Skills + Logging + Tests | 5-7 days |
| 2 | Qwen Planning + Validation + Prompt Tracking | 4-5 days |
| 3 | Qwen Grounding + Test Bench | 4-5 days |
| **MVP Total** | **End-to-end zero-shot** | **~2-2.5 weeks** |
| 4 | Learned Perception | 1-2 weeks |
| 5 | Visual Servo | 1 week |
| 6 | Monitoring | 3-5 days |
