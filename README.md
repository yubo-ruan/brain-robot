# Brain-Inspired LLM-Guided Robot Control

A brain-inspired architecture for robot manipulation that combines Vision-Language Models (VLMs) for high-level planning with neuroscience-inspired action generation.

## Architecture

The system mimics the hierarchical organization of the human motor system:

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen2.5-VL-7B                            │
│                  (Prefrontal Cortex)                        │
│            High-level planning from vision                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ JSON motion plans
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Plan Encoder                             │
│                 (Sensory Processing)                        │
│         Converts symbolic plans to neural embeddings        │
└─────────────────────────┬───────────────────────────────────┘
                          │ 128-dim embeddings
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Primitive Selector (PMC)                       │
│                  (Premotor Cortex)                          │
│           Selects which motion primitives to use            │
└─────────────────────────┬───────────────────────────────────┘
                          │ primitive weights
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            Motion Primitive Library (CPGs)                  │
│              (Central Pattern Generators)                   │
│         Pre-learned trajectory templates for movements      │
└─────────────────────────┬───────────────────────────────────┘
                          │ blended actions
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Primitive Modulator (M1)                       │
│               (Primary Motor Cortex)                        │
│        Modulates amplitude, speed, and offset               │
└─────────────────────────┬───────────────────────────────────┘
                          │ modulated actions
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Forward Model (Cerebellum)                    │
│                    Error Correction                         │
│           Predicts outcomes and corrects errors             │
└─────────────────────────┬───────────────────────────────────┘
                          │ final actions
                          ▼
                    Robot Environment
```

## Components

### VLM Planner (`brain_robot/vlm/`)
- Uses Qwen2.5-VL-7B for visual understanding and motion planning
- Outputs structured JSON plans with:
  - Relative directions (left, right, forward, backward, up, down)
  - Speed levels (very_slow, slow, medium, fast)
  - Task phases (approach, align, descend, grasp, lift, move, place, release)
  - Gripper commands (open, close, maintain)

### Action Generator (`brain_robot/action_generator/`)
- **Plan Encoder**: Converts JSON plans to embeddings with explicit direction bypass
- **Primitive Selector**: Neural network trained to map directions to motion primitives
- **Motion Primitives (CPGs)**: 8 pre-defined trajectory templates for basic movements
- **Modulator**: Adjusts primitive amplitude and adds offsets based on context
- **Forward Model**: Predicts next state for error correction

### Training (`brain_robot/training/`)
- PPO-based reinforcement learning
- Reward shaping for direction following, speed matching, and task completion
- Pre-training for primitive selector with supervised learning

### Environment (`brain_robot/env/`)
- Mock environment for testing without LIBERO dependency
- Simulates pick-and-place tasks with visual observations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-robot.git
cd brain-robot

# Install dependencies
pip install torch torchvision transformers accelerate gymnasium numpy

# Download Qwen2.5-VL-7B model (optional, for full VLM support)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/qwen2.5-vl-7b')"
```

## Usage

### Pre-train the Primitive Selector

```bash
python scripts/pretrain_selector.py
```

This trains the selector to correctly map VLM direction commands to motion primitives.

### Run Training

```bash
python scripts/train_extended.py --episodes 100
```

### Debug and Test

```bash
# Test action generation
python scripts/test_pretrained.py

# Debug the full action chain
python scripts/debug_actions.py

# Test VLM planning
python scripts/test_vlm_detailed.py
```

## Project Structure

```
brain_robot/
├── brain_robot/
│   ├── action_generator/
│   │   ├── brain_model.py      # Main action generator
│   │   ├── plan_encoder.py     # JSON to embedding encoder
│   │   └── forward_model.py    # Cerebellum forward model
│   ├── vlm/
│   │   ├── qwen_planner.py     # VLM planner interface
│   │   └── prompts.py          # System prompts for VLM
│   ├── env/
│   │   └── mock_env.py         # Mock robot environment
│   └── training/
│       └── rewards.py          # Reward shaping
├── scripts/
│   ├── train_extended.py       # Main training script
│   ├── pretrain_selector.py    # Primitive selector pre-training
│   ├── test_pretrained.py      # Test action generation
│   └── debug_actions.py        # Debug utilities
├── checkpoints/                # Saved model checkpoints
└── models/                     # Downloaded VLM weights
```

## Key Features

- **Brain-Inspired Architecture**: Mimics the hierarchical motor control system
- **VLM Integration**: Uses state-of-the-art vision-language model for planning
- **Interpretable Primitives**: Clear separation of motion primitives for debugging
- **Action Chunking**: Generates 10-step action sequences for temporal consistency
- **Error Correction**: Cerebellar-inspired forward model for adaptive control

## LIBERO Benchmark Integration

This project integrates with the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark for realistic robot manipulation evaluation.

### Supported Task Suites
- `libero_spatial` - 10 tasks with varying spatial configurations
- `libero_object` - 10 tasks with different objects
- `libero_goal` - 10 tasks with different goals
- `libero_90` - 90 diverse manipulation tasks
- `libero_10` - 10 long-horizon tasks

### Running LIBERO Experiments

```bash
# Compare Simple MLP vs Brain-Inspired approaches
MUJOCO_GL=egl python scripts/compare_approaches_libero.py --n_demos 50 --n_eval 5

# This generates:
# - recordings/libero_comparison/simple_mlp.gif
# - recordings/libero_comparison/brain_inspired.gif
# - recordings/libero_comparison/comparison.gif (side-by-side)
```

### GIF Recordings with VLM Visualization

The comparison script generates GIF recordings that include:
- **Environment view**: Robot camera feed
- **VLM phase annotation**: Current task phase (approach, grasp, lift, transport, place, release)
- **Confidence score**: VLM planning confidence
- **Primitive weights**: For brain-inspired policy, shows which primitives are active
- **Phase progress bar**: Color-coded progress indicator

---

## Experiment Results Summary

### Phase A-E Ablation Study Findings

| Experiment | Key Finding |
|------------|-------------|
| **Phase A: PPO Fix** | Proper PPO achieves 99% on simple reaching, but struggles with multi-phase tasks |
| **Phase B: No VLM** | Behavioral Cloning achieves **100%** success, outperforming PPO |
| **Phase C: Distillation** | Distilled MLP achieves **99% @ 4612 FPS** (4600x speedup over VLM) |
| **Phase D: Phase Conditioning** | Marginal benefit - baseline already 99% |
| **Phase E: Forward Model** | **Hurts performance!** Best to remove entirely |

### Optimal Architecture

Based on extensive ablation:

```
State (9-15 dims) → MLP (128 hidden) → Action (7 dims)
```

**Key insights:**
1. Simple MLP outperforms brain-inspired architecture
2. Behavioral Cloning > PPO for multi-phase manipulation
3. Remove VLM from inner loop (use distillation)
4. Remove forward model (degrades performance)
5. Remove motion primitives (unnecessary complexity)

### LIBERO Results

| Approach | Success Rate | FPS |
|----------|-------------|-----|
| Simple MLP (BC) | Testing | ~40 |
| Brain-Inspired (BC) | Testing | ~40 |

*Note: LIBERO tasks are significantly more challenging than mock environment tasks.*

---

## Visualization Tools

### Record Task Execution
```bash
# Basic recording
python scripts/record_task.py

# Detailed recording with trajectory plot
python scripts/record_detailed.py
```

### Output Files
- `recordings/libero_comparison/` - LIBERO comparison GIFs
- `recordings/detailed_episode.gif` - Detailed mock env recording
- `recordings/trajectory_plot.png` - 3D trajectory visualization

---

## Project Structure

```
brain_robot/
├── brain_robot/
│   ├── action_generator/
│   │   ├── brain_model.py      # Main action generator
│   │   ├── plan_encoder.py     # JSON to embedding encoder
│   │   └── forward_model.py    # Cerebellum forward model
│   ├── vlm/
│   │   ├── qwen_planner.py     # VLM planner interface
│   │   └── prompts.py          # System prompts for VLM
│   ├── env/
│   │   ├── mock_env.py         # Mock robot environment
│   │   └── libero_wrapper.py   # LIBERO benchmark wrapper
│   └── training/
│       └── rewards.py          # Reward shaping
├── scripts/
│   ├── train_extended.py       # Main training script
│   ├── compare_approaches_libero.py  # LIBERO comparison
│   ├── record_task.py          # Task recording
│   ├── record_detailed.py      # Detailed recording
│   └── ...                     # Other training scripts
├── recordings/                 # Generated GIFs and plots
├── checkpoints/                # Saved model checkpoints
└── EXPERIMENTS_SUMMARY.md      # Detailed experiment results
```

## Key Features

- **Brain-Inspired Architecture**: Mimics the hierarchical motor control system
- **VLM Integration**: Uses state-of-the-art vision-language model for planning
- **LIBERO Benchmark**: Integration with realistic manipulation tasks
- **Visualization Tools**: GIF recordings with VLM output overlay
- **Ablation Study**: Comprehensive experiments comparing approaches
- **Interpretable Primitives**: Clear separation of motion primitives for debugging

## License

MIT License
