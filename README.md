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

## Results

After pre-training the primitive selector:
- 100% accuracy on direction-to-primitive mapping
- Robot successfully moves toward target objects
- VLM correctly identifies task phases and generates appropriate plans

## License

MIT License
