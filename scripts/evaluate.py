#!/usr/bin/env python3
"""
Evaluation script for Brain-Inspired Robot Control.
"""

import argparse
import yaml
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vlm.qwen_planner import QwenVLPlanner
from src.action_generator.brain_model import BrainInspiredActionGenerator
from src.env.libero_wrapper import make_libero_env


def evaluate(
    config: dict,
    checkpoint_path: str,
    num_episodes: int = 10,
    device: str = "cuda",
    render: bool = False,
):
    """
    Evaluate trained model.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        device: Device to use
        render: Whether to render episodes
    """
    # Create environment
    env_config = config['env']
    env = make_libero_env(
        task_suite=env_config['task_suite'],
        task_id=env_config['task_ids'][0],
        max_episode_steps=env_config['max_episode_steps'],
        action_scale=env_config['action_scale'],
    )
    task_description = env.task_description

    # Create VLM planner
    vlm_config = config['model']['vlm']
    planner = QwenVLPlanner(
        model_name=vlm_config['model_name'],
        device=device,
        max_new_tokens=vlm_config['max_new_tokens'],
        temperature=vlm_config['temperature'],
    )

    # Create action generator
    ag_config = config['model']['action_generator']
    action_generator = BrainInspiredActionGenerator(
        plan_dim=ag_config['plan_dim'],
        proprio_dim=ag_config['proprio_dim'],
        action_dim=ag_config['action_dim'],
        chunk_size=ag_config['chunk_size'],
        num_primitives=ag_config['num_primitives'],
        hidden_dim=ag_config['hidden_dim'],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    action_generator.load_state_dict(checkpoint['action_generator'])
    action_generator.eval()

    # Evaluate
    successes = []
    rewards = []
    lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        planner.reset()

        episode_reward = 0
        current_plan = None
        steps_since_plan = 0
        replan_every = vlm_config['replan_every']

        for step in range(env.max_episode_steps):
            # Replan if needed
            if current_plan is None or steps_since_plan >= replan_every:
                gripper_state = "open" if obs['gripper_state'] == 0 else "closed"

                current_plan = planner.plan(
                    image=obs['image'],
                    task_description=task_description,
                    gripper_state=gripper_state,
                    steps_since_plan=steps_since_plan,
                )

                steps_since_plan = 0
                action_idx = 0

                # Generate action chunk
                proprio = torch.tensor(
                    obs['proprio'], dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action_chunk = action_generator([current_plan], proprio)
                    action_chunk = action_chunk.squeeze(0).cpu().numpy()

            # Get action from chunk
            if action_idx < len(action_chunk):
                action = action_chunk[action_idx]
            else:
                action = action_chunk[-1]

            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            steps_since_plan += 1
            action_idx += 1

            if done or truncated:
                break

        success = info.get('success', False)
        successes.append(success)
        rewards.append(episode_reward)
        lengths.append(step + 1)

        print(f"Episode {ep + 1}/{num_episodes}: "
              f"Success={success}, Reward={episode_reward:.2f}, Length={step + 1}")

    # Summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"Success Rate: {np.mean(successes) * 100:.1f}%")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate brain-inspired robot control")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Evaluate
    evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        device=args.device,
        render=args.render,
    )


if __name__ == "__main__":
    main()
