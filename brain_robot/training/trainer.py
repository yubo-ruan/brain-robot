"""
Main Training Loop.
Combines VLM planner, action generator, and RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import os
from tqdm import tqdm

from ..vlm.qwen_planner import QwenVLPlanner
from ..action_generator.brain_model import BrainInspiredActionGenerator
from ..env import make_libero_env, make_mock_env, LIBERO_AVAILABLE
from .rewards import RewardShaper

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BrainRobotTrainer:
    """
    Trainer for brain-inspired robot control system.

    Training phases:
    1. Forward model pretraining (cerebellum learning)
    2. RL training with VLM guidance (motor learning)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # Create environment
        env_config = config['env']
        if LIBERO_AVAILABLE:
            print("Using LIBERO environment")
            self.env = make_libero_env(
                task_suite=env_config['task_suite'],
                task_id=env_config['task_ids'][0],  # Start with first task
                max_episode_steps=env_config['max_episode_steps'],
                action_scale=env_config['action_scale'],
            )
        else:
            print("LIBERO not available, using mock environment")
            self.env = make_mock_env(
                max_episode_steps=env_config['max_episode_steps'],
                action_scale=env_config['action_scale'],
            )
        self.task_description = self.env.task_description

        # Create VLM planner
        vlm_config = config['model']['vlm']
        print("Loading VLM planner...")
        self.planner = QwenVLPlanner(
            model_name=vlm_config['model_name'],
            device=device,
            max_new_tokens=vlm_config['max_new_tokens'],
            temperature=vlm_config['temperature'],
        )
        self.replan_every = vlm_config['replan_every']

        # Create action generator
        ag_config = config['model']['action_generator']
        self.action_generator = BrainInspiredActionGenerator(
            plan_dim=ag_config['plan_dim'],
            proprio_dim=ag_config['proprio_dim'],
            action_dim=ag_config['action_dim'],
            chunk_size=ag_config['chunk_size'],
            num_primitives=ag_config['num_primitives'],
            hidden_dim=ag_config['hidden_dim'],
        ).to(device)

        # Create critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(ag_config['plan_dim'] + ag_config['proprio_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        # Create reward shaper
        reward_config = config['reward']
        self.reward_shaper = RewardShaper(
            task_success_reward=reward_config['task_success'],
            direction_reward_scale=reward_config['direction_following'],
            speed_reward_scale=reward_config['speed_matching'],
            gripper_reward_scale=reward_config['gripper_consistency'],
            forward_model_bonus=reward_config['forward_model_bonus'],
            time_penalty=reward_config['time_penalty'],
        )

        # Optimizers
        train_config = config['training']
        self.actor_optimizer = torch.optim.Adam(
            self.action_generator.parameters(),
            lr=train_config['lr_actor'],
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=train_config['lr_critic'],
        )
        self.fm_optimizer = torch.optim.Adam(
            self.action_generator.cerebellum.parameters(),
            lr=train_config['lr_forward_model'],
        )

        # PPO hyperparameters
        self.gamma = train_config['gamma']
        self.gae_lambda = train_config['gae_lambda']
        self.clip_epsilon = train_config['clip_epsilon']
        self.value_coef = train_config['value_coef']
        self.entropy_coef = train_config['entropy_coef']
        self.max_grad_norm = train_config['max_grad_norm']
        self.ppo_epochs = train_config['ppo_epochs']

        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Logging
        if WANDB_AVAILABLE and config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project'],
                entity=config['logging']['entity'],
                config=config,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def pretrain_forward_model(self, num_episodes: int = 100):
        """
        Phase 1: Pretrain forward model (cerebellum).
        Collect random trajectories and learn state prediction.
        """
        print("\n" + "="*60)
        print("Phase 1: Pretraining Forward Model (Cerebellum)")
        print("="*60)

        all_proprios = []
        all_actions = []

        # Collect random trajectories
        for ep in tqdm(range(num_episodes), desc="Collecting trajectories"):
            obs, _ = self.env.reset()

            ep_proprios = [obs['proprio']]
            ep_actions = []

            for step in range(self.env.max_episode_steps):
                # Random action
                action = self.env.action_space.sample()

                next_obs, _, done, truncated, _ = self.env.step(action)

                ep_proprios.append(next_obs['proprio'])
                ep_actions.append(action)

                obs = next_obs
                if done or truncated:
                    break

            if len(ep_actions) > 1:
                all_proprios.append(np.array(ep_proprios[:-1]))
                all_actions.append(np.array(ep_actions))

        # Train forward model
        proprio_tensor = torch.tensor(
            np.concatenate(all_proprios), dtype=torch.float32, device=self.device
        )
        action_tensor = torch.tensor(
            np.concatenate(all_actions), dtype=torch.float32, device=self.device
        )

        # Simple training loop
        fm_config = self.config['model']['forward_model']
        for epoch in range(fm_config['pretrain_epochs']):
            # Shuffle
            perm = torch.randperm(len(proprio_tensor) - 1)

            losses = []
            for i in range(0, len(perm), 256):
                idx = perm[i:i+256]

                proprio = proprio_tensor[idx]
                action = action_tensor[idx]
                next_proprio = proprio_tensor[idx + 1]

                pred_next = self.action_generator.cerebellum(proprio, action)
                loss = F.mse_loss(pred_next, next_proprio)

                self.fm_optimizer.zero_grad()
                loss.backward()
                self.fm_optimizer.step()

                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f"Forward Model Epoch {epoch+1}: Loss = {avg_loss:.6f}")

            if self.use_wandb:
                wandb.log({'forward_model_loss': avg_loss})

        print("Forward model pretraining complete!")

    def collect_episode(self) -> Tuple[List[Dict[str, Any]], bool]:
        """Collect one episode of experience."""
        obs, info = self.env.reset()
        self.planner.reset()

        episode_data = []
        current_plan = None
        steps_since_plan = 0
        total_reward = 0
        action_chunk = None
        action_idx = 0
        predicted_proprio = None

        for step in range(self.env.max_episode_steps):
            # Replan if needed
            if current_plan is None or steps_since_plan >= self.replan_every:
                gripper_state = "open" if obs['gripper_state'] == 0 else "closed"

                current_plan = self.planner.plan(
                    image=obs['image'],
                    task_description=self.task_description,
                    gripper_state=gripper_state,
                    steps_since_plan=steps_since_plan,
                )

                steps_since_plan = 0
                action_idx = 0

                # Generate action chunk
                proprio = torch.tensor(
                    obs['proprio'], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action_chunk, components = self.action_generator(
                        [current_plan], proprio, return_components=True
                    )
                    action_chunk = action_chunk.squeeze(0).cpu().numpy()
                    predicted_proprio = components['predicted_next_proprio'].cpu().numpy()[0]

            # Get action from chunk
            if action_chunk is not None and action_idx < len(action_chunk):
                action = action_chunk[action_idx]
            elif action_chunk is not None:
                action = action_chunk[-1]
            else:
                action = np.zeros(7)

            # Execute action
            next_obs, env_reward, done, truncated, info = self.env.step(action)

            # Compute shaped reward
            reward = self.reward_shaper.compute_reward(
                obs=obs,
                action=action,
                next_obs=next_obs,
                plan=current_plan,
                info=info,
                predicted_proprio=predicted_proprio if action_idx == 0 else None,
            )

            # Store transition
            episode_data.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done or truncated,
                'plan': current_plan,
                'proprio': obs['proprio'],
            })

            total_reward += reward
            obs = next_obs
            steps_since_plan += 1
            action_idx += 1

            if done or truncated:
                break

        # Record statistics
        success = info.get('success', False)
        self.episode_rewards.append(total_reward)
        self.success_rate.append(float(success))
        self.episode_lengths.append(len(episode_data))

        return episode_data, success

    def train_step(self, episodes_data: List[List[Dict]]) -> Dict[str, float]:
        """Perform PPO update."""
        # Flatten episodes
        all_plans = []
        all_proprios = []
        all_actions = []
        all_rewards = []
        all_dones = []

        for episode in episodes_data:
            for step in episode:
                all_plans.append(step['plan'])
                all_proprios.append(step['proprio'])
                all_actions.append(step['action'])
                all_rewards.append(step['reward'])
                all_dones.append(step['done'])

        # Convert to tensors
        proprios = torch.tensor(
            np.array(all_proprios), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(all_actions), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            all_rewards, dtype=torch.float32, device=self.device
        )

        # Compute values and advantages
        with torch.no_grad():
            plan_embeds = self.action_generator.plan_encoder(all_plans).to(self.device)
            values = self.critic(
                torch.cat([plan_embeds, proprios], dim=-1)
            ).squeeze(-1)

        # Simple advantage estimation
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = rewards  # Simplified - should use GAE

        # PPO update
        policy_losses = []
        value_losses = []

        for _ in range(self.ppo_epochs):
            # Get current predictions
            pred_actions = self.action_generator(all_plans, proprios)[:, 0, :]

            plan_embeds = self.action_generator.plan_encoder(all_plans).to(self.device)
            values_new = self.critic(
                torch.cat([plan_embeds, proprios], dim=-1)
            ).squeeze(-1)

            # Policy loss (simplified MSE weighted by advantage)
            action_diff = (pred_actions - actions).pow(2).mean(dim=-1)
            policy_loss = (action_diff * advantages.detach()).mean()

            # Value loss
            value_loss = F.mse_loss(values_new, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.action_generator.parameters(), self.max_grad_norm
            )
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.critic_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }

    def train(self):
        """Main training loop."""
        config = self.config['training']

        # Phase 1: Pretrain forward model
        self.pretrain_forward_model(num_episodes=100)

        # Phase 2: RL training
        print("\n" + "="*60)
        print("Phase 2: RL Training with VLM Guidance")
        print("="*60)

        num_episodes = config['num_episodes']
        episodes_per_update = config['episodes_per_update']

        for episode in range(0, num_episodes, episodes_per_update):
            # Collect episodes
            episodes_data = []
            successes = 0

            for _ in range(episodes_per_update):
                ep_data, success = self.collect_episode()
                episodes_data.append(ep_data)
                successes += int(success)

            # Train
            losses = self.train_step(episodes_data)

            # Log
            avg_reward = np.mean(list(self.episode_rewards))
            avg_success = np.mean(list(self.success_rate)) * 100
            avg_length = np.mean(list(self.episode_lengths))

            print(f"\nEpisode {episode + episodes_per_update}/{num_episodes}")
            print(f"  Policy Loss: {losses['policy_loss']:.4f}")
            print(f"  Value Loss: {losses['value_loss']:.4f}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {avg_success:.1f}%")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  This batch: {successes}/{episodes_per_update} successes")

            if self.use_wandb:
                wandb.log({
                    'episode': episode + episodes_per_update,
                    'policy_loss': losses['policy_loss'],
                    'value_loss': losses['value_loss'],
                    'avg_reward': avg_reward,
                    'success_rate': avg_success,
                    'avg_episode_length': avg_length,
                    'batch_successes': successes,
                })

            # Save checkpoint
            if (episode + episodes_per_update) % config['save_every'] == 0:
                self.save_checkpoint(episode + episodes_per_update)

    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        path = f"{checkpoint_dir}/checkpoint_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'action_generator': self.action_generator.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.action_generator.load_state_dict(checkpoint['action_generator'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        print(f"Loaded checkpoint from: {path}")
        return checkpoint['episode']
