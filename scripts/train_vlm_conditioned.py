#!/usr/bin/env python3
"""
Train brain-inspired policy with VLM-generated plans.

This script trains the policy using:
1. Pre-computed VLM plans from demonstrations
2. Behavioral cloning loss on actions

The VLM provides high-level guidance, while the policy learns
low-level motor control.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import argparse
from typing import Dict, List, Any

from brain_robot.policy.vlm_policy import VLMConditionedPolicySimple
from brain_robot.action_generator.brain_model import BrainInspiredActionGenerator


class VLMPlanDataset(Dataset):
    """Dataset of (state, action, vlm_plan) tuples."""

    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.states = torch.tensor(data['states'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)
        self.vlm_plans = data['vlm_plans']

        print(f"Loaded {len(self.states)} samples")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'vlm_plan': self.vlm_plans[idx],
        }


def collate_fn(batch):
    """Custom collate function to handle VLM plans."""
    states = torch.stack([b['state'] for b in batch])
    actions = torch.stack([b['action'] for b in batch])
    vlm_plans = [b['vlm_plan'] for b in batch]
    return {
        'states': states,
        'actions': actions,
        'vlm_plans': vlm_plans,
    }


def train_vlm_conditioned_policy(
    policy: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    log_interval: int = 10,
):
    """
    Train policy with behavioral cloning on VLM-conditioned data.

    Args:
        policy: VLM-conditioned policy network
        train_loader: DataLoader with (state, action, vlm_plan) samples
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        log_interval: Log every N batches
    """
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float('inf')
    losses = []

    for epoch in range(n_epochs):
        policy.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch_idx, batch in enumerate(pbar):
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            vlm_plans = batch['vlm_plans']

            # Forward pass
            pred_actions = policy(vlm_plans, states)

            # BC loss
            loss = F.mse_loss(pred_actions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'checkpoints/vlm_conditioned_best.pt')

    return losses


def train_brain_inspired_policy(
    policy: BrainInspiredActionGenerator,
    train_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    Train the full brain-inspired policy with VLM plans.
    """
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(n_epochs):
        policy.train()
        epoch_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            vlm_plans = batch['vlm_plans']

            # Forward pass - brain policy takes full action chunk
            # We compare first action of chunk with target
            pred_actions = policy(vlm_plans, states)  # (B, chunk, action_dim)
            pred_first = pred_actions[:, 0, :]  # (B, action_dim)

            # BC loss
            loss = F.mse_loss(pred_first, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/n_batches:.4f}")

    return policy


def evaluate_policy(
    policy: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
):
    """Evaluate policy on test set."""
    policy.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            vlm_plans = batch['vlm_plans']

            if isinstance(policy, BrainInspiredActionGenerator):
                pred_actions = policy(vlm_plans, states)[:, 0, :]
            else:
                pred_actions = policy(vlm_plans, states)

            loss = F.mse_loss(pred_actions, actions)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to pre-computed VLM plans")
    parser.add_argument("--policy_type", type=str, default="simple",
                        choices=["simple", "brain_inspired"])
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/brain_robot/checkpoints")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = VLMPlanDataset(args.data_path)

    # Split train/test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create policy
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    if args.policy_type == "simple":
        policy = VLMConditionedPolicySimple(
            plan_dim=128,
            proprio_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
        )
    else:
        policy = BrainInspiredActionGenerator(
            plan_dim=128,
            proprio_dim=state_dim,
            action_dim=action_dim,
            chunk_size=10,
        )

    print(f"Policy type: {args.policy_type}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training samples: {train_size}, Test samples: {test_size}")

    # Train
    if args.policy_type == "simple":
        losses = train_vlm_conditioned_policy(
            policy,
            train_loader,
            n_epochs=args.n_epochs,
            lr=args.lr,
            device=args.device,
        )
    else:
        train_brain_inspired_policy(
            policy,
            train_loader,
            n_epochs=args.n_epochs,
            lr=args.lr,
            device=args.device,
        )

    # Evaluate
    test_loss = evaluate_policy(policy, test_loader, args.device)
    print(f"\nTest Loss: {test_loss:.4f}")

    # Save final model
    save_path = os.path.join(args.output_dir, f"vlm_{args.policy_type}_final.pt")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'policy_type': args.policy_type,
        'state_dim': state_dim,
        'action_dim': action_dim,
    }, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
