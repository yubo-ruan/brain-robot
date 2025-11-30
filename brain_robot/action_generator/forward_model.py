"""
Cerebellum Forward Model.
Predicts sensory consequences of actions (like the cerebellum).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CerebellumForwardModel(nn.Module):
    """
    Forward model that predicts next proprioceptive state.

    Inspired by cerebellar function:
    - Receives efference copy (motor command)
    - Receives current state (proprioception)
    - Predicts sensory consequences
    - Enables anticipatory control and smooth movements
    """

    def __init__(
        self,
        proprio_dim: int = 15,
        action_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        # Input: current proprio + action
        input_dim = proprio_dim + action_dim

        # Build network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, proprio_dim))

        self.network = nn.Sequential(*layers)

        # Residual connection (predict delta, not absolute)
        self.use_residual = True

    def forward(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next proprioceptive state.

        Args:
            proprio: (B, proprio_dim) current proprioception
            action: (B, action_dim) action to execute

        Returns:
            next_proprio: (B, proprio_dim) predicted next state
        """
        x = torch.cat([proprio, action], dim=-1)

        if self.use_residual:
            # Predict delta
            delta = self.network(x)
            next_proprio = proprio + delta
        else:
            # Predict absolute
            next_proprio = self.network(x)

        return next_proprio

    def compute_loss(
        self,
        proprio_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction loss for training the forward model.

        Args:
            proprio_seq: (B, T, proprio_dim) sequence of proprioception
            action_seq: (B, T, action_dim) sequence of actions

        Returns:
            loss: Scalar prediction loss
        """
        B, T, _ = proprio_seq.shape

        losses = []
        for t in range(T - 1):
            proprio_t = proprio_seq[:, t]
            action_t = action_seq[:, t]
            proprio_next_actual = proprio_seq[:, t + 1]

            proprio_next_pred = self.forward(proprio_t, action_t)

            loss = F.mse_loss(proprio_next_pred, proprio_next_actual)
            losses.append(loss)

        return torch.stack(losses).mean()
