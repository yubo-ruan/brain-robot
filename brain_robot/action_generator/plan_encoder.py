"""
Plan Encoder: Converts JSON motion plans to embeddings.
Like sensory processing in the brain - converts symbolic info to neural representation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any


class RelativePlanEncoder(nn.Module):
    """
    Encodes relative motion commands (JSON) into embeddings.
    Uses separate embeddings for each component (like different sensory streams).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        max_movements: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_movements = max_movements

        # Direction embedding (8 directions + pad)
        self.direction_vocab = {
            'left': 0, 'right': 1, 'forward': 2, 'backward': 3,
            'up': 4, 'down': 5, 'rotate_left': 6, 'rotate_right': 7, 'pad': 8
        }
        self.direction_embed = nn.Embedding(9, 32)

        # Speed embedding (4 speeds + pad)
        self.speed_vocab = {
            'very_slow': 0, 'slow': 1, 'medium': 2, 'fast': 3, 'pad': 4
        }
        self.speed_embed = nn.Embedding(5, 16)

        # Steps embedding (0-5)
        self.steps_embed = nn.Embedding(6, 16)

        # Phase embedding (8 phases)
        self.phase_vocab = {
            'approach': 0, 'align': 1, 'descend': 2, 'grasp': 3,
            'lift': 4, 'move': 5, 'place': 6, 'release': 7
        }
        self.phase_embed = nn.Embedding(8, 32)

        # Gripper embedding (open, close, maintain)
        self.gripper_vocab = {'open': 0, 'close': 1, 'maintain': 2}
        self.gripper_embed = nn.Embedding(3, 16)

        # Distance embedding (far, medium, close, touching)
        self.distance_vocab = {'far': 0, 'medium': 1, 'close': 2, 'touching': 3}
        self.distance_embed = nn.Embedding(4, 16)

        # Movement encoder
        self.movement_encoder = nn.Sequential(
            nn.Linear(32 + 16 + 16, hidden_dim),
            nn.ReLU(),
        )

        # Aggregate multiple movements
        self.movement_aggregator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Final projection
        # movement_hidden + phase + gripper + distance + direction_explicit
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 16 + 16 + 32, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Explicit direction projection (bypass path for clear direction signal)
        self.direction_proj = nn.Linear(32, 32)

    def forward(self, plans: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode batch of plans.

        Args:
            plans: List of plan dictionaries

        Returns:
            embeddings: (B, embed_dim)
        """
        device = self.direction_embed.weight.device
        batch_embeds = []

        for plan in plans:
            embed = self._encode_single(plan, device)
            batch_embeds.append(embed)

        return torch.stack(batch_embeds)

    def _encode_single(self, plan: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Encode a single plan dictionary."""

        plan_data = plan.get('plan', {})
        obs_data = plan.get('observation', {})

        # Get fields with defaults
        movements = plan_data.get('movements', [])
        phase = plan_data.get('phase', 'approach')
        gripper = plan_data.get('gripper', 'maintain')
        distance = obs_data.get('distance_to_target', 'far')

        # Encode movements
        movement_embeds = []
        for i in range(self.max_movements):
            if i < len(movements):
                m = movements[i]
                dir_idx = self.direction_vocab.get(m.get('direction', 'forward'), 2)
                speed_idx = self.speed_vocab.get(m.get('speed', 'medium'), 2)
                steps_idx = min(m.get('steps', 1), 5)
            else:
                # Padding
                dir_idx, speed_idx, steps_idx = 8, 4, 0

            dir_emb = self.direction_embed(torch.tensor(dir_idx, device=device))
            speed_emb = self.speed_embed(torch.tensor(speed_idx, device=device))
            steps_emb = self.steps_embed(torch.tensor(steps_idx, device=device))

            m_concat = torch.cat([dir_emb, speed_emb, steps_emb])
            m_emb = self.movement_encoder(m_concat)
            movement_embeds.append(m_emb)

        # Aggregate movements with GRU
        movement_seq = torch.stack(movement_embeds).unsqueeze(0)  # (1, max_mov, hidden)
        _, movement_hidden = self.movement_aggregator(movement_seq)
        movement_final = movement_hidden.squeeze(0).squeeze(0)  # (hidden,)

        # Encode other components
        phase_emb = self.phase_embed(
            torch.tensor(self.phase_vocab.get(phase, 0), device=device)
        )
        gripper_emb = self.gripper_embed(
            torch.tensor(self.gripper_vocab.get(gripper, 2), device=device)
        )
        distance_emb = self.distance_embed(
            torch.tensor(self.distance_vocab.get(distance, 0), device=device)
        )

        # Get primary direction (first movement's direction) for explicit bypass
        if movements and len(movements) > 0:
            primary_dir = movements[0].get('direction', 'forward')
            primary_dir_idx = self.direction_vocab.get(primary_dir, 2)
        else:
            primary_dir_idx = 8  # pad
        primary_dir_emb = self.direction_embed(
            torch.tensor(primary_dir_idx, device=device)
        )
        # Add explicit direction signal with learned projection
        direction_explicit = self.direction_proj(primary_dir_emb)

        # Combine all components including explicit direction bypass
        combined = torch.cat([movement_final, phase_emb, gripper_emb, distance_emb, direction_explicit])

        return self.output_proj(combined)
