"""
Brain-Inspired Action Generator.

Architecture inspired by motor neuroscience:
1. Motion Primitives (Central Pattern Generators - CPGs)
2. Primitive Selector (Premotor Cortex - PMC)
3. Primitive Modulator (Primary Motor Cortex - M1)
4. Forward Model (Cerebellum)
5. Error Correction (Cerebellar Feedback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from .plan_encoder import RelativePlanEncoder
from .forward_model import CerebellumForwardModel


class MotionPrimitiveLibrary(nn.Module):
    """
    Central Pattern Generators (CPGs).
    Pre-learned trajectory templates for basic movements.
    """

    def __init__(
        self,
        num_primitives: int = 8,
        chunk_size: int = 10,
        action_dim: int = 7,
    ):
        super().__init__()

        self.num_primitives = num_primitives
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Learnable primitives (initialized with structure)
        self.primitives = nn.Parameter(
            self._initialize_primitives()
        )

        # Primitive names for interpretability
        self.primitive_names = [
            'move_left', 'move_right', 'move_forward', 'move_backward',
            'move_up', 'move_down', 'open_gripper', 'close_gripper'
        ]

    def _initialize_primitives(self) -> torch.Tensor:
        """Initialize primitives with meaningful structure."""
        primitives = torch.zeros(self.num_primitives, self.chunk_size, self.action_dim)

        # Action dimensions: [dx, dy, dz, rx, ry, rz, gripper]

        # move_left: negative x
        primitives[0, :, 0] = -0.3

        # move_right: positive x
        primitives[1, :, 0] = 0.3

        # move_forward: positive y
        primitives[2, :, 1] = 0.3

        # move_backward: negative y
        primitives[3, :, 1] = -0.3

        # move_up: positive z
        primitives[4, :, 2] = 0.3

        # move_down: negative z
        primitives[5, :, 2] = -0.3

        # open_gripper: gripper = -1
        primitives[6, :, 6] = -1.0

        # close_gripper: gripper = +1
        primitives[7, :, 6] = 1.0

        # Add small noise for symmetry breaking
        primitives += torch.randn_like(primitives) * 0.01

        return primitives

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Blend primitives according to weights.

        Args:
            weights: (B, num_primitives) soft selection weights

        Returns:
            blended: (B, chunk_size, action_dim)
        """
        # Weighted sum of primitives
        blended = torch.einsum('bp,pca->bca', weights, self.primitives)
        return blended

    def get_primitive(self, idx: int) -> torch.Tensor:
        """Get a specific primitive by index."""
        return self.primitives[idx]


class PrimitiveSelector(nn.Module):
    """
    Premotor Cortex (PMC).
    Selects which primitive(s) to activate based on the plan.
    """

    def __init__(
        self,
        plan_dim: int = 128,
        num_primitives: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.selector = nn.Sequential(
            nn.Linear(plan_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives),
        )

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, plan_embed: torch.Tensor) -> torch.Tensor:
        """
        Select primitives based on plan embedding.

        Args:
            plan_embed: (B, plan_dim)

        Returns:
            weights: (B, num_primitives) soft selection weights
        """
        logits = self.selector(plan_embed)
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        return weights


class PrimitiveModulator(nn.Module):
    """
    Primary Motor Cortex (M1) - Population Coding.
    Modulates the amplitude, speed, and offset of selected primitives.
    """

    def __init__(
        self,
        plan_dim: int = 128,
        proprio_dim: int = 15,
        action_dim: int = 7,
        hidden_dim: int = 128,
    ):
        super().__init__()

        input_dim = plan_dim + proprio_dim

        # Shared feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Modulation heads (like different neuron populations)
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),  # 0-1, will be scaled to 0-2
        )

        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 0-1, will be scaled to 0.5-2.0
        )

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # -1 to 1
        )

    def forward(
        self,
        plan_embed: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute modulation parameters.

        Args:
            plan_embed: (B, plan_dim)
            proprio: (B, proprio_dim)

        Returns:
            amplitude: (B, action_dim) scale factors 0-2
            speed: (B, 1) time scaling 0.5-2.0
            offset: (B, action_dim) baseline offset -0.5 to 0.5
        """
        x = torch.cat([plan_embed, proprio], dim=-1)
        features = self.encoder(x)

        amplitude = self.amplitude_head(features) * 2.0  # Scale to 0-2
        speed = self.speed_head(features) * 1.5 + 0.5  # Scale to 0.5-2.0
        offset = self.offset_head(features) * 0.5  # Scale to -0.5-0.5

        return amplitude, speed, offset


class BrainInspiredActionGenerator(nn.Module):
    """
    Complete brain-inspired action generator.

    Combines:
    - Motion Primitives (CPGs)
    - Primitive Selector (Premotor Cortex)
    - Primitive Modulator (M1)
    - Forward Model (Cerebellum)
    - Error Correction
    """

    def __init__(
        self,
        plan_dim: int = 128,
        proprio_dim: int = 15,
        action_dim: int = 7,
        chunk_size: int = 10,
        num_primitives: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.plan_dim = plan_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Plan Encoder (sensory processing)
        self.plan_encoder = RelativePlanEncoder(
            embed_dim=plan_dim,
            max_movements=5,
            hidden_dim=64,
        )

        # Motion Primitives (CPGs)
        self.primitives = MotionPrimitiveLibrary(
            num_primitives=num_primitives,
            chunk_size=chunk_size,
            action_dim=action_dim,
        )

        # Primitive Selector (Premotor Cortex)
        self.selector = PrimitiveSelector(
            plan_dim=plan_dim,
            num_primitives=num_primitives,
            hidden_dim=hidden_dim // 2,
        )

        # Primitive Modulator (M1)
        self.modulator = PrimitiveModulator(
            plan_dim=plan_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Forward Model (Cerebellum)
        self.cerebellum = CerebellumForwardModel(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Error Correction (Cerebellar Feedback)
        self.error_corrector = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        # Learnable correction gain
        self.correction_gain = nn.Parameter(torch.tensor(0.1))

        # Temporal smoother (like spinal cord reflexes)
        # Initialize as identity + small smoothing
        self.smoother = nn.Conv1d(
            action_dim, action_dim,
            kernel_size=3, padding=1, groups=action_dim
        )
        # Initialize to mostly pass-through with slight smoothing
        with torch.no_grad():
            self.smoother.weight.zero_()
            self.smoother.weight[:, :, 1] = 1.0  # Center weight = 1 (identity)
            self.smoother.bias.zero_()

    def forward(
        self,
        plans: List[Dict[str, Any]],
        proprio: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Generate action chunk from plan and proprioception.

        Args:
            plans: List of plan dictionaries from VLM
            proprio: (B, proprio_dim) robot proprioception

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = len(plans)
        device = proprio.device

        # 1. Encode plans (sensory processing)
        plan_embed = self.plan_encoder(plans)  # (B, plan_dim)
        plan_embed = plan_embed.to(device)

        # 2. Select primitives (Premotor Cortex)
        primitive_weights = self.selector(plan_embed)  # (B, num_primitives)

        # 3. Blend primitives (CPG activation)
        blended = self.primitives(primitive_weights)  # (B, chunk, action)

        # 4. Modulate (M1 population coding)
        amplitude, speed, offset = self.modulator(plan_embed, proprio)

        # Apply amplitude modulation
        modulated = blended * amplitude.unsqueeze(1)

        # Apply offset
        modulated = modulated + offset.unsqueeze(1)

        # 5. Forward model prediction (Cerebellum)
        first_action = modulated[:, 0, :]
        predicted_next_proprio = self.cerebellum(proprio, first_action)

        # 6. Error correction (Cerebellar feedback)
        # Error = predicted - current (simplified)
        prediction_error = predicted_next_proprio - proprio
        correction = self.error_corrector(prediction_error)
        correction = correction * torch.sigmoid(self.correction_gain)

        # Apply correction to all actions
        corrected = modulated + correction.unsqueeze(1)

        # 7. Temporal smoothing (spinal cord)
        smoothed = corrected.permute(0, 2, 1)  # (B, action, chunk)
        smoothed = self.smoother(smoothed)
        smoothed = smoothed.permute(0, 2, 1)  # (B, chunk, action)

        # 8. Final output
        actions = torch.tanh(smoothed)

        if return_components:
            return actions, {
                'plan_embed': plan_embed,
                'primitive_weights': primitive_weights,
                'blended': blended,
                'amplitude': amplitude,
                'speed': speed,
                'offset': offset,
                'correction': correction,
                'predicted_next_proprio': predicted_next_proprio,
            }

        return actions

    def get_primitive_usage(self, plans: List[Dict[str, Any]]) -> None:
        """Print which primitives are being used (for interpretability)."""
        plan_embed = self.plan_encoder(plans)
        weights = self.selector(plan_embed)

        for b, w in enumerate(weights):
            print(f"\nSample {b} primitive usage:")
            for i, (name, weight) in enumerate(zip(self.primitives.primitive_names, w)):
                if weight > 0.1:
                    print(f"  {name}: {weight:.2%}")
