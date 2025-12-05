"""
VLM-Conditioned Policy for Brain-Inspired Robot Control.

Combines:
1. Qwen2.5-VL for high-level visual planning
2. Brain-inspired policy for low-level motor control
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import time


class VLMConditionedPolicy:
    """
    Wrapper that combines VLM planning with brain-inspired policy execution.

    The VLM runs at low frequency (1-2 Hz) to generate high-level plans,
    while the brain-inspired policy runs at high frequency (20+ Hz) for
    smooth motor control.
    """

    def __init__(
        self,
        vlm_planner,  # QwenVLPlanner instance
        brain_policy: nn.Module,  # BrainInspiredActionGenerator
        plan_interval: int = 20,  # Steps between VLM queries
        device: str = "cuda",
        use_async: bool = False,
    ):
        """
        Initialize VLM-conditioned policy.

        Args:
            vlm_planner: QwenVLPlanner instance for visual planning
            brain_policy: Brain-inspired policy for motor control
            plan_interval: Number of steps between VLM queries
            device: Device to run policy on
            use_async: Whether to use async VLM planning
        """
        self.vlm = vlm_planner
        self.brain_policy = brain_policy.to(device)
        self.plan_interval = plan_interval
        self.device = device
        self.use_async = use_async

        # State tracking
        self.cached_plan = None
        self.steps_since_plan = 0
        self.previous_phase = "approach"
        self.task_description = None

        # Timing stats
        self.vlm_times = []
        self.policy_times = []

    def reset(self, task_description: str = None):
        """Reset policy state for new episode."""
        self.cached_plan = self._default_plan()
        self.steps_since_plan = 0
        self.previous_phase = "approach"
        self.task_description = task_description
        self.vlm_times = []
        self.policy_times = []

        if self.vlm is not None:
            self.vlm.reset()

    def _default_plan(self) -> Dict[str, Any]:
        """Return default plan when VLM is not available."""
        return {
            "observation": {
                "target_object": "unknown",
                "gripper_position": "unknown",
                "distance_to_target": "far",
                "obstacles": [],
            },
            "plan": {
                "phase": "approach",
                "movements": [
                    {"direction": "forward", "speed": "slow", "steps": 1}
                ],
                "gripper": "maintain",
                "confidence": 0.5,
            },
            "reasoning": "Default plan",
        }

    def should_replan(self, force: bool = False) -> bool:
        """Determine if we should query the VLM."""
        if force:
            return True
        if self.cached_plan is None:
            return True
        if self.steps_since_plan >= self.plan_interval:
            return True
        # Could add confidence-based or phase-change-based triggers here
        return False

    def get_vlm_plan(
        self,
        image: np.ndarray,
        task_description: str,
        gripper_state: str = "open",
    ) -> Dict[str, Any]:
        """Query VLM for a new plan."""
        if self.vlm is None:
            return self._default_plan()

        start_time = time.time()

        try:
            plan = self.vlm.plan(
                image=image,
                task_description=task_description,
                gripper_state=gripper_state,
                steps_since_plan=self.steps_since_plan,
            )
        except Exception as e:
            print(f"[VLM Error] {e}")
            plan = self._default_plan()

        vlm_time = time.time() - start_time
        self.vlm_times.append(vlm_time)

        return plan

    def get_action(
        self,
        image: np.ndarray,
        proprio: np.ndarray,
        task_description: str = None,
        gripper_state: str = "open",
        force_replan: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action from VLM-conditioned policy.

        Args:
            image: RGB image from robot camera (H, W, 3)
            proprio: Proprioception vector
            task_description: Task description for VLM
            gripper_state: Current gripper state ("open" or "closed")
            force_replan: Force VLM query regardless of interval

        Returns:
            action: Action vector (action_dim,)
            info: Dictionary with plan and timing info
        """
        task_desc = task_description or self.task_description or "complete the task"

        # Query VLM if needed
        if self.should_replan(force=force_replan):
            self.cached_plan = self.get_vlm_plan(image, task_desc, gripper_state)
            self.steps_since_plan = 0

            # Update phase tracking
            if "plan" in self.cached_plan and "phase" in self.cached_plan["plan"]:
                self.previous_phase = self.cached_plan["plan"]["phase"]

        # Get action from brain-inspired policy
        start_time = time.time()

        proprio_tensor = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions, components = self.brain_policy(
                plans=[self.cached_plan],
                proprio=proprio_tensor,
                return_components=True,
            )
            # Take first action from chunk
            action = actions[0, 0, :].cpu().numpy()

        policy_time = time.time() - start_time
        self.policy_times.append(policy_time)

        self.steps_since_plan += 1

        # Build info dict
        info = {
            "plan": self.cached_plan,
            "steps_since_plan": self.steps_since_plan,
            "vlm_time": self.vlm_times[-1] if self.vlm_times else 0,
            "policy_time": policy_time,
            "primitive_weights": components["primitive_weights"][0].cpu().numpy(),
            "phase": self.cached_plan.get("plan", {}).get("phase", "unknown"),
            "confidence": self.cached_plan.get("plan", {}).get("confidence", 0.5),
        }

        return action, info

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        return {
            "avg_vlm_time": np.mean(self.vlm_times) if self.vlm_times else 0,
            "avg_policy_time": np.mean(self.policy_times) if self.policy_times else 0,
            "vlm_fps": 1.0 / np.mean(self.vlm_times) if self.vlm_times else 0,
            "policy_fps": 1.0 / np.mean(self.policy_times) if self.policy_times else 0,
        }


class VLMConditionedPolicySimple(nn.Module):
    """
    Simpler VLM-conditioned policy that uses plan embeddings directly.

    This version takes pre-computed plan embeddings and outputs actions,
    suitable for training with BC when VLM plans are pre-computed.
    """

    def __init__(
        self,
        plan_dim: int = 128,
        proprio_dim: int = 15,
        action_dim: int = 7,
        hidden_dim: int = 256,
        n_primitives: int = 8,
    ):
        super().__init__()

        self.plan_dim = plan_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        # Plan encoder (for JSON plans)
        from src.action_generator.plan_encoder import RelativePlanEncoder
        self.plan_encoder = RelativePlanEncoder(embed_dim=plan_dim)

        # Primitive selector
        self.primitive_selector = nn.Sequential(
            nn.Linear(plan_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_primitives),
            nn.Softmax(dim=-1),
        )

        # Learned primitives
        self.primitives = nn.Parameter(
            torch.randn(n_primitives, action_dim) * 0.1
        )

        # Modulator for fine adjustment
        self.modulator = nn.Sequential(
            nn.Linear(plan_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.n_primitives = n_primitives

    def forward(
        self,
        plans: List[Dict[str, Any]],
        proprio: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Generate action from plan and proprioception.

        Args:
            plans: List of plan dictionaries from VLM
            proprio: (B, proprio_dim) proprioception
            return_info: Whether to return component info

        Returns:
            action: (B, action_dim) action vector
        """
        # Encode plans
        plan_embed = self.plan_encoder(plans).to(proprio.device)  # (B, plan_dim)

        # Concatenate plan and proprio
        x = torch.cat([plan_embed, proprio], dim=-1)

        # Select primitives
        weights = self.primitive_selector(x)  # (B, n_primitives)

        # Blend primitives
        base_action = torch.einsum('bp,pa->ba', weights, self.primitives)

        # Add modulation
        modulation = self.modulator(x) * 0.2  # Scale modulation

        action = base_action + modulation
        action = torch.tanh(action)

        if return_info:
            info = {
                "plan_embed": plan_embed,
                "primitive_weights": weights,
                "base_action": base_action,
                "modulation": modulation,
            }
            return action, info

        return action


class OraclePlanPolicy(nn.Module):
    """
    Policy that uses oracle (scripted) plans for comparison.

    This allows testing the brain-inspired policy with perfect plans
    to isolate VLM quality from policy capability.
    """

    def __init__(
        self,
        proprio_dim: int = 15,
        action_dim: int = 7,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Phase embedding
        self.phase_vocab = {
            'approach': 0, 'align': 1, 'descend': 2, 'grasp': 3,
            'lift': 4, 'move': 5, 'place': 6, 'release': 7
        }
        self.phase_embed = nn.Embedding(8, 32)

        # Direction embedding
        self.direction_vocab = {
            'left': 0, 'right': 1, 'forward': 2, 'backward': 3,
            'up': 4, 'down': 5, 'none': 6
        }
        self.direction_embed = nn.Embedding(7, 32)

        # Policy network
        self.net = nn.Sequential(
            nn.Linear(proprio_dim + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        proprio: torch.Tensor,
        phase: str,
        direction: str = "forward",
    ) -> torch.Tensor:
        """
        Generate action from proprioception and oracle phase/direction.
        """
        device = proprio.device
        B = proprio.shape[0]

        # Get embeddings
        phase_idx = torch.tensor([self.phase_vocab.get(phase, 0)] * B, device=device)
        dir_idx = torch.tensor([self.direction_vocab.get(direction, 6)] * B, device=device)

        phase_emb = self.phase_embed(phase_idx)
        dir_emb = self.direction_embed(dir_idx)

        # Concatenate and forward
        x = torch.cat([proprio, phase_emb, dir_emb], dim=-1)
        return self.net(x)
