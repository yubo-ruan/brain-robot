"""Centralized configuration for the src system.

All configuration dataclasses for the zero-shot LIBERO implementation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SkillConfig:
    """Configuration for skill execution parameters."""

    # Approach skill
    approach_max_steps: int = 200  # Increased for convergence
    approach_pos_threshold: float = 0.03  # 3cm
    approach_pregrasp_height: float = 0.10  # 10cm above object

    # Grasp skill
    grasp_max_steps: int = 100  # Increased for all phases
    grasp_lower_speed: float = 0.02  # m/step descent rate
    grasp_lift_height: float = 0.05  # 5cm lift after grasp
    grasp_close_threshold: float = 0.01  # Gripper width indicating closed

    # Move skill
    move_max_steps: int = 400  # Increased for longer translations
    move_pos_threshold: float = 0.05  # 5cm (slightly more relaxed)

    # Place skill
    place_max_steps: int = 100  # Increased
    place_lower_speed: float = 0.02
    place_release_height: float = 0.05  # 5cm above surface

    # Control gains - tuned for robosuite action space
    kp_pos: float = 10.0  # Increased from 5.0
    kp_ori: float = 3.0   # Increased from 2.0

    # Safety limits
    max_action_magnitude: float = 1.0


@dataclass
class PerceptionConfig:
    """Configuration for perception system."""

    use_oracle: bool = True

    # Noise parameters for NoisyOraclePerception
    noise_pos_std: float = 0.0  # Position noise standard deviation (meters)
    noise_ori_std: float = 0.0  # Orientation noise standard deviation (radians)

    # Perception staleness threshold
    stale_threshold_sec: float = 1.0


@dataclass
class LoggingConfig:
    """Configuration for logging and recording."""

    output_dir: str = "logs"
    save_observations: bool = True
    observation_sample_rate: int = 10  # Save every N steps
    save_gifs: bool = True
    gif_fps: int = 10


@dataclass
class RunConfig:
    """Top-level configuration for a run."""

    # Seed management
    seed: int = 42

    # Task selection
    task_suite: str = "libero_spatial"
    task_id: int = 0

    # Episode settings
    n_episodes: int = 20
    max_episode_steps: int = 500

    # Success criteria
    success_rate_threshold: float = 0.8

    # Sub-configs
    skill: SkillConfig = field(default_factory=SkillConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Create from dictionary."""
        skill = SkillConfig(**d.pop("skill", {}))
        perception = PerceptionConfig(**d.pop("perception", {}))
        logging = LoggingConfig(**d.pop("logging", {}))
        return cls(skill=skill, perception=perception, logging=logging, **d)
