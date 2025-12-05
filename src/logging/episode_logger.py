"""Episode logging for debugging and training data collection.

Logs complete episode traces including skills, world state, and observations.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from ..skills.base import SkillResult
from ..world_model.state import WorldState
from ..config import RunConfig
from ..utils.git_info import get_git_info
from ..utils.timing import Timer


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


@dataclass
class EpisodeLog:
    """Complete episode record for debugging and training.
    
    Captures everything needed to:
    1. Debug skill failures
    2. Analyze Qwen interactions
    3. Train future policies (BC/RL)
    """
    
    # Episode metadata
    task: str
    task_id: int
    episode_idx: int
    timestamp: str
    seed: int
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    code_version: Dict[str, Any] = field(default_factory=dict)
    
    # Execution trace
    skill_sequence: List[Dict[str, Any]] = field(default_factory=list)
    world_state_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    # Observations (for BC/RL later)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[List[float]] = field(default_factory=list)
    
    # Outcome
    success: bool = False
    failure_reason: Optional[str] = None
    total_steps: int = 0
    total_time: float = 0.0
    
    # Timing breakdown
    timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Qwen interactions (for Phase 2+)
    qwen_prompts: List[str] = field(default_factory=list)
    qwen_responses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EpisodeLogger:
    """Logger for episode traces.
    
    Usage:
        logger = EpisodeLogger("logs/run_001")
        logger.start_episode(task="pick up bowl", task_id=0, episode_idx=0, seed=42)
        
        # During episode:
        logger.log_skill("ApproachObject", {"obj": "bowl_1"}, result)
        logger.log_world_state(world_state)
        logger.log_observation(obs, step=10)
        logger.log_action(action)
        
        # End of episode:
        logger.end_episode(success=True)
    """
    
    def __init__(
        self,
        output_dir: str,
        config: Optional[RunConfig] = None,
        save_observations: bool = True,
        observation_sample_rate: int = 10,
    ):
        """Initialize logger.
        
        Args:
            output_dir: Directory to save episode logs.
            config: Run configuration to log.
            save_observations: Whether to save observations.
            observation_sample_rate: Save every N steps.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.save_observations = save_observations
        self.observation_sample_rate = observation_sample_rate
        
        self.current_episode: Optional[EpisodeLog] = None
        self.timer = Timer()
        self._episode_start_time: float = 0.0
        self._step_count: int = 0
    
    def start_episode(
        self,
        task: str,
        task_id: int,
        episode_idx: int,
        seed: int,
    ):
        """Start logging a new episode.
        
        Args:
            task: Task description.
            task_id: Task ID in suite.
            episode_idx: Episode index in run.
            seed: Random seed for this episode.
        """
        self.timer.reset()
        self._episode_start_time = time.time()
        self._step_count = 0
        
        self.current_episode = EpisodeLog(
            task=task,
            task_id=task_id,
            episode_idx=episode_idx,
            timestamp=datetime.now().isoformat(),
            seed=seed,
            config=self.config.to_dict() if self.config else {},
            code_version=get_git_info(),
        )
    
    def log_skill(
        self,
        skill_name: str,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Log a skill execution.
        
        Args:
            skill_name: Name of the skill.
            args: Skill arguments.
            result: Skill result.
        """
        if self.current_episode is None:
            return
        
        # Convert numpy arrays in result.info
        info = {}
        for k, v in result.info.items():
            if isinstance(v, np.ndarray):
                info[k] = v.tolist()
            else:
                info[k] = v
        
        self.current_episode.skill_sequence.append({
            "skill": skill_name,
            "args": args,
            "success": result.success,
            "info": info,
            "timestamp": time.time() - self._episode_start_time,
        })
    
    def log_world_state(self, world_state: WorldState):
        """Log world state snapshot.
        
        Args:
            world_state: Current world state.
        """
        if self.current_episode is None:
            return
        
        self.current_episode.world_state_trace.append({
            "state": world_state.to_dict(),
            "timestamp": time.time() - self._episode_start_time,
        })
    
    def log_observation(self, obs: Dict[str, Any], step: int):
        """Log observation (sampled).
        
        Args:
            obs: Observation dictionary.
            step: Current step number.
        """
        if self.current_episode is None:
            return
        
        if not self.save_observations:
            return
        
        # Only save at sample rate
        if step % self.observation_sample_rate != 0:
            return
        
        # Extract key info (don't save full images to JSON)
        obs_record = {
            "step": step,
            "timestamp": time.time() - self._episode_start_time,
        }
        
        if 'robot0_eef_pos' in obs:
            obs_record["eef_pos"] = obs['robot0_eef_pos'].tolist() if isinstance(obs['robot0_eef_pos'], np.ndarray) else obs['robot0_eef_pos']
        if 'robot0_eef_quat' in obs:
            obs_record["eef_quat"] = obs['robot0_eef_quat'].tolist() if isinstance(obs['robot0_eef_quat'], np.ndarray) else obs['robot0_eef_quat']
        if 'robot0_gripper_qpos' in obs:
            obs_record["gripper_qpos"] = obs['robot0_gripper_qpos'].tolist() if isinstance(obs['robot0_gripper_qpos'], np.ndarray) else obs['robot0_gripper_qpos']
        
        self.current_episode.observations.append(obs_record)
        self._step_count = max(self._step_count, step)
    
    def log_action(self, action: np.ndarray):
        """Log action taken.
        
        Args:
            action: Action array.
        """
        if self.current_episode is None:
            return
        
        self.current_episode.actions.append(action.tolist() if isinstance(action, np.ndarray) else action)
    
    def log_qwen(self, prompt: str, response: str):
        """Log Qwen interaction.
        
        Args:
            prompt: Prompt sent to Qwen.
            response: Response from Qwen.
        """
        if self.current_episode is None:
            return
        
        self.current_episode.qwen_prompts.append(prompt)
        self.current_episode.qwen_responses.append(response)
    
    def end_episode(
        self,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> str:
        """End episode and save log.
        
        Args:
            success: Whether episode succeeded.
            failure_reason: Reason for failure if any.
            
        Returns:
            Path to saved log file.
        """
        if self.current_episode is None:
            return ""
        
        self.current_episode.success = success
        self.current_episode.failure_reason = failure_reason
        self.current_episode.total_time = time.time() - self._episode_start_time
        self.current_episode.total_steps = self._step_count
        self.current_episode.timing = self.timer.summary()
        
        # Save to disk
        filename = f"episode_{self.current_episode.episode_idx:04d}_{self.current_episode.timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.current_episode.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Reset
        episode = self.current_episode
        self.current_episode = None
        
        return str(filepath)
    
    def get_timer(self) -> Timer:
        """Get timer for external timing measurements."""
        return self.timer


class RunSummary:
    """Summary of a full run (multiple episodes)."""
    
    def __init__(self, output_dir: str, config: Optional[RunConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        self.episodes: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def add_episode(self, episode_log: EpisodeLog):
        """Add episode result to summary."""
        self.episodes.append({
            "episode_idx": episode_log.episode_idx,
            "task": episode_log.task,
            "success": episode_log.success,
            "failure_reason": episode_log.failure_reason,
            "total_steps": episode_log.total_steps,
            "total_time": episode_log.total_time,
            "n_skills": len(episode_log.skill_sequence),
        })
    
    def save(self):
        """Save run summary."""
        n_success = sum(1 for e in self.episodes if e["success"])
        n_total = len(self.episodes)
        
        summary = {
            "config": self.config.to_dict() if self.config else {},
            "code_version": get_git_info(),
            "timestamp": datetime.now().isoformat(),
            "total_episodes": n_total,
            "successful_episodes": n_success,
            "success_rate": n_success / n_total if n_total > 0 else 0.0,
            "total_run_time": time.time() - self.start_time,
            "episodes": self.episodes,
        }
        
        with open(self.output_dir / "run_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
