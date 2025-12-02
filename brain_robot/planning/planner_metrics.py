"""Planner metrics for tracking Phase 2 performance.

Tracks success/failure at each stage:
- Parse: Could we extract JSON from Qwen output?
- Validation: Was the plan structurally valid?
- Execution: Did all skills execute successfully?
- Goal: Was the task goal achieved?
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class PlannerMetrics:
    """Track planner performance at each stage."""

    # Overall counts
    total_attempts: int = 0

    # Parse stage
    parse_success: int = 0
    parse_failures: int = 0

    # Validation stage
    validation_success: int = 0
    validation_failures: int = 0

    # Execution stage
    execution_success: int = 0
    execution_failures: int = 0

    # Goal stage
    goal_reached: int = 0
    goal_not_reached: int = 0

    # Per-task tracking
    per_task_metrics: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {
        "attempts": 0,
        "parse_success": 0,
        "validation_success": 0,
        "execution_success": 0,
        "goal_reached": 0,
    }))

    # Sample failures for debugging (one per failure type)
    sample_parse_failure: Optional[Dict[str, Any]] = None
    sample_validation_failure: Optional[Dict[str, Any]] = None
    sample_execution_failure: Optional[Dict[str, Any]] = None
    sample_goal_failure: Optional[Dict[str, Any]] = None

    def record_attempt(self, task_id: str):
        """Record a new planning attempt."""
        self.total_attempts += 1
        self.per_task_metrics[task_id]["attempts"] += 1

    def record_parse_success(self, task_id: str):
        """Record successful JSON parsing."""
        self.parse_success += 1
        self.per_task_metrics[task_id]["parse_success"] += 1

    def record_parse_failure(self, task_id: str, prompt: str, raw_output: str, error: str):
        """Record JSON parsing failure."""
        self.parse_failures += 1
        if self.sample_parse_failure is None:
            self.sample_parse_failure = {
                "task_id": task_id,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "raw_output": raw_output[:500] + "..." if len(raw_output) > 500 else raw_output,
                "error": error,
            }

    def record_validation_success(self, task_id: str):
        """Record successful plan validation."""
        self.validation_success += 1
        self.per_task_metrics[task_id]["validation_success"] += 1

    def record_validation_failure(self, task_id: str, plan: List[Dict], error: str):
        """Record plan validation failure."""
        self.validation_failures += 1
        if self.sample_validation_failure is None:
            self.sample_validation_failure = {
                "task_id": task_id,
                "plan": plan,
                "error": error,
            }

    def record_execution_success(self, task_id: str):
        """Record successful plan execution (all skills succeeded)."""
        self.execution_success += 1
        self.per_task_metrics[task_id]["execution_success"] += 1

    def record_execution_failure(self, task_id: str, plan: List[Dict], failed_step: int, error: str):
        """Record plan execution failure."""
        self.execution_failures += 1
        if self.sample_execution_failure is None:
            self.sample_execution_failure = {
                "task_id": task_id,
                "plan": plan,
                "failed_step": failed_step,
                "error": error,
            }

    def record_goal_reached(self, task_id: str):
        """Record that the task goal was achieved."""
        self.goal_reached += 1
        self.per_task_metrics[task_id]["goal_reached"] += 1

    def record_goal_not_reached(self, task_id: str, plan: List[Dict], reason: str):
        """Record that execution completed but goal not reached."""
        self.goal_not_reached += 1
        if self.sample_goal_failure is None:
            self.sample_goal_failure = {
                "task_id": task_id,
                "plan": plan,
                "reason": reason,
            }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_attempts": self.total_attempts,
            "parse_rate": self.parse_success / max(1, self.total_attempts),
            "validation_rate": self.validation_success / max(1, self.parse_success),
            "execution_rate": self.execution_success / max(1, self.validation_success),
            "goal_rate": self.goal_reached / max(1, self.execution_success),
            "overall_success_rate": self.goal_reached / max(1, self.total_attempts),
            "per_task": dict(self.per_task_metrics),
        }

    def print_summary(self):
        """Print human-readable summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("PLANNER METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Attempts: {s['total_attempts']}")
        print(f"Parse Rate: {s['parse_rate']:.1%} ({self.parse_success}/{self.total_attempts})")
        print(f"Validation Rate: {s['validation_rate']:.1%} ({self.validation_success}/{self.parse_success})")
        print(f"Execution Rate: {s['execution_rate']:.1%} ({self.execution_success}/{self.validation_success})")
        print(f"Goal Rate: {s['goal_rate']:.1%} ({self.goal_reached}/{self.execution_success})")
        print(f"Overall Success: {s['overall_success_rate']:.1%} ({self.goal_reached}/{self.total_attempts})")

        if self.per_task_metrics:
            print("\nPer-Task Breakdown:")
            for task_id, metrics in self.per_task_metrics.items():
                task_success = metrics["goal_reached"] / max(1, metrics["attempts"])
                print(f"  {task_id}: {task_success:.1%} ({metrics['goal_reached']}/{metrics['attempts']})")

    def save(self, output_path: str):
        """Save metrics to JSON file."""
        output = {
            "summary": self.summary(),
            "raw_counts": {
                "total_attempts": self.total_attempts,
                "parse_success": self.parse_success,
                "parse_failures": self.parse_failures,
                "validation_success": self.validation_success,
                "validation_failures": self.validation_failures,
                "execution_success": self.execution_success,
                "execution_failures": self.execution_failures,
                "goal_reached": self.goal_reached,
                "goal_not_reached": self.goal_not_reached,
            },
            "sample_failures": {
                "parse": self.sample_parse_failure,
                "validation": self.sample_validation_failure,
                "execution": self.sample_execution_failure,
                "goal": self.sample_goal_failure,
            },
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
