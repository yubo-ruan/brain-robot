"""Failure taxonomy classifier for Phase 5 analysis.

Classifies episode failures into categories based on which skill failed
and the failure context.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter


class FailureMode(Enum):
    """Failure mode categories."""

    # Success (not a failure)
    SUCCESS = "success"

    # Perception failures
    PERCEPTION_MISS = "perception_miss"  # Object not detected by YOLO
    PERCEPTION_WRONG = "perception_wrong"  # Wrong object selected by grounder

    # Skill failures
    APPROACH_TIMEOUT = "approach_timeout"  # Failed to reach pre-grasp position
    GRASP_MISS = "grasp_miss"  # Gripper closed on empty space
    GRASP_SLIP = "grasp_slip"  # Object dropped during/after grasp
    MOVE_TIMEOUT = "move_timeout"  # Failed to reach target region
    PLACE_MISS = "place_miss"  # Object not released correctly

    # Unknown/other
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis result for a single episode."""

    episode_file: str
    success: bool
    failure_mode: FailureMode
    failed_skill: Optional[str]
    failure_details: Dict[str, Any]


class FailureClassifier:
    """Classifies failures from episode logs."""

    def __init__(self):
        """Initialize classifier."""
        pass

    def classify_episode(self, episode_log: Dict[str, Any]) -> FailureAnalysis:
        """Classify failure mode from episode log.

        Args:
            episode_log: Episode log dictionary from JSON file.

        Returns:
            FailureAnalysis with classification result.
        """
        success = episode_log.get("success", False)

        if success:
            return FailureAnalysis(
                episode_file="",
                success=True,
                failure_mode=FailureMode.SUCCESS,
                failed_skill=None,
                failure_details={}
            )

        # Find which skill failed
        skill_sequence = episode_log.get("skill_sequence", [])

        for skill in skill_sequence:
            if not skill.get("success", True):
                skill_name = skill.get("skill", "")
                info = skill.get("info", {})

                failure_mode, details = self._classify_skill_failure(
                    skill_name, info, episode_log
                )

                return FailureAnalysis(
                    episode_file="",
                    success=False,
                    failure_mode=failure_mode,
                    failed_skill=skill_name,
                    failure_details=details
                )

        # No explicit failure found but episode failed
        return FailureAnalysis(
            episode_file="",
            success=False,
            failure_mode=FailureMode.UNKNOWN,
            failed_skill=None,
            failure_details={"reason": "Episode failed but no skill failure recorded"}
        )

    def _classify_skill_failure(
        self,
        skill_name: str,
        info: Dict[str, Any],
        episode_log: Dict[str, Any]
    ) -> tuple[FailureMode, Dict[str, Any]]:
        """Classify failure based on which skill failed.

        Args:
            skill_name: Name of the failed skill.
            info: Skill info dictionary.
            episode_log: Full episode log for context.

        Returns:
            Tuple of (FailureMode, details dict).
        """
        error_msg = info.get("error_msg", "")

        if skill_name == "ApproachObject":
            # Check if it's a perception issue (object not found)
            if "not found" in error_msg.lower():
                return FailureMode.PERCEPTION_MISS, {
                    "error": error_msg,
                    "xy_error": info.get("xy_error"),
                    "z_error": info.get("z_error"),
                }

            # Approach timeout
            if info.get("timeout", False):
                return FailureMode.APPROACH_TIMEOUT, {
                    "error": error_msg,
                    "steps_taken": info.get("steps_taken"),
                    "final_error": info.get("final_error"),
                    "xy_error": info.get("xy_error"),
                    "z_error": info.get("z_error"),
                }

            return FailureMode.APPROACH_TIMEOUT, {"error": error_msg}

        elif skill_name == "GraspObject":
            # Check XY refinement info
            xy_ref = info.get("xy_refinement", {})
            xy_error_after = xy_ref.get("xy_error_after", 0)

            # Large XY error suggests perception problem
            if xy_error_after > 0.10:  # >10cm
                return FailureMode.PERCEPTION_WRONG, {
                    "error": error_msg,
                    "xy_error_before": xy_ref.get("xy_error_before"),
                    "xy_error_after": xy_error_after,
                    "converged": xy_ref.get("converged"),
                }

            # Gripper closed but didn't grasp object
            if "did not close on object" in error_msg.lower():
                return FailureMode.GRASP_MISS, {
                    "error": error_msg,
                    "xy_refinement": xy_ref,
                }

            return FailureMode.GRASP_MISS, {"error": error_msg}

        elif skill_name == "MoveObjectToRegion":
            # Check if object was dropped (grasp slip)
            if "dropped" in error_msg.lower() or "lost" in error_msg.lower():
                return FailureMode.GRASP_SLIP, {"error": error_msg}

            # Move timeout
            if "timeout" in error_msg.lower() or "failed to reach" in error_msg.lower():
                return FailureMode.MOVE_TIMEOUT, {
                    "error": error_msg,
                    "steps_taken": info.get("steps_taken"),
                }

            return FailureMode.MOVE_TIMEOUT, {"error": error_msg}

        elif skill_name == "PlaceObject":
            return FailureMode.PLACE_MISS, {"error": error_msg}

        return FailureMode.UNKNOWN, {"error": error_msg, "skill": skill_name}

    def analyze_log_directory(self, log_dir: Path) -> Dict[str, Any]:
        """Analyze all episode logs in a directory.

        Args:
            log_dir: Directory containing episode JSON files.

        Returns:
            Analysis summary dictionary.
        """
        episode_files = list(log_dir.rglob("episode_*.json"))

        analyses = []
        for ep_file in sorted(episode_files):
            try:
                with open(ep_file) as f:
                    episode_log = json.load(f)

                analysis = self.classify_episode(episode_log)
                analysis.episode_file = str(ep_file)
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing {ep_file}: {e}")

        return self._summarize_analyses(analyses)

    def _summarize_analyses(self, analyses: List[FailureAnalysis]) -> Dict[str, Any]:
        """Summarize multiple episode analyses.

        Args:
            analyses: List of FailureAnalysis objects.

        Returns:
            Summary dictionary.
        """
        total = len(analyses)
        successes = sum(1 for a in analyses if a.success)
        failures = total - successes

        # Count failure modes
        failure_counts = Counter(
            a.failure_mode.value for a in analyses if not a.success
        )

        # Count failed skills
        skill_counts = Counter(
            a.failed_skill for a in analyses
            if not a.success and a.failed_skill
        )

        # Build summary
        summary = {
            "total_episodes": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0,

            "failure_modes": {
                mode.value: {
                    "count": failure_counts.get(mode.value, 0),
                    "percentage": failure_counts.get(mode.value, 0) / failures * 100
                        if failures > 0 else 0.0
                }
                for mode in FailureMode
                if mode != FailureMode.SUCCESS
            },

            "failed_skills": dict(skill_counts),

            "failure_details": [
                {
                    "file": a.episode_file,
                    "mode": a.failure_mode.value,
                    "skill": a.failed_skill,
                    "details": a.failure_details,
                }
                for a in analyses if not a.success
            ]
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary.

        Args:
            summary: Summary dictionary from analyze_log_directory.
        """
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal Episodes: {summary['total_episodes']}")
        print(f"Successes: {summary['successes']}")
        print(f"Failures: {summary['failures']}")
        print(f"Success Rate: {summary['success_rate']*100:.1f}%")

        print("\n--- Failure Mode Distribution ---")
        print(f"{'Mode':<25} {'Count':>8} {'Percentage':>12}")
        print("-" * 50)

        for mode, data in sorted(
            summary["failure_modes"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        ):
            if data["count"] > 0:
                print(f"{mode:<25} {data['count']:>8} {data['percentage']:>11.1f}%")

        print("\n--- Failed Skills ---")
        for skill, count in sorted(
            summary["failed_skills"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {skill}: {count}")

        print("\n" + "=" * 60)


def main():
    """Run failure analysis on Phase 5 evaluation logs."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze failure modes from evaluation logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/phase5_full_evaluation",
        help="Directory containing evaluation logs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for analysis results"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    classifier = FailureClassifier()
    summary = classifier.analyze_log_directory(log_dir)
    classifier.print_summary(summary)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
