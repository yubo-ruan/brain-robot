"""MOKA-based task planning with visual prompting.

This module integrates the MOKA (Mark-based Visual Prompting) framework
for open-world robotic manipulation planning.

MOKA provides:
- Task decomposition into subtasks via VLM
- Visual affordance selection (keypoints on objects)
- Waypoint-based motion planning (pre-contact, target, post-contact)
- Collision avoidance through grid-based waypoints
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass

# Add MOKA to path
MOKA_PATH = "/workspace/new_experiment"
if MOKA_PATH not in sys.path:
    sys.path.insert(0, MOKA_PATH)


@dataclass
class MOKAContext:
    """3D motion context from MOKA planning."""
    subtask_id: int
    instruction: str

    # Objects
    object_grasped: str
    object_unattached: Optional[str]
    motion_direction: str  # "upward", "downward", "pull", "push"

    # 2D keypoints (image coordinates)
    grasp_keypoint_2d: Optional[Tuple[int, int]]  # P point
    target_keypoint_2d: Optional[Tuple[int, int]]  # Q point

    # 3D poses (world coordinates)
    grasp_pose_3d: Optional[np.ndarray]  # [x, y, z, qw, qx, qy, qz]
    target_pose_3d: Optional[np.ndarray]
    pre_contact_waypoint_3d: Optional[np.ndarray]
    post_contact_waypoint_3d: Optional[np.ndarray]

    # Additional info
    grasp_yaw: Optional[float]
    target_tile: Optional[str]  # Grid position like "d2"
    confidence: float = 1.0


class MOKAPlanner:
    """Adapter to integrate MOKA planning with brain_robot execution."""

    def __init__(
        self,
        config_path: str = "/workspace/new_experiment/config/moka.yaml",
        use_moka: bool = True,
    ):
        """Initialize MOKA planner.

        Args:
            config_path: Path to MOKA configuration file
            use_moka: If False, falls back to simple heuristic planning
        """
        self.use_moka = use_moka
        self.config_path = config_path

        if use_moka:
            try:
                from moka.planners.visual_prompt_planner import VisualPromptPlanner
                from moka.qwen_utils import load_qwen_model

                print("[MOKAPlanner] Loading MOKA visual prompt planner...")
                self.moka_planner = VisualPromptPlanner(config_path)

                print("[MOKAPlanner] Warming up Qwen model...")
                load_qwen_model()

                print("[MOKAPlanner] MOKA ready!")
                self.moka_available = True

            except Exception as e:
                print(f"[MOKAPlanner] Failed to load MOKA: {e}")
                print("[MOKAPlanner] Falling back to simple planning")
                self.moka_available = False
        else:
            self.moka_available = False

    def plan(
        self,
        task_description: str,
        world_state: Any,
        env: Any,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate skill sequence using MOKA visual prompting.

        Args:
            task_description: Natural language task (e.g., "pick up the black bowl")
            world_state: Current world state with object poses
            env: LIBERO environment
            rgb_image: RGB observation (H, W, 3), if None will render from env
            depth_image: Depth observation (H, W), if None will render from env
            camera_intrinsics: Camera intrinsic matrix (3, 3)
            camera_extrinsics: Camera extrinsic matrix (4, 4)

        Returns:
            Dictionary with:
                - "success": bool
                - "plan": List[Dict] - skill sequence
                - "moka_contexts": List[MOKAContext] - 3D motion contexts
                - "error": Optional[str]
        """
        if not self.moka_available:
            return self._fallback_plan(task_description, world_state)

        # Get observations
        if rgb_image is None:
            rgb_image = self._get_rgb_observation(env)

        if depth_image is None:
            depth_image = self._get_depth_observation(env)

        if camera_intrinsics is None or camera_extrinsics is None:
            camera_intrinsics, camera_extrinsics = self._get_camera_params(env)

        try:
            # Step 1: MOKA subtask decomposition
            print(f"[MOKAPlanner] Planning task: {task_description}")
            subtasks = self._propose_subtasks(task_description, rgb_image)

            if not subtasks:
                return {
                    "success": False,
                    "error": "MOKA failed to generate subtasks",
                    "plan": [],
                    "moka_contexts": []
                }

            print(f"[MOKAPlanner] Generated {len(subtasks)} subtasks")
            for i, st in enumerate(subtasks):
                print(f"  {i+1}. {st.get('instruction', st)}")

            # Step 2: For each subtask, get motion context
            moka_contexts = []
            for i, subtask in enumerate(subtasks):
                context = self._get_motion_context(
                    subtask_id=i,
                    subtask=subtask,
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    world_state=world_state,
                    camera_intrinsics=camera_intrinsics,
                    camera_extrinsics=camera_extrinsics,
                )
                moka_contexts.append(context)

            # Step 3: Convert MOKA contexts to brain_robot skill sequence
            skill_sequence = self._contexts_to_skills(moka_contexts, world_state)

            return {
                "success": True,
                "plan": skill_sequence,
                "moka_contexts": moka_contexts,
                "num_subtasks": len(subtasks),
            }

        except Exception as e:
            print(f"[MOKAPlanner] Error during planning: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "plan": [],
                "moka_contexts": []
            }

    def _propose_subtasks(
        self,
        task_description: str,
        rgb_image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Use MOKA VLM to decompose task into subtasks.

        Returns:
            List of subtask dicts with keys:
                - instruction: str
                - object_grasped: str
                - object_unattached: str (optional)
                - motion_direction: str ("upward", "downward", etc.)
        """
        # TODO: Call actual MOKA planner
        # For now, return a simple hardcoded plan as placeholder
        # This will be replaced with actual MOKA VLM call

        print("[MOKAPlanner] TODO: Call MOKA VLM for subtask proposal")
        print("[MOKAPlanner] Using placeholder decomposition for now")

        # Parse simple pick-place tasks
        task_lower = task_description.lower()

        if "pick" in task_lower and "place" in task_lower:
            # Extract object names (simplified)
            subtasks = [
                {
                    "instruction": f"Pick up object from task",
                    "object_grasped": "unknown",  # Will be resolved via grounding
                    "object_unattached": None,
                    "motion_direction": "upward"
                },
                {
                    "instruction": f"Place object at target",
                    "object_grasped": "unknown",
                    "object_unattached": "target_region",
                    "motion_direction": "downward"
                }
            ]
            return subtasks

        return []

    def _get_motion_context(
        self,
        subtask_id: int,
        subtask: Dict[str, Any],
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        world_state: Any,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
    ) -> MOKAContext:
        """Get 3D motion context for a subtask using MOKA.

        This involves:
        1. Object detection & segmentation (GroundingDINO + SAM)
        2. Keypoint extraction from masks
        3. VLM selection of grasp/target keypoints and waypoints
        4. 3D unprojection using depth + camera params
        """
        # TODO: Call actual MOKA vision + VLM pipeline
        # For now, return placeholder context

        print(f"[MOKAPlanner] TODO: Get MOKA motion context for subtask {subtask_id}")

        # Placeholder: return empty context
        return MOKAContext(
            subtask_id=subtask_id,
            instruction=subtask.get("instruction", ""),
            object_grasped=subtask.get("object_grasped", "unknown"),
            object_unattached=subtask.get("object_unattached"),
            motion_direction=subtask.get("motion_direction", "upward"),
            grasp_keypoint_2d=None,
            target_keypoint_2d=None,
            grasp_pose_3d=None,
            target_pose_3d=None,
            pre_contact_waypoint_3d=None,
            post_contact_waypoint_3d=None,
            grasp_yaw=None,
            target_tile=None,
        )

    def _contexts_to_skills(
        self,
        moka_contexts: List[MOKAContext],
        world_state: Any
    ) -> List[Dict[str, Any]]:
        """Convert MOKA motion contexts to brain_robot skill sequence.

        Args:
            moka_contexts: List of MOKAContext with 3D motion info
            world_state: Current world state

        Returns:
            List of skill specifications compatible with brain_robot
        """
        skills = []

        for ctx in moka_contexts:
            motion_dir = ctx.motion_direction.lower()

            if motion_dir in ["upward", "grasp", "pick"]:
                # Grasp sequence: Approach → Grasp → Lift

                # 1. Approach with MOKA pre-contact waypoint
                if ctx.pre_contact_waypoint_3d is not None:
                    skills.append({
                        "skill": "ApproachSkill",
                        "args": {
                            "obj": ctx.object_grasped,
                            "waypoint": ctx.pre_contact_waypoint_3d,  # MOKA waypoint
                        }
                    })
                else:
                    # Fallback to default approach
                    skills.append({
                        "skill": "ApproachSkill",
                        "args": {"obj": ctx.object_grasped}
                    })

                # 2. Grasp with MOKA keypoint
                grasp_args = {"obj": ctx.object_grasped}
                if ctx.grasp_pose_3d is not None:
                    grasp_args["grasp_point"] = ctx.grasp_pose_3d[:3]  # [x, y, z]
                if ctx.grasp_yaw is not None:
                    grasp_args["grasp_yaw"] = ctx.grasp_yaw

                skills.append({
                    "skill": "GraspSkill",
                    "args": grasp_args
                })

                # 3. Lift with MOKA post-contact waypoint
                if ctx.post_contact_waypoint_3d is not None:
                    skills.append({
                        "skill": "MoveSkill",
                        "args": {
                            "obj": ctx.object_grasped,
                            "waypoint": ctx.post_contact_waypoint_3d,
                        }
                    })

            elif motion_dir in ["downward", "place"]:
                # Place sequence: Move to pre-contact → Place

                # 1. Move to pre-contact waypoint above target
                if ctx.pre_contact_waypoint_3d is not None:
                    skills.append({
                        "skill": "MoveSkill",
                        "args": {
                            "obj": ctx.object_grasped,
                            "region": ctx.object_unattached,
                            "waypoint": ctx.pre_contact_waypoint_3d,
                        }
                    })

                # 2. Place at target keypoint
                place_args = {
                    "obj": ctx.object_grasped,
                    "region": ctx.object_unattached,
                }
                if ctx.target_pose_3d is not None:
                    place_args["place_point"] = ctx.target_pose_3d[:3]

                skills.append({
                    "skill": "PlaceSkill",
                    "args": place_args
                })

            elif motion_dir in ["pull", "push"]:
                # TODO: Handle drawer/door manipulation
                print(f"[MOKAPlanner] Warning: {motion_dir} motion not yet implemented")

        return skills

    def _get_rgb_observation(self, env: Any) -> np.ndarray:
        """Get RGB image from environment."""
        # LIBERO environment rendering
        if hasattr(env, 'sim'):
            rgb = env.sim.render(
                camera_name="agentview",
                height=256,
                width=256,
                depth=False
            )
            # Convert from (H, W, 3) uint8 to float if needed
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            return rgb

        raise ValueError("Cannot get RGB observation from environment")

    def _get_depth_observation(self, env: Any) -> np.ndarray:
        """Get depth image from environment."""
        if hasattr(env, 'sim'):
            depth = env.sim.render(
                camera_name="agentview",
                height=256,
                width=256,
                depth=True
            )[1]  # depth is second element of tuple
            return depth

        raise ValueError("Cannot get depth observation from environment")

    def _get_camera_params(
        self,
        env: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera intrinsics and extrinsics from environment.

        Returns:
            (intrinsics, extrinsics) - both as numpy arrays
        """
        # TODO: Extract from LIBERO environment
        # For now, use default values

        # Default intrinsics for 256x256 image
        K = np.array([
            [128.0, 0.0, 128.0],
            [0.0, 128.0, 128.0],
            [0.0, 0.0, 1.0]
        ])

        # Default extrinsics (identity - camera at origin)
        T = np.eye(4)

        return K, T

    def _fallback_plan(
        self,
        task_description: str,
        world_state: Any
    ) -> Dict[str, Any]:
        """Simple fallback planning when MOKA unavailable."""
        print("[MOKAPlanner] Using fallback planning (MOKA unavailable)")

        # Simple pick-place decomposition
        skills = [
            {"skill": "ApproachSkill", "args": {"obj": "source"}},
            {"skill": "GraspSkill", "args": {"obj": "source"}},
            {"skill": "MoveSkill", "args": {"obj": "source", "region": "target"}},
            {"skill": "PlaceSkill", "args": {"obj": "source", "region": "target"}},
        ]

        return {
            "success": True,
            "plan": skills,
            "moka_contexts": [],
            "fallback": True,
        }
