"""Oracle perception using ground truth from simulator.

Uses privileged access to robosuite/LIBERO to get perfect object poses.
No learning involved - this is for validating the architecture.
"""

import time
import numpy as np
from typing import Optional

from .interface import PerceptionInterface, PerceptionResult


class OraclePerception(PerceptionInterface):
    """Oracle perception using ground truth from simulator.
    
    Extracts poses directly from the MuJoCo simulation state.
    All poses are returned in world frame.
    """
    
    def __init__(self):
        self._last_perceive_time = 0.0
    
    def perceive(self, env) -> PerceptionResult:
        """Extract ground truth perception from environment.

        Args:
            env: The robosuite/LIBERO environment.

        Returns:
            PerceptionResult with ground truth poses.
        """
        result = PerceptionResult(timestamp=time.time())

        # Get the underlying sim/env
        sim = self._get_sim(env)
        base_env = self._get_base_env(env)

        if sim is None:
            return result

        # Extract object poses
        result.objects, result.object_names = self._extract_object_poses(sim, base_env)

        # Extract spatial relations from poses
        result.on, result.inside = self._extract_spatial_relations(result.objects)

        # Extract gripper state
        result.gripper_pose = self._extract_gripper_pose(sim, base_env, env)
        result.gripper_width = self._extract_gripper_width(sim, base_env, env)

        # Extract joint state
        result.joint_positions, result.joint_velocities = self._extract_joint_state(sim, base_env)

        self._last_perceive_time = result.timestamp
        return result
    
    def _get_sim(self, env):
        """Get MuJoCo sim object from wrapped environment."""
        # Handle various wrapping levels
        if hasattr(env, 'sim'):
            return env.sim
        if hasattr(env, '_env'):
            return self._get_sim(env._env)
        if hasattr(env, 'env'):
            return self._get_sim(env.env)
        return None
    
    def _get_base_env(self, env):
        """Get base robosuite environment."""
        if hasattr(env, 'robots'):
            return env
        if hasattr(env, '_env'):
            return self._get_base_env(env._env)
        if hasattr(env, 'env'):
            return self._get_base_env(env.env)
        return env
    
    def _extract_object_poses(self, sim, base_env) -> tuple:
        """Extract object poses from simulation.
        
        Returns:
            Tuple of (poses_dict, object_names_list).
        """
        objects = {}
        object_names = []
        
        # Try to get object names from environment
        if hasattr(base_env, 'object_names'):
            obj_names = base_env.object_names
        elif hasattr(base_env, '_obj_names'):
            obj_names = base_env._obj_names
        else:
            # Fall back to searching model body names
            obj_names = self._find_object_bodies(sim)
        
        for name in obj_names:
            pose = self._get_body_pose(sim, name)
            if pose is not None:
                objects[name] = pose
                object_names.append(name)
        
        return objects, object_names
    
    def _find_object_bodies(self, sim) -> list:
        """Find object bodies in MuJoCo model."""
        object_names = []
        
        # Common object naming patterns in LIBERO
        skip_patterns = ['robot', 'gripper', 'world', 'floor', 'table', 'mount']
        
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name and not any(skip in name.lower() for skip in skip_patterns):
                # Additional filter: objects typically have a geom
                if self._body_has_geom(sim, i):
                    object_names.append(name)
        
        return object_names
    
    def _body_has_geom(self, sim, body_id: int) -> bool:
        """Check if body has at least one geom."""
        for i in range(sim.model.ngeom):
            if sim.model.geom_bodyid[i] == body_id:
                return True
        return False
    
    def _get_body_pose(self, sim, body_name: str) -> Optional[np.ndarray]:
        """Get 7D pose of named body.
        
        Returns:
            [x, y, z, qw, qx, qy, qz] or None if not found.
        """
        try:
            body_id = sim.model.body_name2id(body_name)
            pos = sim.data.body_xpos[body_id].copy()
            # MuJoCo uses xquat format: [w, x, y, z]
            quat = sim.data.body_xquat[body_id].copy()
            return np.concatenate([pos, quat])
        except (KeyError, ValueError):
            return None
    
    def _extract_gripper_pose(self, sim, base_env, env) -> Optional[np.ndarray]:
        """Extract gripper end-effector pose."""
        # Try to get from observation first (most reliable)
        if hasattr(env, '_get_observations'):
            try:
                obs = env._get_observations()
                if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
                    pos = obs['robot0_eef_pos']
                    quat = obs['robot0_eef_quat']
                    return np.concatenate([pos, quat])
            except:
                pass
        
        # Fall back to direct sim access
        eef_names = ['robot0_eef', 'gripper0_eef', 'right_hand', 'robot0_right_hand']
        for name in eef_names:
            pose = self._get_body_pose(sim, name)
            if pose is not None:
                return pose
        
        # Try site instead of body
        eef_sites = ['grip_site', 'gripper0_grip_site', 'robot0_grip_site']
        for site_name in eef_sites:
            try:
                site_id = sim.model.site_name2id(site_name)
                pos = sim.data.site_xpos[site_id].copy()
                # Sites don't have orientation, use identity quaternion
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                return np.concatenate([pos, quat])
            except (KeyError, ValueError):
                continue
        
        return None
    
    def _extract_gripper_width(self, sim, base_env, env) -> float:
        """Extract current gripper opening width."""
        # Try observation first
        if hasattr(env, '_get_observations'):
            try:
                obs = env._get_observations()
                if 'robot0_gripper_qpos' in obs:
                    gripper_qpos = obs['robot0_gripper_qpos']
                    # Gripper width is typically sum of finger positions
                    return float(np.sum(np.abs(gripper_qpos)))
            except:
                pass
        
        # Fall back to robot accessor
        if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
            robot = base_env.robots[0]
            if hasattr(robot, 'gripper') and hasattr(robot.gripper, 'get_gripper_state'):
                return float(robot.gripper.get_gripper_state())
        
        return 0.0
    
    def _extract_joint_state(self, sim, base_env) -> tuple:
        """Extract robot joint positions and velocities.

        Returns:
            Tuple of (joint_positions, joint_velocities) or (None, None).
        """
        if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
            robot = base_env.robots[0]
            if hasattr(robot, '_joint_positions'):
                pos = np.array(robot._joint_positions).copy()
                vel = np.array(robot._joint_velocities).copy() if hasattr(robot, '_joint_velocities') else None
                return pos, vel

        return None, None

    def _extract_spatial_relations(self, objects: dict) -> tuple:
        """Extract spatial relations (on, inside) from object poses.

        Uses geometric heuristics:
        - ON: object A is "on" object B if A is above B and horizontally close
        - INSIDE: object A is "inside" B if A is within B's bounding region (for containers)

        Args:
            objects: Dict of object_name → 7D pose

        Returns:
            Tuple of (on_dict, inside_dict)
        """
        on = {}
        inside = {}

        # Classify objects by type for spatial reasoning
        # Small/movable objects that can be ON or INSIDE something
        movable_patterns = ['bowl', 'mug', 'cup', 'can', 'bottle']
        # Surface objects that things can be ON
        surface_patterns = ['plate', 'ramekin', 'cookies', 'cabinet_top', 'stove', 'box']
        # Container objects that things can be INSIDE
        container_patterns = ['drawer', 'cabinet']

        # Get lists of each type
        movable_objs = []
        surface_objs = []
        container_objs = []

        for name, pose in objects.items():
            name_lower = name.lower()
            if any(p in name_lower for p in movable_patterns):
                movable_objs.append((name, pose))
            if any(p in name_lower for p in surface_patterns):
                surface_objs.append((name, pose))
            if any(p in name_lower for p in container_patterns):
                container_objs.append((name, pose))

        # Thresholds for spatial relations
        ON_HEIGHT_THRESHOLD = 0.15  # Max vertical distance to be "on"
        ON_HORIZONTAL_THRESHOLD = 0.10  # Max horizontal distance to be "on"
        INSIDE_HEIGHT_THRESHOLD = 0.05  # Object below container top surface
        INSIDE_HORIZONTAL_THRESHOLD = 0.15  # Horizontal containment

        # Check ON relations
        for mov_name, mov_pose in movable_objs:
            mov_pos = mov_pose[:3]
            best_surface = None
            best_height_diff = float('inf')

            for surf_name, surf_pose in surface_objs:
                # Skip self
                if surf_name == mov_name:
                    continue

                surf_pos = surf_pose[:3]

                # Check if movable is above surface
                height_diff = mov_pos[2] - surf_pos[2]
                if height_diff < 0:
                    continue  # Movable is below surface

                if height_diff > ON_HEIGHT_THRESHOLD:
                    continue  # Too far above

                # Check horizontal proximity
                horiz_dist = np.sqrt((mov_pos[0] - surf_pos[0])**2 + (mov_pos[1] - surf_pos[1])**2)
                if horiz_dist > ON_HORIZONTAL_THRESHOLD:
                    continue  # Too far horizontally

                # Found a candidate - prefer closest
                if height_diff < best_height_diff:
                    best_height_diff = height_diff
                    best_surface = surf_name

            if best_surface:
                on[mov_name] = best_surface
            else:
                # Default to table if not on any specific surface
                on[mov_name] = "table"

        # Check INSIDE relations (for drawers/cabinets)
        # LIBERO drawer detection: look for _top parts of cabinets
        # and check if bowl is horizontally close to drawer front
        DRAWER_HORIZONTAL_THRESHOLD = 0.05  # Tight horizontal match for drawer
        DRAWER_HEIGHT_RANGE = 0.35  # Bowl can be above cabinet base (drawer is elevated)

        for mov_name, mov_pose in movable_objs:
            mov_pos = mov_pose[:3]

            # Look for cabinet_top parts specifically (drawer front in LIBERO)
            for cont_name, cont_pose in container_objs:
                cont_pos = cont_pose[:3]

                # Check for drawer-specific parts
                if '_top' in cont_name.lower() or 'drawer' in cont_name.lower():
                    # For drawers: check tight horizontal alignment
                    horiz_dist = np.sqrt((mov_pos[0] - cont_pos[0])**2 + (mov_pos[1] - cont_pos[1])**2)
                    height_diff = mov_pos[2] - cont_pos[2]

                    # Bowl should be close horizontally and within drawer height range
                    if horiz_dist < DRAWER_HORIZONTAL_THRESHOLD and 0 < height_diff < DRAWER_HEIGHT_RANGE:
                        # This is likely inside a drawer
                        # Use the base cabinet name for the relation
                        base_name = cont_name.replace('_cabinet_top', '').replace('_top', '')
                        # Find the actual cabinet body name
                        cabinet_names = [n for n in objects.keys() if base_name in n and ('_base' in n or n == base_name + '_main')]
                        if cabinet_names:
                            inside[mov_name] = cabinet_names[0]
                        else:
                            inside[mov_name] = cont_name
                        if mov_name in on:
                            del on[mov_name]
                        break

        return on, inside
