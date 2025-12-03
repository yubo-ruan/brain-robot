"""IK feasibility checking for approach pose selection.

Uses Jacobian-based IK to verify if a target end-effector pose
is kinematically reachable before attempting the approach.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import copy


class IKFeasibilityChecker:
    """Check if target poses are kinematically feasible.

    Uses iterative Jacobian-based IK to test reachability.
    This is more reliable than analytic IK for complex poses.
    """

    # Panda arm joint limits (radians)
    JOINT_LIMITS = {
        'robot0_joint1': (-2.8973, 2.8973),
        'robot0_joint2': (-1.7628, 1.7628),
        'robot0_joint3': (-2.8973, 2.8973),
        'robot0_joint4': (-3.0718, -0.0698),
        'robot0_joint5': (-2.8973, 2.8973),
        'robot0_joint6': (-0.0175, 3.7525),
        'robot0_joint7': (-2.8973, 2.8973),
    }

    # Joint names in order
    JOINT_NAMES = [
        'robot0_joint1', 'robot0_joint2', 'robot0_joint3', 'robot0_joint4',
        'robot0_joint5', 'robot0_joint6', 'robot0_joint7',
    ]

    def __init__(
        self,
        max_iterations: int = 100,
        pos_threshold: float = 0.01,  # 1cm position accuracy
        step_size: float = 0.1,
        damping: float = 0.01,
    ):
        """Initialize IK checker.

        Args:
            max_iterations: Maximum IK iterations
            pos_threshold: Position error threshold for success (meters)
            step_size: IK step size (learning rate)
            damping: Damping factor for damped least squares
        """
        self.max_iterations = max_iterations
        self.pos_threshold = pos_threshold
        self.step_size = step_size
        self.damping = damping

        # Cache for sim/model references
        self._sim = None
        self._model = None
        self._raw_model = None  # Raw mujoco model for jacobian
        self._raw_data = None   # Raw mujoco data for jacobian
        self._joint_ids = None
        self._site_id = None

    def _init_from_env(self, env) -> bool:
        """Initialize from environment, caching model info.

        Args:
            env: LIBERO environment

        Returns:
            True if initialization succeeded
        """
        try:
            self._sim = env.sim
            self._model = self._sim.model

            # Get raw mujoco objects for jacobian computation
            # Robosuite wraps mujoco objects, we need the underlying ones
            if hasattr(self._model, '_model'):
                self._raw_model = self._model._model
                self._raw_data = self._sim.data._data
            else:
                self._raw_model = self._model
                self._raw_data = self._sim.data

            # Get joint IDs
            self._joint_ids = []
            for jname in self.JOINT_NAMES:
                try:
                    jid = self._model.joint(jname).id
                    self._joint_ids.append(jid)
                except ValueError:
                    return False

            # Get EE site ID
            try:
                self._site_id = self._model.site('gripper0_grip_site').id
            except ValueError:
                # Try alternative names
                site_names = [self._model.site(i).name for i in range(self._model.nsite)]
                grip_sites = [s for s in site_names if 'grip' in s.lower()]
                if grip_sites:
                    self._site_id = self._model.site(grip_sites[0]).id
                else:
                    return False

            return True
        except Exception:
            return False

    def _get_current_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        qpos = np.zeros(7)
        for i, jid in enumerate(self._joint_ids):
            qpos[i] = self._sim.data.qpos[self._model.jnt_qposadr[jid]]
        return qpos

    def _get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        return self._sim.data.site_xpos[self._site_id].copy()

    def _get_jacobian(self) -> np.ndarray:
        """Compute position Jacobian for the EE site.

        Returns:
            3x7 Jacobian matrix (position only, no orientation)
        """
        import mujoco

        # Allocate Jacobian matrices (use raw model's nv)
        nv = self._raw_model.nv
        jacp = np.zeros((3, nv))  # Position Jacobian
        jacr = np.zeros((3, nv))  # Rotation Jacobian

        # Compute Jacobian using raw mujoco objects
        mujoco.mj_jacSite(self._raw_model, self._raw_data, jacp, jacr, self._site_id)

        # Extract columns for our joints
        J = np.zeros((3, 7))
        for i, jid in enumerate(self._joint_ids):
            dof_adr = self._model.jnt_dofadr[jid]
            J[:, i] = jacp[:, dof_adr]

        return J

    def _clamp_to_limits(self, qpos: np.ndarray) -> np.ndarray:
        """Clamp joint positions to limits."""
        clamped = qpos.copy()
        for i, jname in enumerate(self.JOINT_NAMES):
            lo, hi = self.JOINT_LIMITS[jname]
            clamped[i] = np.clip(clamped[i], lo, hi)
        return clamped

    def _is_within_limits(self, qpos: np.ndarray, margin: float = 0.05) -> bool:
        """Check if joint positions are within limits with margin."""
        for i, jname in enumerate(self.JOINT_NAMES):
            lo, hi = self.JOINT_LIMITS[jname]
            if qpos[i] < lo + margin or qpos[i] > hi - margin:
                return False
        return True

    def check_feasibility(
        self,
        env,
        target_pos: np.ndarray,
        return_solution: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if target position is kinematically feasible.

        Uses Jacobian-based IK without modifying the actual simulation.

        Args:
            env: LIBERO environment
            target_pos: Target EE position [x, y, z]
            return_solution: If True, return the IK solution

        Returns:
            Tuple of (feasible, info_dict)
        """
        info = {
            'target_pos': target_pos.tolist(),
            'iterations': 0,
            'final_error': float('inf'),
        }

        # Initialize from environment
        if not self._init_from_env(env):
            info['error'] = 'Failed to initialize from environment'
            return False, info

        # Save current state
        initial_qpos = self._get_current_joint_positions()
        current_ee = self._get_ee_position()

        info['initial_ee_pos'] = current_ee.tolist()
        info['initial_error'] = float(np.linalg.norm(target_pos - current_ee))

        # If already at target, trivially feasible
        if np.linalg.norm(target_pos - current_ee) < self.pos_threshold:
            info['final_error'] = info['initial_error']
            info['solution'] = initial_qpos.tolist() if return_solution else None
            return True, info

        # Create a copy of qpos for IK iteration
        # We'll use forward kinematics to test without modifying sim state
        qpos = initial_qpos.copy()

        # Iterative IK using damped least squares
        for iteration in range(self.max_iterations):
            # Temporarily set joint positions to compute FK
            for i, jid in enumerate(self._joint_ids):
                self._sim.data.qpos[self._model.jnt_qposadr[jid]] = qpos[i]

            # Forward kinematics (use robosuite's sim.forward() wrapper)
            self._sim.forward()

            # Get current EE position
            current_ee = self._get_ee_position()
            error = target_pos - current_ee
            error_norm = np.linalg.norm(error)

            if error_norm < self.pos_threshold:
                # Success - restore original state
                for i, jid in enumerate(self._joint_ids):
                    self._sim.data.qpos[self._model.jnt_qposadr[jid]] = initial_qpos[i]
                self._sim.forward()

                info['iterations'] = iteration + 1
                info['final_error'] = float(error_norm)
                info['solution'] = qpos.tolist() if return_solution else None
                info['within_limits'] = self._is_within_limits(qpos)
                return True, info

            # Compute Jacobian
            J = self._get_jacobian()

            # Damped least squares: dq = J^T (J J^T + λI)^-1 * error
            JJT = J @ J.T
            damped = JJT + self.damping * np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(damped, error)
            except np.linalg.LinAlgError:
                # Singular matrix - target may be unreachable
                break

            # Update joint positions
            qpos = qpos + self.step_size * dq
            qpos = self._clamp_to_limits(qpos)

        # Restore original state
        for i, jid in enumerate(self._joint_ids):
            self._sim.data.qpos[self._model.jnt_qposadr[jid]] = initial_qpos[i]
        self._sim.forward()

        info['iterations'] = self.max_iterations
        info['final_error'] = float(error_norm) if 'error_norm' in dir() else float('inf')
        return False, info

    def find_best_approach(
        self,
        env,
        approach_candidates: List[Tuple[str, np.ndarray]],
        min_margin: float = 0.02,
    ) -> Tuple[Optional[str], np.ndarray, Dict[str, Any]]:
        """Find the best feasible approach from candidates.

        Args:
            env: LIBERO environment
            approach_candidates: List of (name, target_pos) tuples
            min_margin: Minimum distance from joint limits

        Returns:
            Tuple of (best_name, best_pos, info_dict)
            Returns (None, zeros, info) if none are feasible
        """
        info = {
            'candidates_tested': len(approach_candidates),
            'results': {},
        }

        best_name = None
        best_pos = np.zeros(3)
        best_error = float('inf')

        for name, target_pos in approach_candidates:
            feasible, result = self.check_feasibility(env, target_pos)
            info['results'][name] = {
                'feasible': feasible,
                'final_error': result.get('final_error', float('inf')),
                'iterations': result.get('iterations', 0),
            }

            if feasible:
                # Prefer solutions with lower final error
                if result['final_error'] < best_error:
                    best_name = name
                    best_pos = target_pos
                    best_error = result['final_error']

        info['best_approach'] = best_name
        return best_name, best_pos, info


def check_approach_feasibility(
    env,
    target_pos: np.ndarray,
    approach_strategies: List[str] = None,
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """High-level function to check approach feasibility.

    Args:
        env: LIBERO environment
        target_pos: Object position to approach
        approach_strategies: List of strategy names to try

    Returns:
        Tuple of (best_strategy, approach_pos, info)
    """
    from .approach_selection import (
        APPROACH_DIRECTIONS,
        APPROACH_ORIENTATIONS,
        compute_angled_pregrasp_pose,
    )

    if approach_strategies is None:
        approach_strategies = ['top_down', 'front_angled', 'front_angled_steep']

    # Build candidate positions
    candidates = []
    pregrasp_distance = 0.12  # Standard pregrasp distance

    for strategy in approach_strategies:
        if strategy not in APPROACH_DIRECTIONS:
            continue

        approach_dir = APPROACH_DIRECTIONS[strategy]
        gripper_ori = APPROACH_ORIENTATIONS[strategy]

        # Compute pregrasp position
        obj_pose = np.zeros(7)
        obj_pose[:3] = target_pos
        obj_pose[3:] = gripper_ori

        pregrasp = compute_angled_pregrasp_pose(
            obj_pose, approach_dir, gripper_ori, pregrasp_distance
        )
        candidates.append((strategy, pregrasp[:3]))

    # Check feasibility
    checker = IKFeasibilityChecker()
    return checker.find_best_approach(env, candidates)
