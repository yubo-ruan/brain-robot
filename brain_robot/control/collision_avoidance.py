"""Collision-aware path planning for constrained locations.

Generates waypoint-based trajectories that avoid obstacles when reaching
objects inside drawers, on cabinet tops, or other constrained locations.
"""

import numpy as np
from typing import List, Optional, Tuple


def compute_waypoints_for_drawer(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    drawer_front_y: Optional[float] = None,
    clearance_height: float = 0.15,
) -> List[np.ndarray]:
    """Compute waypoints to reach inside a drawer without collision.

    Strategy for drawer: HORIZONTAL approach
    1. Move to same height as object (drawer level)
    2. Position in front of drawer opening
    3. Move horizontally INTO the drawer toward object

    Args:
        start_pos: Current gripper position [x, y, z]
        target_pos: Target position inside drawer [x, y, z]
        drawer_front_y: Y position of drawer front edge (if known)
        clearance_height: Not used for horizontal drawer approach

    Returns:
        List of waypoint positions to traverse
    """
    waypoints = []

    # Estimate drawer front Y if not provided (typically 8-10cm in front of object)
    if drawer_front_y is None:
        drawer_front_y = target_pos[1] + 0.10

    # Object height (same as drawer interior height)
    obj_height = target_pos[2]

    # Waypoint 1: Move to object height at current XY (or slightly above)
    # Stay slightly above to avoid hitting drawer bottom during transit
    approach_height = obj_height + 0.03
    wp1 = np.array([start_pos[0], start_pos[1], approach_height])
    waypoints.append(wp1)

    # Waypoint 2: Move to in front of drawer opening at approach height
    # Position at X of object, Y in front of drawer, Z at approach height
    wp2 = np.array([target_pos[0], drawer_front_y, approach_height])
    waypoints.append(wp2)

    # Waypoint 3: Move horizontally into drawer, closer to object
    # This is still in front of the object by a small margin for final approach
    wp3 = np.array([target_pos[0], target_pos[1] + 0.04, approach_height])
    waypoints.append(wp3)

    # Final target is added by caller (will be the actual pregrasp position)
    return waypoints


def compute_waypoints_for_cabinet_top(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    cabinet_edge_y: Optional[float] = None,
    clearance_height: float = 0.10,
) -> List[np.ndarray]:
    """Compute waypoints to reach object on cabinet top.

    Strategy: Approach from front at height, then descend

    Args:
        start_pos: Current gripper position [x, y, z]
        target_pos: Target position on cabinet [x, y, z]
        cabinet_edge_y: Y position of cabinet front edge (if known)
        clearance_height: Height above target for safe approach

    Returns:
        List of waypoint positions to traverse
    """
    waypoints = []

    # Cabinet tops are high - approach from front at same height or higher
    approach_height = max(target_pos[2] + clearance_height, start_pos[2])

    # Estimate cabinet edge if not provided
    if cabinet_edge_y is None:
        cabinet_edge_y = target_pos[1] + 0.12  # Cabinet tops are deep

    # Waypoint 1: Move to approach height at current XY
    wp1 = np.array([start_pos[0], start_pos[1], approach_height])
    waypoints.append(wp1)

    # Waypoint 2: Move forward to front of cabinet at approach height
    wp2 = np.array([target_pos[0], cabinet_edge_y, approach_height])
    waypoints.append(wp2)

    # Waypoint 3: Move to above target at approach height
    wp3 = np.array([target_pos[0], target_pos[1], approach_height])
    waypoints.append(wp3)

    return waypoints


def compute_waypoints_for_elevated_surface(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    clearance_height: float = 0.08,
) -> List[np.ndarray]:
    """Compute waypoints for elevated surfaces (stove, cookie box).

    Strategy: Simple lift -> move XY -> descend

    Args:
        start_pos: Current gripper position [x, y, z]
        target_pos: Target position [x, y, z]
        clearance_height: Height above target for safe approach

    Returns:
        List of waypoint positions to traverse
    """
    waypoints = []

    # Approach height
    approach_height = target_pos[2] + clearance_height

    # Waypoint 1: Lift to approach height
    if start_pos[2] < approach_height:
        wp1 = np.array([start_pos[0], start_pos[1], approach_height])
        waypoints.append(wp1)

    # Waypoint 2: Move to above target
    wp2 = np.array([target_pos[0], target_pos[1], approach_height])
    waypoints.append(wp2)

    return waypoints


def generate_collision_aware_path(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    in_drawer: bool = False,
    on_cabinet: bool = False,
    on_elevated: bool = False,
    drawer_front_y: Optional[float] = None,
    cabinet_edge_y: Optional[float] = None,
) -> List[np.ndarray]:
    """Generate collision-aware path from start to target.

    Args:
        start_pos: Current gripper position [x, y, z]
        target_pos: Target position [x, y, z]
        in_drawer: Whether target is inside a drawer
        on_cabinet: Whether target is on a cabinet top
        on_elevated: Whether target is on an elevated surface
        drawer_front_y: Y position of drawer front (if known)
        cabinet_edge_y: Y position of cabinet edge (if known)

    Returns:
        List of waypoint positions including final target
    """
    waypoints = []

    # Priority order: drawer > cabinet > elevated > direct
    if in_drawer:
        waypoints = compute_waypoints_for_drawer(
            start_pos, target_pos, drawer_front_y
        )
    elif on_cabinet:
        waypoints = compute_waypoints_for_cabinet_top(
            start_pos, target_pos, cabinet_edge_y
        )
    elif on_elevated:
        waypoints = compute_waypoints_for_elevated_surface(
            start_pos, target_pos
        )

    # Always add final target
    waypoints.append(target_pos.copy())

    return waypoints


def is_waypoint_reached(
    current_pos: np.ndarray,
    waypoint: np.ndarray,
    threshold: float = 0.04,
) -> bool:
    """Check if gripper has reached a waypoint.

    Args:
        current_pos: Current gripper position [x, y, z]
        waypoint: Waypoint position [x, y, z]
        threshold: Distance threshold for reaching waypoint

    Returns:
        True if within threshold distance
    """
    return np.linalg.norm(current_pos - waypoint) < threshold
