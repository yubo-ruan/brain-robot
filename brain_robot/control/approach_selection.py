"""Multi-directional approach selection for grasping.

Selects optimal approach direction based on object position relative to robot.
This is critical for objects at the edge of the workspace where top-down
grasps are kinematically infeasible.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation as R


# Predefined approach orientations (quaternion [w, x, y, z])
# Note: In robosuite OSC, orientation control is often unreliable.
# We keep the default top-down orientation for all approaches and just
# change the approach POSITION (come from front/side instead of above).
# This works better with the existing PD controller.

# Robosuite default gripper-down orientation
DEFAULT_ORIENTATION = np.array([-0.02, 0.707, 0.707, -0.02])

APPROACH_ORIENTATIONS = {
    # All strategies use the same orientation - only position changes
    "top_down": DEFAULT_ORIENTATION,
    "front_angled": DEFAULT_ORIENTATION,
    "front_angled_steep": DEFAULT_ORIENTATION,
    "front_horizontal": DEFAULT_ORIENTATION,
    "side_left": DEFAULT_ORIENTATION,
    "side_right": DEFAULT_ORIENTATION,
}

# Approach directions (unit vectors pointing FROM pregrasp TO object)
APPROACH_DIRECTIONS = {
    "top_down": np.array([0.0, 0.0, -1.0]),      # Straight down
    "front_angled": np.array([0.0, -0.5, -0.866]),  # Forward + down (30 deg)
    "front_angled_steep": np.array([0.0, -0.707, -0.707]),  # Forward + down (45 deg)
    "front_horizontal": np.array([0.0, -0.95, -0.31]),  # Almost horizontal (18 deg) for drawer
    "side_left": np.array([-1.0, 0.0, 0.0]),     # From +X side
    "side_right": np.array([1.0, 0.0, 0.0]),     # From -X side
}


def select_approach_strategy(
    obj_pos: np.ndarray,
    robot_base_pos: np.ndarray = np.array([-0.5, 0.0, 0.9]),
    obj_height: float = 0.0,
    in_drawer: bool = False,
    on_elevated_surface: bool = False,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Select optimal approach direction based on object position.

    Args:
        obj_pos: Object position [x, y, z]
        robot_base_pos: Robot base position (default for Panda)
        obj_height: Height of object above table (for elevated objects)
        in_drawer: Whether object is inside a drawer
        on_elevated_surface: Whether object is on elevated surface (cabinet, stove)

    Returns:
        Tuple of (strategy_name, approach_direction, gripper_orientation)
    """
    # Compute relative position from robot base
    rel_pos = obj_pos - robot_base_pos
    rel_y = rel_pos[1]  # Y relative to robot (+ is front, - is back)
    rel_x = rel_pos[0]  # X relative to robot (+ is right, - is left)
    obj_z = obj_pos[2]  # Absolute Z height

    # Decision logic based on object position

    # Case 1: Object inside drawer - needs near-horizontal approach to reach inside
    if in_drawer:
        return ("front_horizontal",
                APPROACH_DIRECTIONS["front_horizontal"],
                APPROACH_ORIENTATIONS["front_horizontal"])

    # Case 2: Object on elevated surface (cabinet top, stove) - front angled
    if on_elevated_surface or obj_z > 1.1:
        return ("front_angled",
                APPROACH_DIRECTIONS["front_angled"],
                APPROACH_ORIENTATIONS["front_angled"])

    # Case 3: Object behind robot (negative Y) - front angled approach
    # This includes objects near Y=0 which are still hard to reach top-down
    if rel_y < 0.05:  # Behind or at center line
        # Use steeper angle for objects further back
        if rel_y < -0.15:
            return ("front_angled_steep",
                    APPROACH_DIRECTIONS["front_angled_steep"],
                    APPROACH_ORIENTATIONS["front_angled_steep"])
        else:
            return ("front_angled",
                    APPROACH_DIRECTIONS["front_angled"],
                    APPROACH_ORIENTATIONS["front_angled"])

    # Default: top-down approach (works for most positions)
    return ("top_down",
            APPROACH_DIRECTIONS["top_down"],
            APPROACH_ORIENTATIONS["top_down"])


def compute_angled_pregrasp_pose(
    object_pose: np.ndarray,
    approach_direction: np.ndarray,
    gripper_orientation: np.ndarray,
    pregrasp_distance: float = 0.12,
) -> np.ndarray:
    """Compute pregrasp pose for angled approach.

    Args:
        object_pose: 7D object pose [x, y, z, qw, qx, qy, qz]
        approach_direction: Unit vector pointing from pregrasp to object
        gripper_orientation: Quaternion for gripper orientation
        pregrasp_distance: Distance from object along approach direction

    Returns:
        7D pregrasp pose
    """
    pregrasp = np.zeros(7)

    # Position: offset from object along negative approach direction
    approach_dir = approach_direction / np.linalg.norm(approach_direction)
    pregrasp[:3] = object_pose[:3] - approach_dir * pregrasp_distance

    # Orientation
    pregrasp[3:7] = gripper_orientation

    return pregrasp


def get_approach_info(strategy_name: str) -> Dict[str, Any]:
    """Get detailed info about an approach strategy."""
    return {
        "name": strategy_name,
        "direction": APPROACH_DIRECTIONS.get(strategy_name, APPROACH_DIRECTIONS["top_down"]),
        "orientation": APPROACH_ORIENTATIONS.get(strategy_name, APPROACH_ORIENTATIONS["top_down"]),
        "is_angled": strategy_name != "top_down",
    }
