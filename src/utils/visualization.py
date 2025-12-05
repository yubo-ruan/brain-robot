"""Debug visualization utilities.

Helper functions for visualizing poses and debugging skill failures.
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path


def debug_plot_poses(
    gripper_pose: np.ndarray,
    object_poses: Dict[str, np.ndarray],
    target_pose: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Quick 3D scatter plot for debugging skill failures.
    
    Args:
        gripper_pose: 7D gripper pose (pos + quat).
        object_poses: Dictionary mapping object names to 7D poses.
        target_pose: Optional target pose to highlight (7D).
        save_path: If provided, save plot to this path.
        title: Optional title for the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Gripper (blue)
    ax.scatter(*gripper_pose[:3], c='blue', s=100, label='gripper', marker='o')
    
    # Add gripper orientation arrow
    # Simplified: just show forward direction
    if len(gripper_pose) >= 7:
        # Convert quaternion to forward direction (simplified)
        ax.quiver(
            gripper_pose[0], gripper_pose[1], gripper_pose[2],
            0, 0, -0.05,  # Point downward (typical gripper orientation)
            color='blue', alpha=0.5
        )
    
    # Objects (green)
    for name, pose in object_poses.items():
        ax.scatter(*pose[:3], c='green', s=80, marker='s')
        ax.text(pose[0], pose[1], pose[2] + 0.02, name, fontsize=8, ha='center')
    
    # Target (red X)
    if target_pose is not None:
        ax.scatter(*target_pose[:3], c='red', s=150, marker='x', label='target', linewidths=3)
    
    # Set labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left')
    
    if title:
        ax.set_title(title)
    
    # Set reasonable axis limits based on data
    all_points = [gripper_pose[:3]]
    all_points.extend([p[:3] for p in object_poses.values()])
    if target_pose is not None:
        all_points.append(target_pose[:3])
    all_points = np.array(all_points)
    
    center = all_points.mean(axis=0)
    max_range = np.abs(all_points - center).max() + 0.1
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def debug_plot_trajectory(
    gripper_positions: np.ndarray,
    object_poses: Dict[str, np.ndarray],
    target_pose: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot gripper trajectory in 3D.
    
    Args:
        gripper_positions: Nx3 array of gripper positions over time.
        object_poses: Dictionary mapping object names to 7D poses (initial).
        target_pose: Optional target pose (7D).
        save_path: If provided, save plot to this path.
        title: Optional title for the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trajectory line
    ax.plot(
        gripper_positions[:, 0],
        gripper_positions[:, 1],
        gripper_positions[:, 2],
        c='blue', linewidth=2, alpha=0.7, label='trajectory'
    )
    
    # Start point
    ax.scatter(*gripper_positions[0], c='green', s=100, marker='o', label='start')
    
    # End point
    ax.scatter(*gripper_positions[-1], c='blue', s=100, marker='o', label='end')
    
    # Objects
    for name, pose in object_poses.items():
        ax.scatter(*pose[:3], c='orange', s=80, marker='s')
        ax.text(pose[0], pose[1], pose[2] + 0.02, name, fontsize=8, ha='center')
    
    # Target
    if target_pose is not None:
        ax.scatter(*target_pose[:3], c='red', s=150, marker='x', label='target', linewidths=3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left')
    
    if title:
        ax.set_title(title)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
