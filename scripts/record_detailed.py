#!/usr/bin/env python3
"""
Create detailed visualization with annotations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from src.env.mock_env import make_mock_env


def get_expert_action(env):
    """Scripted expert policy."""
    ee_pos = env.robot_pos
    obj_pos = env.object_pos
    target_pos = env.target_pos
    dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_to_target = np.linalg.norm(obj_pos[:2] - target_pos[:2])

    if not env.object_grasped:
        if dist_to_obj > 0.08:
            direction = obj_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = -1.0
            phase = "Approach Object"
        else:
            action = np.zeros(7)
            action[6] = 1.0
            phase = "Grasp Object"
    else:
        if dist_to_target > 0.08:
            direction = target_pos - ee_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            action = np.zeros(7)
            action[:3] = direction * 1.0
            action[6] = 1.0
            phase = "Transport to Target"
        else:
            action = np.zeros(7)
            action[6] = -1.0
            phase = "Release Object"

    return action, phase


def create_detailed_gif(output_path):
    """Create detailed visualization with matplotlib."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    obs, _ = env.reset()

    # Collect trajectory
    trajectory = []
    for step in range(100):
        action, phase = get_expert_action(env)

        state = {
            'step': step,
            'robot_pos': env.robot_pos.copy(),
            'object_pos': env.object_pos.copy(),
            'target_pos': env.target_pos.copy(),
            'gripper_open': env.gripper_open,
            'object_grasped': env.object_grasped,
            'phase': phase,
        }
        trajectory.append(state)

        _, _, done, truncated, info = env.step(action)
        if done:
            # Add final success state
            state = {
                'step': step + 1,
                'robot_pos': env.robot_pos.copy(),
                'object_pos': env.object_pos.copy(),
                'target_pos': env.target_pos.copy(),
                'gripper_open': env.gripper_open,
                'object_grasped': env.object_grasped,
                'phase': "SUCCESS!",
                'success': True,
            }
            trajectory.append(state)
            break

    env.close()

    # Create animation
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def animate(frame_idx):
        ax.clear()

        state = trajectory[min(frame_idx, len(trajectory) - 1)]

        # Set limits
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.1, 0.6)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)

        # Draw workspace boundary
        ax.add_patch(patches.Rectangle((-0.5, 0), 1.0, 0.5, fill=False,
                                        edgecolor='gray', linestyle='--', linewidth=2))

        # Draw target (green circle)
        target = plt.Circle((state['target_pos'][0], state['target_pos'][1]),
                            0.05, color='green', alpha=0.5, label='Target')
        ax.add_patch(target)
        ax.plot(state['target_pos'][0], state['target_pos'][1], 'g+', markersize=15, mew=3)

        # Draw object (red circle)
        obj_color = 'orange' if state['object_grasped'] else 'red'
        obj = plt.Circle((state['object_pos'][0], state['object_pos'][1]),
                         0.04, color=obj_color, label='Object')
        ax.add_patch(obj)

        # Draw robot gripper
        robot_color = 'blue' if state['gripper_open'] else 'purple'
        gripper_size = 0.03 if state['gripper_open'] else 0.02

        # Draw gripper as two rectangles
        gripper_width = 0.015
        gripper_gap = gripper_size if state['gripper_open'] else 0.005

        left_gripper = patches.Rectangle(
            (state['robot_pos'][0] - gripper_gap - gripper_width,
             state['robot_pos'][1] - 0.02),
            gripper_width, 0.04, color=robot_color
        )
        right_gripper = patches.Rectangle(
            (state['robot_pos'][0] + gripper_gap,
             state['robot_pos'][1] - 0.02),
            gripper_width, 0.04, color=robot_color
        )
        ax.add_patch(left_gripper)
        ax.add_patch(right_gripper)

        # Draw gripper base
        base = patches.Rectangle(
            (state['robot_pos'][0] - 0.02, state['robot_pos'][1] + 0.02),
            0.04, 0.02, color='gray'
        )
        ax.add_patch(base)

        # Title with phase
        phase_colors = {
            'Approach Object': 'orange',
            'Grasp Object': 'red',
            'Transport to Target': 'blue',
            'Release Object': 'green',
            'SUCCESS!': 'green',
        }
        title_color = phase_colors.get(state['phase'], 'black')

        ax.set_title(f"Step {state['step']}: {state['phase']}",
                    fontsize=16, fontweight='bold', color=title_color)

        # Add legend
        legend_elements = [
            patches.Patch(facecolor='red', label='Object'),
            patches.Patch(facecolor='green', alpha=0.5, label='Target'),
            patches.Patch(facecolor='blue', label='Gripper (Open)'),
            patches.Patch(facecolor='purple', label='Gripper (Closed)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Add info text
        info_text = f"Robot: ({state['robot_pos'][0]:.2f}, {state['robot_pos'][1]:.2f})\n"
        info_text += f"Object: ({state['object_pos'][0]:.2f}, {state['object_pos'][1]:.2f})\n"
        info_text += f"Grasped: {state['object_grasped']}"
        ax.text(-0.55, 0.55, info_text, fontsize=10, verticalalignment='top',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Success indicator
        if state.get('success', False):
            ax.text(0, 0.3, "✓ SUCCESS!", fontsize=30, ha='center', va='center',
                   color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax.grid(True, alpha=0.3)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(trajectory) + 10,
                        interval=100, repeat=True)

    # Save as GIF
    print(f"Saving detailed GIF with {len(trajectory)} frames...")
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved to {output_path}")


def create_trajectory_plot(output_path):
    """Create static trajectory plot."""
    env = make_mock_env(max_episode_steps=100, action_scale=1.0)
    obs, _ = env.reset()

    robot_trajectory = []
    object_trajectory = []

    for step in range(100):
        action, phase = get_expert_action(env)
        robot_trajectory.append(env.robot_pos.copy())
        object_trajectory.append(env.object_pos.copy())

        _, _, done, _, _ = env.step(action)
        if done:
            robot_trajectory.append(env.robot_pos.copy())
            object_trajectory.append(env.object_pos.copy())
            break

    target_pos = env.target_pos.copy()
    env.close()

    robot_trajectory = np.array(robot_trajectory)
    object_trajectory = np.array(object_trajectory)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Robot trajectory
    ax.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], 'b-', linewidth=2,
           label='Gripper Path', alpha=0.7)
    ax.scatter(robot_trajectory[0, 0], robot_trajectory[0, 1], c='blue', s=100,
              marker='o', zorder=5, label='Gripper Start')
    ax.scatter(robot_trajectory[-1, 0], robot_trajectory[-1, 1], c='blue', s=100,
              marker='s', zorder=5, label='Gripper End')

    # Object trajectory
    ax.plot(object_trajectory[:, 0], object_trajectory[:, 1], 'r--', linewidth=2,
           label='Object Path', alpha=0.7)
    ax.scatter(object_trajectory[0, 0], object_trajectory[0, 1], c='red', s=100,
              marker='o', zorder=5, label='Object Start')

    # Target
    ax.scatter(target_pos[0], target_pos[1], c='green', s=200, marker='*',
              zorder=5, label='Target')

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.1, 0.6)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Pick-and-Place Task Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    output_dir = "/workspace/src/recordings"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Creating Detailed Visualizations")
    print("=" * 60)

    # Create detailed GIF
    print("\n1. Creating detailed animation...")
    create_detailed_gif(os.path.join(output_dir, "detailed_episode.gif"))

    # Create trajectory plot
    print("\n2. Creating trajectory plot...")
    create_trajectory_plot(os.path.join(output_dir, "trajectory_plot.png"))

    print(f"\nAll visualizations saved to {output_dir}/")
