"""
Phase 1: Episode Replay Viewer

Provides visualization of recorded episodes with:
- Scrub through episode frames
- Overlay MOKA keypoints on images
- Show 3D subgoals projected to 2D
- Phase timeline bar
- Proprioception plots

Can be used as:
1. CLI tool: python -m brain_robot.logging.replay_viewer /path/to/episode.hdf5
2. Library: from brain_robot.logging.replay_viewer import ReplayViewer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse

from .hdf5_episode_logger import load_hdf5_episode, MOKAOutput, get_moka_outputs


# Skill name mapping
SKILL_NAMES = {
    -1: 'Unknown',
    0: 'ApproachObject',
    1: 'GraspObject',
    2: 'MoveObjectToRegion',
    3: 'PlaceObject',
    4: 'OpenDrawer',
    5: 'CloseDrawer',
    6: 'TurnOnStove',
    7: 'TurnOffStove',
}

# Phase name mapping
PHASE_NAMES = {
    -1: 'Unknown',
    0: 'PreApproach',
    1: 'Approach',
    2: 'Contact',
    3: 'Execute',
    4: 'PostExecute',
}

# Colors for visualization
COLORS = {
    'grasp': 'red',
    'function': 'yellow',
    'target': 'blue',
    'pre_waypoint': 'cyan',
    'post_waypoint': 'magenta',
    'trajectory': 'green',
}


class ReplayViewer:
    """Interactive episode replay viewer."""

    def __init__(self, episode_path: str):
        """Initialize viewer with episode file.

        Args:
            episode_path: Path to HDF5 episode file.
        """
        self.episode_path = Path(episode_path)
        self.data = load_hdf5_episode(str(episode_path))

        self.num_frames = self.data['num_timesteps']
        self.current_frame = 0

        # Extract MOKA outputs
        self.moka_outputs = get_moka_outputs(self.data)

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure with interactive elements."""
        self.fig = plt.figure(figsize=(16, 10))

        # Main image display
        self.ax_image = self.fig.add_axes([0.05, 0.25, 0.55, 0.7])
        self.ax_image.set_title(f"Episode: {self.episode_path.name}")

        # Depth display (if available)
        if 'depth' in self.data:
            self.ax_depth = self.fig.add_axes([0.65, 0.55, 0.3, 0.4])
            self.ax_depth.set_title("Depth")
        else:
            self.ax_depth = None

        # Info panel
        self.ax_info = self.fig.add_axes([0.65, 0.25, 0.3, 0.25])
        self.ax_info.axis('off')

        # Timeline (skill/phase bar)
        self.ax_timeline = self.fig.add_axes([0.05, 0.15, 0.9, 0.05])
        self.ax_timeline.set_title("Timeline")

        # Slider
        self.ax_slider = self.fig.add_axes([0.1, 0.05, 0.65, 0.03])
        self.slider = Slider(
            self.ax_slider, 'Frame',
            0, self.num_frames - 1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self._on_slider_change)

        # Buttons
        self.ax_prev = self.fig.add_axes([0.8, 0.05, 0.05, 0.03])
        self.btn_prev = Button(self.ax_prev, '< Prev')
        self.btn_prev.on_clicked(self._on_prev)

        self.ax_next = self.fig.add_axes([0.86, 0.05, 0.05, 0.03])
        self.btn_next = Button(self.ax_next, 'Next >')
        self.btn_next.on_clicked(self._on_next)

        # Draw timeline
        self._draw_timeline()

        # Initial display
        self._update_display()

    def _draw_timeline(self):
        """Draw skill/phase timeline bar."""
        self.ax_timeline.clear()
        self.ax_timeline.set_xlim(0, self.num_frames)
        self.ax_timeline.set_ylim(0, 2)

        skill_ids = self.data['skill_id_per_step']
        phase_ids = self.data['phase_id']

        # Color map for skills
        cmap = plt.cm.get_cmap('tab10')

        # Draw skill bars
        prev_skill = None
        start_idx = 0
        for i, skill_id in enumerate(skill_ids):
            if skill_id != prev_skill:
                if prev_skill is not None:
                    color = cmap(prev_skill % 10)
                    self.ax_timeline.barh(1.5, i - start_idx, left=start_idx, height=0.8, color=color, alpha=0.7)
                    # Add label if segment is wide enough
                    if i - start_idx > self.num_frames * 0.05:
                        skill_name = SKILL_NAMES.get(prev_skill, str(prev_skill))
                        self.ax_timeline.text(
                            (start_idx + i) / 2, 1.5, skill_name[:8],
                            ha='center', va='center', fontsize=8
                        )
                start_idx = i
                prev_skill = skill_id

        # Last segment
        if prev_skill is not None:
            color = cmap(prev_skill % 10)
            self.ax_timeline.barh(1.5, len(skill_ids) - start_idx, left=start_idx, height=0.8, color=color, alpha=0.7)

        # Draw phase bars
        prev_phase = None
        start_idx = 0
        phase_cmap = plt.cm.get_cmap('Pastel1')
        for i, phase_id in enumerate(phase_ids):
            if phase_id != prev_phase:
                if prev_phase is not None:
                    color = phase_cmap(prev_phase % 9)
                    self.ax_timeline.barh(0.5, i - start_idx, left=start_idx, height=0.8, color=color, alpha=0.7)
                start_idx = i
                prev_phase = phase_id

        # Last segment
        if prev_phase is not None:
            color = phase_cmap(prev_phase % 9)
            self.ax_timeline.barh(0.5, len(phase_ids) - start_idx, left=start_idx, height=0.8, color=color, alpha=0.7)

        # Current position indicator
        self.timeline_marker, = self.ax_timeline.plot(
            [self.current_frame, self.current_frame], [0, 2],
            'r-', linewidth=2
        )

        self.ax_timeline.set_yticks([0.5, 1.5])
        self.ax_timeline.set_yticklabels(['Phase', 'Skill'])
        self.ax_timeline.set_xlabel('Frame')

    def _update_display(self):
        """Update all displays for current frame."""
        # Clear axes
        self.ax_image.clear()
        if self.ax_depth:
            self.ax_depth.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Get current frame data
        rgb = self.data['rgb'][self.current_frame]
        moka = self.moka_outputs[self.current_frame]
        skill_id = self.data['skill_id_per_step'][self.current_frame]
        phase_id = self.data['phase_id'][self.current_frame]
        ee_pos = self.data['ee_pos'][self.current_frame]
        ee_quat = self.data['ee_quat'][self.current_frame]
        gripper = self.data['gripper_state'][self.current_frame]

        # Display RGB with keypoint overlay
        self.ax_image.imshow(rgb)
        self._draw_keypoints(self.ax_image, moka, rgb.shape[:2])
        self.ax_image.set_title(
            f"Frame {self.current_frame}/{self.num_frames-1} | "
            f"Skill: {SKILL_NAMES.get(skill_id, str(skill_id))} | "
            f"Phase: {PHASE_NAMES.get(phase_id, str(phase_id))}"
        )
        self.ax_image.axis('off')

        # Display depth
        if self.ax_depth and 'depth' in self.data:
            depth = self.data['depth'][self.current_frame]
            im = self.ax_depth.imshow(depth, cmap='viridis')
            self.ax_depth.set_title("Depth")
            self.ax_depth.axis('off')

        # Info panel
        info_text = self._format_info(moka, ee_pos, ee_quat, gripper, skill_id, phase_id)
        self.ax_info.text(
            0.05, 0.95, info_text,
            transform=self.ax_info.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace'
        )

        # Update timeline marker
        self.timeline_marker.set_xdata([self.current_frame, self.current_frame])

        self.fig.canvas.draw_idle()

    def _draw_keypoints(self, ax, moka: MOKAOutput, img_shape: Tuple[int, int]):
        """Draw MOKA keypoints on image."""
        h, w = img_shape[:2]

        # Draw grasp keypoint
        if moka.grasp_kp and moka.grasp_kp[0] >= 0:
            u, v = moka.grasp_kp
            # Scale if keypoints are normalized
            if 0 <= u <= 1 and 0 <= v <= 1:
                u, v = u * w, v * h
            ax.plot(u, v, 'o', color=COLORS['grasp'], markersize=15, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('G', (u, v), color='white', fontsize=10, ha='center', va='center', fontweight='bold')

        # Draw function keypoint
        if moka.function_kp and moka.function_kp[0] >= 0:
            u, v = moka.function_kp
            if 0 <= u <= 1 and 0 <= v <= 1:
                u, v = u * w, v * h
            ax.plot(u, v, 's', color=COLORS['function'], markersize=12, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('F', (u, v), color='black', fontsize=10, ha='center', va='center', fontweight='bold')

        # Draw target keypoint
        if moka.target_kp and moka.target_kp[0] >= 0:
            u, v = moka.target_kp
            if 0 <= u <= 1 and 0 <= v <= 1:
                u, v = u * w, v * h
            ax.plot(u, v, '^', color=COLORS['target'], markersize=15, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('T', (u, v), color='white', fontsize=10, ha='center', va='center', fontweight='bold')

        # Draw motion arrow from grasp to target
        if moka.grasp_kp and moka.target_kp:
            if moka.grasp_kp[0] >= 0 and moka.target_kp[0] >= 0:
                g_u, g_v = moka.grasp_kp
                t_u, t_v = moka.target_kp
                if 0 <= g_u <= 1:
                    g_u, g_v = g_u * w, g_v * h
                if 0 <= t_u <= 1:
                    t_u, t_v = t_u * w, t_v * h

                ax.annotate(
                    '', xy=(t_u, t_v), xytext=(g_u, g_v),
                    arrowprops=dict(arrowstyle='->', color=COLORS['trajectory'], lw=2, alpha=0.6)
                )

    def _format_info(self, moka: MOKAOutput, ee_pos, ee_quat, gripper, skill_id, phase_id) -> str:
        """Format info text for display."""
        lines = [
            f"=== Episode Info ===",
            f"Task: {self.data['task']}",
            f"Success: {self.data['success']}",
            f"",
            f"=== MOKA Output ===",
            f"grasp_kp:    {moka.grasp_kp}",
            f"function_kp: {moka.function_kp}",
            f"target_kp:   {moka.target_kp}",
            f"tiles: {moka.pre_tile} -> {moka.target_tile} -> {moka.post_tile}",
            f"heights: {moka.pre_height} / {moka.post_height}",
            f"angle: {moka.target_angle}",
            f"",
            f"=== Proprioception ===",
            f"ee_pos:   [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
            f"ee_quat:  [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]",
            f"gripper:  {gripper:.2f}",
        ]

        # Add action if available
        if 'actions' in self.data:
            action = self.data['actions'][self.current_frame]
            lines.extend([
                f"",
                f"=== Action ===",
                f"pos: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]",
                f"rot: [{action[3]:.3f}, {action[4]:.3f}, ...]",
                f"grip: {action[9]:.2f}",
            ])

        return '\n'.join(lines)

    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_frame = int(val)
        self._update_display()

    def _on_prev(self, event):
        """Handle prev button."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.set_val(self.current_frame)

    def _on_next(self, event):
        """Handle next button."""
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.slider.set_val(self.current_frame)

    def show(self):
        """Show the viewer."""
        plt.show()

    def save_frame(self, output_path: str, frame: Optional[int] = None):
        """Save current or specified frame to image.

        Args:
            output_path: Path to save image.
            frame: Frame number (uses current if None).
        """
        if frame is not None:
            self.current_frame = frame
            self.slider.set_val(frame)
            self._update_display()

        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved frame {self.current_frame} to {output_path}")

    def save_video(self, output_path: str, fps: int = 10):
        """Save episode as video.

        Args:
            output_path: Path to save video (e.g., episode.mp4).
            fps: Frames per second.
        """
        import matplotlib.animation as animation

        def update(frame):
            self.current_frame = frame
            self._update_display()
            return []

        anim = animation.FuncAnimation(
            self.fig, update,
            frames=self.num_frames,
            interval=1000 // fps,
            blit=False
        )

        anim.save(output_path, fps=fps, dpi=100)
        print(f"Saved video to {output_path}")


def quick_view(episode_path: str, frame: Optional[int] = None):
    """Quick visualization of a single frame.

    Args:
        episode_path: Path to episode file.
        frame: Frame to display (middle frame if None).
    """
    data = load_hdf5_episode(episode_path)
    moka_outputs = get_moka_outputs(data)

    if frame is None:
        frame = data['num_timesteps'] // 2

    # Simple matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RGB with keypoints
    rgb = data['rgb'][frame]
    moka = moka_outputs[frame]

    axes[0].imshow(rgb)
    h, w = rgb.shape[:2]

    # Draw keypoints
    if moka.grasp_kp and moka.grasp_kp[0] >= 0:
        u, v = moka.grasp_kp
        if u <= 1:
            u, v = u * w, v * h
        axes[0].plot(u, v, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Grasp')

    if moka.target_kp and moka.target_kp[0] >= 0:
        u, v = moka.target_kp
        if u <= 1:
            u, v = u * w, v * h
        axes[0].plot(u, v, 'b^', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Target')

    axes[0].legend()
    axes[0].set_title(f"Frame {frame} | Task: {data['task']}")
    axes[0].axis('off')

    # Depth if available
    if 'depth' in data:
        depth = data['depth'][frame]
        axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title("Depth")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, "No depth data", ha='center', va='center')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Episode Replay Viewer')
    parser.add_argument('episode', type=str, help='Path to episode HDF5 file')
    parser.add_argument('--frame', type=int, default=None, help='Start at specific frame')
    parser.add_argument('--quick', action='store_true', help='Quick view (single frame)')
    parser.add_argument('--save-frame', type=str, default=None, help='Save frame to image')
    parser.add_argument('--save-video', type=str, default=None, help='Save as video')

    args = parser.parse_args()

    if args.quick:
        quick_view(args.episode, args.frame)
    else:
        viewer = ReplayViewer(args.episode)

        if args.frame:
            viewer.current_frame = args.frame
            viewer.slider.set_val(args.frame)

        if args.save_frame:
            viewer.save_frame(args.save_frame, args.frame)
        elif args.save_video:
            viewer.save_video(args.save_video)
        else:
            viewer.show()


if __name__ == '__main__':
    main()
