"""
Phase 1: HDF5 Episode Logger for MOKA-Guided Skill Policies

Extends the existing episode logger with HDF5 support for:
- Efficient storage of RGB/depth images
- MOKA keypoint outputs
- BC/RL training data format

This is complementary to the JSON-based episode_logger.py which is better for debugging.
"""

import os
import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
from PIL import Image


@dataclass
class MOKAOutput:
    """MOKA output schema (from Phase 0 validation)."""
    grasp_kp: Optional[Tuple[float, float]] = None
    function_kp: Optional[Tuple[float, float]] = None
    target_kp: Optional[Tuple[float, float]] = None
    pre_tile: Optional[str] = None
    target_tile: Optional[str] = None
    post_tile: Optional[str] = None
    pre_height: Optional[str] = None
    post_height: Optional[str] = None
    target_angle: Optional[str] = None

    # Confidence scores
    conf_grasp: float = 1.0
    conf_target: float = 1.0

    def to_array(self) -> np.ndarray:
        """Convert to flat array for storage.

        Returns:
            Array of shape (16,):
            [grasp_u, grasp_v, function_u, function_v, target_u, target_v,
             conf_grasp, conf_target, pre_height_enc, post_height_enc,
             angle_enc, tile_enc_1, tile_enc_2, tile_enc_3, tile_enc_4, tile_enc_5]
        """
        arr = np.zeros(16, dtype=np.float32)

        # Keypoints (normalized 0-1 assumed)
        if self.grasp_kp:
            arr[0:2] = self.grasp_kp
        else:
            arr[0:2] = -1

        if self.function_kp:
            arr[2:4] = self.function_kp
        else:
            arr[2:4] = -1

        if self.target_kp:
            arr[4:6] = self.target_kp
        else:
            arr[4:6] = -1

        # Confidence
        arr[6] = self.conf_grasp
        arr[7] = self.conf_target

        # Height encoding: 0=none, 1=same, 2=above
        height_map = {None: 0, '': 0, 'same': 1, 'above': 2}
        arr[8] = height_map.get(self.pre_height, 0)
        arr[9] = height_map.get(self.post_height, 0)

        # Angle encoding (simplified)
        angle_map = {None: 0, '': 0, 'forward': 1, 'downward': 2, 'left': 3, 'right': 4}
        arr[10] = angle_map.get(self.target_angle, 0)

        # Tile encoding (grid 5x5 = 25 possibilities + 0 for none)
        def tile_to_int(tile):
            if not tile:
                return 0
            try:
                col = ord(tile[0]) - ord('a')  # 0-4
                row = int(tile[1]) - 1  # 0-4
                return col * 5 + row + 1  # 1-25
            except:
                return 0

        arr[11] = tile_to_int(self.pre_tile)
        arr[12] = tile_to_int(self.target_tile)
        arr[13] = tile_to_int(self.post_tile)

        # Reserved
        arr[14:16] = 0

        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'MOKAOutput':
        """Reconstruct from array."""
        grasp_kp = tuple(arr[0:2]) if arr[0] >= 0 else None
        function_kp = tuple(arr[2:4]) if arr[2] >= 0 else None
        target_kp = tuple(arr[4:6]) if arr[4] >= 0 else None

        height_decode = {0: None, 1: 'same', 2: 'above'}
        angle_decode = {0: None, 1: 'forward', 2: 'downward', 3: 'left', 4: 'right'}

        def int_to_tile(val):
            val = int(val)
            if val <= 0:
                return None
            val -= 1
            col = val // 5
            row = val % 5
            return f"{chr(ord('a') + col)}{row + 1}"

        return cls(
            grasp_kp=grasp_kp,
            function_kp=function_kp,
            target_kp=target_kp,
            pre_tile=int_to_tile(arr[11]),
            target_tile=int_to_tile(arr[12]),
            post_tile=int_to_tile(arr[13]),
            pre_height=height_decode.get(int(arr[8])),
            post_height=height_decode.get(int(arr[9])),
            target_angle=angle_decode.get(int(arr[10])),
            conf_grasp=float(arr[6]),
            conf_target=float(arr[7]),
        )


@dataclass
class Timestep:
    """Single timestep data for HDF5 storage."""
    # Images
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32

    # Proprioception
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    ee_quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32))
    gripper_state: float = 0.0  # 0=open, 1=closed
    joint_pos: Optional[np.ndarray] = None  # (7,) for 7-DOF arm

    # Forces
    ee_force: Optional[np.ndarray] = None  # (3,)
    ee_torque: Optional[np.ndarray] = None  # (3,)

    # MOKA output (may be updated periodically)
    moka_output: Optional[MOKAOutput] = None

    # Action taken (10D: 3 pos + 6D rot + gripper)
    action: Optional[np.ndarray] = None

    # Skill and phase
    skill_id: int = -1
    phase_id: int = -1

    # Timing
    timestamp: float = 0.0

    # Termination
    is_terminal: bool = False
    success: bool = False


class HDF5EpisodeLogger:
    """
    Logger for recording robot episodes to HDF5.

    Designed for efficient storage and loading of training data.
    Uses chunked, compressed datasets for images.

    Usage:
        logger = HDF5EpisodeLogger(log_dir='/path/to/logs')

        with logger.start_episode(task='pick_and_place', skill_id=0) as ep:
            for step in range(max_steps):
                timestep = Timestep(
                    rgb=obs['rgb'],
                    depth=obs['depth'],
                    ee_pos=obs['ee_pos'],
                    ...
                )
                ep.add_timestep(timestep)

                if done:
                    ep.mark_success(True)
                    break
    """

    def __init__(
        self,
        log_dir: str = './episode_logs',
        image_size: Tuple[int, int] = (256, 256),
        compress: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.compress = compress

        self._current_episode: Optional['HDF5Episode'] = None

    def start_episode(
        self,
        task: str = '',
        skill_id: int = -1,
        metadata: Optional[Dict] = None,
    ) -> 'HDF5Episode':
        """Start a new episode."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'episode_{timestamp}.hdf5'
        filepath = self.log_dir / filename

        episode = HDF5Episode(
            filepath=filepath,
            task=task,
            skill_id=skill_id,
            metadata=metadata or {},
            image_size=self.image_size,
            compress=self.compress,
        )
        self._current_episode = episode
        return episode

    def list_episodes(self) -> List[Path]:
        """List all recorded episodes."""
        return sorted(self.log_dir.glob('episode_*.hdf5'))


class HDF5Episode:
    """Single episode recorder with HDF5 backend."""

    def __init__(
        self,
        filepath: Path,
        task: str,
        skill_id: int,
        metadata: Dict,
        image_size: Tuple[int, int],
        compress: bool,
    ):
        self.filepath = filepath
        self.task = task
        self.skill_id = skill_id
        self.metadata = metadata
        self.image_size = image_size
        self.compress = compress

        self._timesteps: List[Timestep] = []
        self._success: Optional[bool] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        return False

    def add_timestep(self, timestep: Timestep):
        """Add a timestep to the episode."""
        self._timesteps.append(timestep)

    def mark_success(self, success: bool):
        """Mark episode success/failure."""
        self._success = success

    @property
    def num_timesteps(self) -> int:
        return len(self._timesteps)

    def save(self):
        """Save episode to HDF5 file."""
        if len(self._timesteps) == 0:
            return

        compression = 'gzip' if self.compress else None
        h, w = self.image_size

        with h5py.File(self.filepath, 'w') as f:
            # Metadata
            f.attrs['task'] = self.task
            f.attrs['skill_id'] = self.skill_id
            f.attrs['num_timesteps'] = len(self._timesteps)
            f.attrs['success'] = self._success if self._success is not None else False
            f.attrs['metadata'] = json.dumps(self.metadata)
            f.attrs['image_size'] = self.image_size
            f.attrs['created_at'] = datetime.now().isoformat()

            n = len(self._timesteps)

            # Images - chunked for efficient access
            rgb_data = np.stack([self._resize_image(t.rgb) for t in self._timesteps])
            f.create_dataset(
                'rgb', data=rgb_data,
                compression=compression,
                chunks=(1, h, w, 3)  # One frame per chunk
            )

            # Depth (if available)
            has_depth = self._timesteps[0].depth is not None
            if has_depth:
                depth_data = np.stack([
                    self._resize_depth(t.depth) if t.depth is not None
                    else np.zeros((h, w), dtype=np.float32)
                    for t in self._timesteps
                ])
                f.create_dataset(
                    'depth', data=depth_data,
                    compression=compression,
                    chunks=(1, h, w)
                )

            # Proprioception
            f.create_dataset('ee_pos', data=np.stack([t.ee_pos for t in self._timesteps]))
            f.create_dataset('ee_quat', data=np.stack([t.ee_quat for t in self._timesteps]))
            f.create_dataset('gripper_state', data=np.array([t.gripper_state for t in self._timesteps], dtype=np.float32))

            has_joint = self._timesteps[0].joint_pos is not None
            if has_joint:
                f.create_dataset('joint_pos', data=np.stack([
                    t.joint_pos if t.joint_pos is not None else np.zeros(7, dtype=np.float32)
                    for t in self._timesteps
                ]))

            # Forces (if available)
            has_force = self._timesteps[0].ee_force is not None
            if has_force:
                f.create_dataset('ee_force', data=np.stack([
                    t.ee_force if t.ee_force is not None else np.zeros(3, dtype=np.float32)
                    for t in self._timesteps
                ]))

            has_torque = self._timesteps[0].ee_torque is not None
            if has_torque:
                f.create_dataset('ee_torque', data=np.stack([
                    t.ee_torque if t.ee_torque is not None else np.zeros(3, dtype=np.float32)
                    for t in self._timesteps
                ]))

            # MOKA outputs (16D per timestep)
            moka_data = np.stack([
                t.moka_output.to_array() if t.moka_output else np.zeros(16, dtype=np.float32)
                for t in self._timesteps
            ])
            f.create_dataset('moka', data=moka_data)

            # Actions (10D: 3 pos + 6D rot + gripper)
            has_action = self._timesteps[0].action is not None
            if has_action:
                actions = np.stack([
                    t.action if t.action is not None else np.zeros(10, dtype=np.float32)
                    for t in self._timesteps
                ])
                f.create_dataset('actions', data=actions)

            # Skill and phase IDs
            f.create_dataset('skill_id', data=np.array([t.skill_id for t in self._timesteps], dtype=np.int32))
            f.create_dataset('phase_id', data=np.array([t.phase_id for t in self._timesteps], dtype=np.int32))

            # Timestamps
            f.create_dataset('timestamps', data=np.array([t.timestamp for t in self._timesteps], dtype=np.float64))

            # Termination info
            f.create_dataset('is_terminal', data=np.array([t.is_terminal for t in self._timesteps], dtype=bool))
            f.create_dataset('step_success', data=np.array([t.success for t in self._timesteps], dtype=bool))

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        h, w = self.image_size
        if img.shape[:2] != (h, w):
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            return np.array(pil_img)
        return img

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Resize depth to target size."""
        h, w = self.image_size
        if depth.shape[:2] != (h, w):
            pil_img = Image.fromarray(depth)
            pil_img = pil_img.resize((w, h), Image.NEAREST)
            return np.array(pil_img)
        return depth


def load_hdf5_episode(filepath: str) -> Dict:
    """Load an episode from HDF5 file.

    Returns dictionary with all episode data ready for training.
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['task'] = f.attrs['task']
        data['skill_id'] = int(f.attrs['skill_id'])
        data['num_timesteps'] = int(f.attrs['num_timesteps'])
        data['success'] = bool(f.attrs['success'])
        data['metadata'] = json.loads(f.attrs['metadata'])
        data['image_size'] = tuple(f.attrs['image_size'])
        data['created_at'] = f.attrs['created_at']

        # Images
        data['rgb'] = f['rgb'][:]
        if 'depth' in f:
            data['depth'] = f['depth'][:]

        # Proprioception
        data['ee_pos'] = f['ee_pos'][:]
        data['ee_quat'] = f['ee_quat'][:]
        data['gripper_state'] = f['gripper_state'][:]

        if 'joint_pos' in f:
            data['joint_pos'] = f['joint_pos'][:]

        if 'ee_force' in f:
            data['ee_force'] = f['ee_force'][:]
        if 'ee_torque' in f:
            data['ee_torque'] = f['ee_torque'][:]

        # MOKA (16D array)
        data['moka'] = f['moka'][:]

        # Actions
        if 'actions' in f:
            data['actions'] = f['actions'][:]

        data['skill_id_per_step'] = f['skill_id'][:]
        data['phase_id'] = f['phase_id'][:]
        data['timestamps'] = f['timestamps'][:]
        data['is_terminal'] = f['is_terminal'][:]
        data['step_success'] = f['step_success'][:]

    return data


def get_moka_outputs(episode_data: Dict) -> List[MOKAOutput]:
    """Extract MOKA outputs from episode data."""
    moka_arr = episode_data['moka']
    return [MOKAOutput.from_array(arr) for arr in moka_arr]
