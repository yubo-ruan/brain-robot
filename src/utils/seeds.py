"""Seed management for reproducibility.

CRITICAL: All RNG seeds including simulator must be set for true reproducibility.
"""

import random
import numpy as np

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_global_seed(seed: int, env=None) -> int:
    """Set ALL RNG seeds including simulator for true reproducibility.
    
    Args:
        seed: The seed value to use.
        env: Optional environment to seed (robosuite/LIBERO).
        
    Returns:
        The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # CRITICAL: Also seed the simulator
    if env is not None:
        if hasattr(env, 'seed'):
            env.seed(seed)
        # robosuite/LIBERO wrapped env
        if hasattr(env, '_env') and hasattr(env._env, 'seed'):
            env._env.seed(seed)
        # Handle OffScreenRenderEnv wrapping
        if hasattr(env, 'env') and hasattr(env.env, 'seed'):
            env.env.seed(seed)
    
    return seed


def get_episode_seed(run_seed: int, episode_idx: int) -> int:
    """Generate deterministic per-episode seed.
    
    This ensures each episode gets a unique but reproducible seed
    based on the run seed and episode index.
    
    Args:
        run_seed: The base seed for the entire run.
        episode_idx: The index of the episode (0-based).
        
    Returns:
        Deterministic seed for this episode.
    """
    return run_seed + episode_idx
