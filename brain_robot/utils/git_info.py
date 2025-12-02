"""Git version tracking for reproducibility.

Captures git commit and dirty state for logging.
"""

import subprocess
from typing import Dict, Any, Optional


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get current git state for reproducibility.
    
    Args:
        repo_path: Optional path to git repository. If None, uses current directory.
        
    Returns:
        Dictionary with git_commit (8-char hash) and git_dirty (bool).
    """
    try:
        # Get current commit hash
        cmd = ['git', 'rev-parse', 'HEAD']
        if repo_path:
            cmd = ['git', '-C', repo_path, 'rev-parse', 'HEAD']
        
        commit = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        
        # Check if working directory is dirty
        cmd = ['git', 'diff', '--quiet']
        if repo_path:
            cmd = ['git', '-C', repo_path, 'diff', '--quiet']
        
        dirty = subprocess.call(
            cmd,
            stderr=subprocess.DEVNULL
        ) != 0
        
        # Also check for staged changes
        cmd = ['git', 'diff', '--cached', '--quiet']
        if repo_path:
            cmd = ['git', '-C', repo_path, 'diff', '--cached', '--quiet']
        
        staged_dirty = subprocess.call(
            cmd,
            stderr=subprocess.DEVNULL
        ) != 0
        
        # Get branch name
        cmd = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        if repo_path:
            cmd = ['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD']
        
        try:
            branch = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            branch = "unknown"
        
        return {
            "git_commit": commit,
            "git_dirty": dirty or staged_dirty,
            "git_branch": branch,
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {
            "git_commit": "unknown",
            "git_dirty": None,
            "git_branch": "unknown",
        }


def get_code_version() -> Dict[str, Any]:
    """Get comprehensive code version info for logging.
    
    Returns:
        Dictionary with git info and package versions.
    """
    version_info = get_git_info()
    
    # Add package versions
    try:
        import numpy
        version_info["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import torch
        version_info["torch_version"] = torch.__version__
    except ImportError:
        pass
    
    return version_info
