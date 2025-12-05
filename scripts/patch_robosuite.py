"""Patch robosuite macros to use EGL rendering with NVIDIA.

This module must be imported BEFORE any robosuite imports.
It patches the macros module and sets up library paths for EGL rendering.

Usage:
    import patch_robosuite  # Must be first import
    import robosuite
    from libero.libero import benchmark
    ...

NOTE: For best results, set these environment variables BEFORE running Python:
    export LD_LIBRARY_PATH="/workspace/brain_robot/lib:$LD_LIBRARY_PATH"
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_GL=egl

Or use the wrapper script: ./scripts/run_with_gl.sh your_script.py
"""

import os
import sys
import types
from pathlib import Path

# Get the lib directory with our GL libraries
_script_dir = Path(__file__).parent.absolute()
_lib_dir = str(_script_dir.parent / "lib")

# IMPORTANT: Tell PyOpenGL to use EGL platform (not GLX which requires X11)
# This MUST be set before any OpenGL imports
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Set MUJOCO_GL to use EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

# Add lib dir to LD_LIBRARY_PATH
# Note: This won't help for libraries already loaded, but is needed for MuJoCo's renderer
_existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if _lib_dir not in _existing_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{_lib_dir}:{_existing_ld_path}" if _existing_ld_path else _lib_dir

# Create fake macros module
fake_macros = types.ModuleType('robosuite.macros')
fake_macros.MUJOCO_GPU_RENDERING = False  # Key change - disable EGL
fake_macros.SIMULATION_TIMESTEP = 0.002
fake_macros.USING_INSTANCE_RANDOMIZATION = False
fake_macros.ENABLE_NUMBA = True
fake_macros.CACHE_NUMBA = True
fake_macros.IMAGE_CONVENTION = 'opengl'
fake_macros.CONCATENATE_IMAGES = False
fake_macros.SPACEMOUSE_VENDOR_ID = 9583
fake_macros.SPACEMOUSE_PRODUCT_ID = 50735
fake_macros.CONSOLE_LOGGING_LEVEL = 'WARN'
fake_macros.FILE_LOGGING_LEVEL = 'DEBUG'

# Inject into sys.modules before robosuite imports it
sys.modules['robosuite.macros'] = fake_macros

print("[patch_robosuite] Patched robosuite macros for GLFW rendering")
