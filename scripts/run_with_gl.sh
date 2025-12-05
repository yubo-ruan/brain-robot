#!/bin/bash
# Wrapper script to run Python with GL libraries available

# Add our local GL libraries to the path
export LD_LIBRARY_PATH="/workspace/brain_robot/lib:${LD_LIBRARY_PATH:-}"

# Tell PyOpenGL to use EGL platform (not GLX which needs X11)
export PYOPENGL_PLATFORM=egl

# Set MuJoCo to use EGL
export MUJOCO_GL=egl

# Run Python with the provided arguments
exec python3 "$@"
