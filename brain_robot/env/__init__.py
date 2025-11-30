"""
Environment module for LIBERO simulation and mock testing.
"""

from .mock_env import MockRobotEnv, make_mock_env

# Try to import LIBERO wrapper, fallback to mock if not available
try:
    from .libero_wrapper import LIBEROEnvWrapper, make_libero_env
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    LIBEROEnvWrapper = MockRobotEnv
    make_libero_env = make_mock_env

__all__ = [
    "LIBEROEnvWrapper",
    "make_libero_env",
    "MockRobotEnv",
    "make_mock_env",
    "LIBERO_AVAILABLE",
]
