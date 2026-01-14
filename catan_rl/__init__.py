"""
Catan Reinforcement Learning Research Package
==============================================

A comprehensive RL research platform for Settlers of Catan with support for
multiple RL frameworks (DQN, PPO, SAC, LLM-based agents).
"""

__version__ = "0.1.0"

# Core game engine (no heavy dependencies)
from . import core

# Optional imports with lazy loading to avoid dependency issues
__all__ = ['core', '__version__']

# Try importing agents if dependencies available
try:
    from . import agents
    __all__.append('agents')
except ImportError:
    pass

# Try importing environments if dependencies available
try:
    from . import environments
    __all__.append('environments')
except ImportError:
    pass

# Try importing experiments if dependencies available
try:
    from . import experiments
    __all__.append('experiments')
except ImportError:
    pass
