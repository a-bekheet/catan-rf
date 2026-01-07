"""
Catan Reinforcement Learning Research Framework

A comprehensive framework for researching and comparing different RL approaches
for playing Settlers of Catan.

Author: AI Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from . import core
from . import agents
from . import experiments
from . import environments
from . import analysis
from . import utils

__all__ = [
    "core",
    "agents",
    "experiments",
    "environments",
    "analysis",
    "utils"
]