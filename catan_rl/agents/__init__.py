"""Reinforcement Learning agents for Catan."""

# Lazy imports to avoid dependency issues
# Import base agents that don't require heavy dependencies
from .base_rl_agent import BaseRLAgent, RandomAgent, AgentMetrics, TrainingMetrics

__all__ = [
    'BaseRLAgent',
    'RandomAgent',
    'AgentMetrics',
    'TrainingMetrics',
]

# Optional: Try importing DQN agent if torch is available
try:
    from .dqn_agent import DQNAgent, DQNAgentFactory
    __all__.extend(['DQNAgent', 'DQNAgentFactory'])
except ImportError:
    pass

# Optional: Try importing RL framework agents
try:
    from .rllib_ppo_agent import RLlibPPOAgent
    __all__.append('RLlibPPOAgent')
except ImportError:
    pass

try:
    from .torchrl_sac_agent import TorchRLSACAgent
    __all__.append('TorchRLSACAgent')
except ImportError:
    pass

try:
    from .langgraph_llm_agent import LangGraphLLMAgent
    __all__.append('LangGraphLLMAgent')
except ImportError:
    pass
