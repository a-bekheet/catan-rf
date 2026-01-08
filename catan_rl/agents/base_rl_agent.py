"""
Base Agent Interface for Multi-RL Framework Support
====================================================

Unified interface that all RL frameworks (RLlib, TorchRL, Agent Lightning) must implement.
This allows seamless swapping between different RL algorithms while maintaining consistent
training and evaluation interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import time

from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action


@dataclass
class AgentMetrics:
    """Metrics collected during agent decision-making."""
    decision_time: float
    confidence: float
    exploration_rate: float
    q_value: Optional[float] = None
    policy_entropy: Optional[float] = None
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    episode: int
    total_steps: int
    episode_reward: float
    episode_length: int
    avg_loss: float
    exploration_rate: float
    win_rate: float
    avg_victory_points: float
    training_time: float
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


class BaseRLAgent(ABC):
    """
    Abstract base class for all RL agents.

    All RL framework implementations (RLlib, TorchRL, Agent Lightning) must inherit
    from this class and implement its abstract methods.
    """

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        config: Dict[str, Any]
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent (0-3 for Catan)
            agent_name: Human-readable name (e.g., "PPO Agent", "SAC Agent")
            config: Configuration dictionary for the agent
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config = config

        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.is_training = True

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.win_count = 0

    @abstractmethod
    def select_action(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Tuple[Action, AgentMetrics]:
        """
        Select an action given the current game state and legal actions.

        Args:
            game_state: Current state of the Catan game
            legal_actions: List of legal actions the agent can take

        Returns:
            Tuple of (selected_action, metrics)
        """
        pass

    @abstractmethod
    def update(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """
        Update the agent's policy based on the experience.

        Args:
            state: Previous game state
            action: Action taken
            reward: Reward received
            next_state: Resulting game state
            done: Whether the episode is complete

        Returns:
            Optional dictionary of training metrics (loss, etc.)
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """
        Save the agent's model and training state.

        Args:
            path: Directory to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """
        Load the agent's model and training state.

        Args:
            path: Directory containing checkpoint
        """
        pass

    def set_training_mode(self, mode: bool) -> None:
        """Set whether agent is in training or evaluation mode."""
        self.is_training = mode

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        win_rate = self.win_count / max(1, self.total_episodes)
        avg_reward = sum(self.episode_rewards[-100:]) / max(1, len(self.episode_rewards[-100:]))
        avg_length = sum(self.episode_lengths[-100:]) / max(1, len(self.episode_lengths[-100:]))

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "win_rate": win_rate,
            "avg_reward_100": avg_reward,
            "avg_length_100": avg_length,
        }

    def reset_episode(self) -> None:
        """Reset agent state for a new episode."""
        self.total_episodes += 1

    def record_episode_end(self, total_reward: float, length: int, won: bool) -> None:
        """Record metrics at the end of an episode."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        if won:
            self.win_count += 1

    @property
    def framework_name(self) -> str:
        """Return the name of the RL framework used."""
        return self.__class__.__name__.replace("Agent", "")


class RandomAgent(BaseRLAgent):
    """Simple random agent for baseline comparison."""

    def __init__(self, agent_id: int, config: Dict[str, Any] = None):
        super().__init__(agent_id, "Random Agent", config or {})
        import random
        self.rng = random.Random(config.get('seed') if config else None)

    def select_action(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Tuple[Action, AgentMetrics]:
        """Select a random legal action."""
        start_time = time.time()
        action = self.rng.choice(legal_actions)

        metrics = AgentMetrics(
            decision_time=time.time() - start_time,
            confidence=0.0,
            exploration_rate=1.0
        )

        return action, metrics

    def update(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """Random agent doesn't learn."""
        return None

    def save_checkpoint(self, path: Path) -> None:
        """Random agent has no state to save."""
        pass

    def load_checkpoint(self, path: Path) -> None:
        """Random agent has no state to load."""
        pass
