"""Base agent interface for all RL approaches."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import time

from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action


@dataclass
class AgentMetrics:
    """Metrics collected for each agent decision."""
    turn_id: int
    player_id: int
    agent_type: str
    thinking_time: float
    action_confidence: float
    legal_action_count: int
    state_encoding: Optional[str] = None
    action_encoding: Optional[str] = None
    internal_metrics: Optional[Dict[str, float]] = None


@dataclass
class GameMetrics:
    """Complete metrics for a finished game."""
    game_id: str
    timestamp: str
    agent_types: List[str]
    game_length: int
    winner: Optional[int]
    final_scores: List[int]
    setup_duration: int
    main_game_duration: int
    turn_metrics: List[AgentMetrics]
    strategic_summary: Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for all Catan agents."""

    def __init__(self, player_id: int, agent_type: str = "BaseAgent", version: str = "1.0"):
        self.player_id = player_id
        self.agent_type = agent_type
        self.version = version
        self.agent_id = f"{agent_type}_p{player_id}_{version}"

        # Metrics collection
        self.decision_times: List[float] = []
        self.confidence_scores: List[float] = []
        self.internal_state: Dict[str, Any] = {}

        # Training state
        self.is_training = True
        self.episode_count = 0

    @abstractmethod
    def select_action(self, state: GameState, legal_actions: List[Action]) -> Tuple[Action, AgentMetrics]:
        """
        Select an action given the current state and legal actions.

        Args:
            state: Current game state
            legal_actions: List of valid actions

        Returns:
            Tuple of (chosen_action, decision_metrics)
        """
        pass

    @abstractmethod
    def update(self, state: GameState, action: Action, reward: float, next_state: GameState) -> Dict[str, float]:
        """
        Update the agent based on experience.

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent for a new episode."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save agent model to file."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load agent model from file."""
        pass

    @abstractmethod
    def get_state_encoding(self, state: GameState) -> str:
        """Get agent-specific state representation."""
        pass

    @abstractmethod
    def get_action_encoding(self, action: Action) -> str:
        """Get agent-specific action representation."""
        pass

    def set_training_mode(self, training: bool) -> None:
        """Set training vs evaluation mode."""
        self.is_training = training

    def get_agent_info(self) -> Dict[str, Any]:
        """Get complete agent information for logging."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "player_id": self.player_id,
            "episode_count": self.episode_count,
            "internal_state": self.internal_state.copy()
        }

    def record_decision_metrics(
        self,
        turn_id: int,
        thinking_time: float,
        confidence: float,
        legal_action_count: int,
        state: GameState,
        action: Action
    ) -> AgentMetrics:
        """Create standardized metrics for this decision."""
        self.decision_times.append(thinking_time)
        self.confidence_scores.append(confidence)

        return AgentMetrics(
            turn_id=turn_id,
            player_id=self.player_id,
            agent_type=self.agent_type,
            thinking_time=thinking_time,
            action_confidence=confidence,
            legal_action_count=legal_action_count,
            state_encoding=self.get_state_encoding(state),
            action_encoding=self.get_action_encoding(action),
            internal_metrics=self._get_internal_metrics()
        )

    def _get_internal_metrics(self) -> Dict[str, float]:
        """Get agent-specific internal metrics for analysis."""
        return {
            "avg_thinking_time": sum(self.decision_times[-100:]) / max(1, len(self.decision_times[-100:])),
            "avg_confidence": sum(self.confidence_scores[-100:]) / max(1, len(self.confidence_scores[-100:])),
            "total_decisions": len(self.decision_times)
        }

    def compute_strategic_metrics(self, state: GameState) -> Dict[str, Any]:
        """Compute strategic analysis metrics for the current state."""
        player = state.players[self.player_id]

        return {
            "victory_points": player.victory_points,
            "settlements": len(player.settlements),
            "cities": len(player.cities),
            "roads": len(player.roads),
            "development_cards": len(player.dev_cards),
            "total_resources": sum(player.resources.values()),
            "resource_diversity": sum(1 for count in player.resources.values() if count > 0),
            "knights_played": player.knights_played
        }


class AgentFactory:
    """Factory for creating different types of agents."""

    _registered_agents: Dict[str, type] = {}

    @classmethod
    def register_agent(cls, agent_name: str, agent_class: type) -> None:
        """Register a new agent type."""
        cls._registered_agents[agent_name] = agent_class

    @classmethod
    def create_agent(cls, agent_name: str, player_id: int, **kwargs) -> BaseAgent:
        """Create an agent of the specified type."""
        if agent_name not in cls._registered_agents:
            raise ValueError(f"Unknown agent type: {agent_name}")

        agent_class = cls._registered_agents[agent_name]
        return agent_class(player_id=player_id, **kwargs)

    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._registered_agents.keys())