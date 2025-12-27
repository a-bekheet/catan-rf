"""Wrap the existing Q-table agent to work with new framework."""

import time
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

from rl_framework.base.agent import BaseAgent, AgentMetrics
from catan.agents.rl_agent import RLAgent as LegacyRLAgent
from catan.engine.game_state import GameState
from catan.engine.types import Action


class LegacyQTableAgent(BaseAgent):
    """Wrapper for the existing Q-learning agent."""

    def __init__(self, player_id: int, version: str = "enhanced", **kwargs):
        super().__init__(player_id, "QTableAgent", version)

        # Create the legacy agent
        self.legacy_agent = LegacyRLAgent(
            player_id=player_id,
            **kwargs
        )

        self.total_actions = 0

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Tuple[Action, AgentMetrics]:
        """Select action using Q-table."""
        start_time = time.time()

        action = self.legacy_agent.select_action(state, legal_actions)

        thinking_time = time.time() - start_time

        # Estimate confidence from Q-values if available
        state_key = self.legacy_agent._state_key(state)
        action_key = self.legacy_agent._action_key(action)

        confidence = 0.5  # Default confidence
        if state_key in self.legacy_agent.q_table:
            state_q_values = self.legacy_agent.q_table[state_key]
            if action_key in state_q_values and len(state_q_values) > 1:
                q_values = list(state_q_values.values())
                max_q = max(q_values)
                min_q = min(q_values)
                current_q = state_q_values[action_key]

                # Confidence based on how much better this action is than alternatives
                if max_q > min_q:
                    confidence = (current_q - min_q) / (max_q - min_q)
                else:
                    confidence = 0.5

        self.total_actions += 1

        metrics = self.record_decision_metrics(
            turn_id=self.total_actions,
            thinking_time=thinking_time,
            confidence=confidence,
            legal_action_count=len(legal_actions),
            state=state,
            action=action
        )

        return action, metrics

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState) -> Dict[str, float]:
        """Update Q-table using legacy agent."""
        # Compute reward using legacy agent's method
        legacy_reward = self.legacy_agent.compute_reward(state, next_state)

        # Update using legacy method
        self.legacy_agent.update(state, action, legacy_reward, next_state)

        return {
            "q_table_size": len(self.legacy_agent.q_table),
            "epsilon": self.legacy_agent.epsilon,
            "learning_rate": self.legacy_agent.learning_rate,
            "reward": legacy_reward,
            "total_actions": self.total_actions
        }

    def reset(self) -> None:
        """Reset for new episode."""
        self.legacy_agent.reset()
        self.episode_count += 1

    def save_model(self, path: str) -> None:
        """Save Q-table model."""
        self.legacy_agent.save_model()  # Uses its own path

        # Also save our metadata
        metadata = {
            "agent_type": self.agent_type,
            "version": self.version,
            "total_actions": self.total_actions,
            "episode_count": self.episode_count,
            "legacy_model_path": self.legacy_agent.model_path
        }

        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

    def load_model(self, path: str) -> None:
        """Load Q-table model."""
        try:
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
                self.total_actions = metadata.get("total_actions", 0)
                self.episode_count = metadata.get("episode_count", 0)
        except FileNotFoundError:
            pass

        # Load legacy model if it exists
        self.legacy_agent.load_model()

    def get_state_encoding(self, state: GameState) -> str:
        """Get Q-table state representation."""
        return self.legacy_agent._state_key(state)

    def get_action_encoding(self, action: Action) -> str:
        """Get Q-table action representation."""
        return self.legacy_agent._action_key(action)

    def compute_reward(self, state: GameState, next_state: GameState) -> float:
        """Use legacy reward computation."""
        return self.legacy_agent.compute_reward(state, next_state)

    def set_training_mode(self, training: bool) -> None:
        """Set training vs evaluation mode."""
        super().set_training_mode(training)
        if training:
            self.legacy_agent.epsilon = max(self.legacy_agent.epsilon, 0.1)
        else:
            self.legacy_agent.epsilon = 0.0  # Greedy during evaluation

    def _get_internal_metrics(self) -> Dict[str, float]:
        """Get Q-table specific metrics."""
        base_metrics = super()._get_internal_metrics()

        q_metrics = {
            "q_table_size": len(self.legacy_agent.q_table),
            "epsilon": self.legacy_agent.epsilon,
            "learning_rate": self.legacy_agent.learning_rate,
            "discount": self.legacy_agent.discount
        }

        base_metrics.update(q_metrics)
        return base_metrics


# Register with factory
from rl_framework.base.agent import AgentFactory
AgentFactory.register_agent("QTableAgent", LegacyQTableAgent)