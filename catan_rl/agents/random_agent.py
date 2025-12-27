"""Random baseline agent for comparison."""

import random
import time
import pickle
from typing import Dict, List, Tuple

from rl_framework.base.agent import BaseAgent, AgentMetrics
from catan.engine.game_state import GameState, TurnPhase
from catan.engine.types import Action, ActionType, ResourceType


class RandomAgent(BaseAgent):
    """Agent that selects random legal actions."""

    def __init__(self, player_id: int, seed: int = None, version: str = "1.0"):
        super().__init__(player_id, "RandomAgent", version)
        self.rng = random.Random(seed)
        self.total_actions = 0

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Tuple[Action, AgentMetrics]:
        """Select a random legal action."""
        start_time = time.time()

        if not legal_actions:
            raise ValueError("No legal actions available")

        # Simple uniform random selection
        action = self.rng.choice(legal_actions)

        # Handle special cases requiring resource specification
        if action.action_type == ActionType.DISCARD and state.phase == TurnPhase.DISCARD:
            action = self._generate_discard_action(state, action)

        thinking_time = time.time() - start_time
        confidence = 1.0 / len(legal_actions)  # Uniform confidence

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

    def _generate_discard_action(self, state: GameState, template_action: Action) -> Action:
        """Generate a valid discard action with proper resource allocation."""
        player = state.players[self.player_id]
        required = state.pending_discards.get(self.player_id, 0)

        if required <= 0:
            return template_action

        # Get available resources
        available = []
        for resource_type in ResourceType:
            count = player.resources.get(resource_type, 0)
            for _ in range(count):
                available.append(resource_type)

        if len(available) < required:
            required = len(available)

        # Randomly select resources to discard
        to_discard = self.rng.sample(available, required)

        # Count resources to discard
        discard_counts = {}
        for resource in to_discard:
            discard_counts[resource.value] = discard_counts.get(resource.value, 0) + 1

        return Action(
            action_type=ActionType.DISCARD,
            payload={
                "player_id": self.player_id,
                "resources": discard_counts
            }
        )

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState) -> Dict[str, float]:
        """Random agent doesn't learn, but we track some metrics."""
        return {
            "total_actions": self.total_actions,
            "exploration_rate": 1.0,  # Always exploring
            "learning_rate": 0.0
        }

    def reset(self) -> None:
        """Reset for new episode."""
        self.episode_count += 1

    def save_model(self, path: str) -> None:
        """Save random agent state (minimal)."""
        model_data = {
            "agent_type": self.agent_type,
            "version": self.version,
            "player_id": self.player_id,
            "total_actions": self.total_actions,
            "episode_count": self.episode_count
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str) -> None:
        """Load random agent state."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                self.total_actions = model_data.get("total_actions", 0)
                self.episode_count = model_data.get("episode_count", 0)
        except FileNotFoundError:
            pass  # Start fresh

    def get_state_encoding(self, state: GameState) -> str:
        """Get simple state representation for random agent."""
        player = state.players[self.player_id]
        return f"phase:{state.phase.value}_vp:{player.victory_points}_res:{sum(player.resources.values())}"

    def get_action_encoding(self, action: Action) -> str:
        """Get action representation."""
        return f"{action.action_type.value}:{hash(str(action.payload)) % 1000}"


# Register with factory
from rl_framework.base.agent import AgentFactory
AgentFactory.register_agent("RandomAgent", RandomAgent)