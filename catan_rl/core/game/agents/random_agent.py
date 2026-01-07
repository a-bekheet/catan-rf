"""Random agent for baseline comparison."""

from __future__ import annotations

import random
from typing import Dict, List

from catan_rl.core.game.engine.game_state import GameState, TurnPhase
from catan_rl.core.game.engine.types import Action, ActionType, ResourceType


class RandomAgent:
    """A random agent that selects random legal actions."""

    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.rng = random.Random(seed)

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """Select a random legal action."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        action = self.rng.choice(legal_actions)

        # Handle discard actions by generating valid resource combinations
        if action.action_type == ActionType.DISCARD and state.phase == TurnPhase.DISCARD:
            action = self._generate_discard_action(state, action)

        return action

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
            # Not enough resources, discard what we have
            required = len(available)

        # Randomly select resources to discard
        to_discard = self.rng.sample(available, required)

        # Count resources to discard
        discard_counts = {}
        for resource in to_discard:
            discard_counts[resource.value] = discard_counts.get(resource.value, 0) + 1

        # Create new action with proper resources
        return Action(
            action_type=ActionType.DISCARD,
            payload={
                "player_id": self.player_id,
                "resources": discard_counts
            }
        )

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState):
        """Update agent (no-op for random agent)."""
        pass

    def reset(self):
        """Reset agent state for new episode."""
        pass