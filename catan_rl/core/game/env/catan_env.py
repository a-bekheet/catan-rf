from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import GameState, initial_game_state
from catan_rl.core.game.engine.types import Action

from .specs import EnvSpec, StepOutput


class CatanEnv:
    """Gym-style environment wrapper around the Catan engine."""

    def __init__(self, num_players: int = 4, max_turns: int = 500, seed: int | None = None):
        self.spec = EnvSpec(num_players=num_players, max_turns=max_turns, seed=seed)
        self._state: GameState | None = None
        self._turns = 0

    def reset(self, seed: int | None = None) -> Dict[str, object]:
        if seed is not None:
            self.spec = EnvSpec(
                num_players=self.spec.num_players,
                max_turns=self.spec.max_turns,
                seed=seed,
            )

        board = standard_board(seed=self.spec.seed)
        self._state = initial_game_state(board, num_players=self.spec.num_players)
        self._turns = 0
        return self._build_observation(self._state)

    def step(self, action: Action) -> StepOutput:
        if self._state is None:
            raise RuntimeError("Environment not reset")

        next_state = self._state.apply(action)
        self._turns += 1
        self._state = next_state

        terminated = next_state.winner is not None
        truncated = self._turns >= self.spec.max_turns

        # Calculate reward for the current player
        current_player_id = self._state.current_player_id
        reward = self._calculate_reward(self._state, next_state, current_player_id, terminated)

        info = {
            'winner': next_state.winner,
            'victory_points': {i: player.victory_points for i, player in next_state.players.items()}
        }

        return StepOutput(
            observation=self._build_observation(next_state),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _calculate_reward(self, prev_state: GameState, new_state: GameState, player_id: int, terminated: bool) -> float:
        """Calculate reward for the current player's action."""
        reward = 0.0

        # Get player data
        prev_player = prev_state.players[player_id]
        new_player = new_state.players[player_id]

        # Victory bonus - big reward for winning
        if terminated and new_state.winner == player_id:
            reward += 100.0

        # Victory point progression - reward gaining VP
        vp_gained = new_player.victory_points - prev_player.victory_points
        reward += vp_gained * 10.0

        # Building progression - smaller rewards for building
        buildings_gained = (len(new_player.settlements) - len(prev_player.settlements) +
                           len(new_player.cities) - len(prev_player.cities))
        reward += buildings_gained * 1.0

        # Road building - small rewards for expansion
        roads_gained = len(new_player.roads) - len(prev_player.roads)
        reward += roads_gained * 0.5

        # Small time penalty to encourage faster completion
        reward -= 0.01

        # Penalty for losing
        if terminated and new_state.winner is not None and new_state.winner != player_id:
            reward -= 10.0

        return reward

    def _build_observation(self, state: GameState) -> Dict[str, object]:
        return {
            "board": {
                "tiles": [asdict(tile) for tile in state.board.tiles.values()],
                "vertices": [asdict(vertex) for vertex in state.board.graph.vertices.values()],
                "edges": [asdict(edge) for edge in state.board.graph.edges.values()],
                "hex_to_vertices": state.board.graph.hex_to_vertices,
            },
            "current_player": state.current_player,
            "phase": state.phase.value,
        }
