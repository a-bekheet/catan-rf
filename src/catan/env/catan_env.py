from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from catan.engine.board import standard_board
from catan.engine.game_state import GameState, initial_game_state
from catan.engine.types import Action

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

        terminated = False
        truncated = self._turns >= self.spec.max_turns
        reward = 0.0
        info = {}

        return StepOutput(
            observation=self._build_observation(next_state),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

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
