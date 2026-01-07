from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class ObservationKey(str, Enum):
    BOARD_TILES = "board_tiles"
    BOARD_GRAPH = "board_graph"
    PLAYER_STATE = "player_state"
    TURN_STATE = "turn_state"


@dataclass(frozen=True)
class EnvSpec:
    num_players: int
    max_turns: int
    seed: int | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_players": self.num_players,
            "max_turns": self.max_turns,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class StepOutput:
    observation: Dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]
