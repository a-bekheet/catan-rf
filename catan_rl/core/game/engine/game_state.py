from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .board import Board
from .types import Action, ActionType, BuildingType, DevCardType, ResourceType


class TurnPhase(str, Enum):
    SETUP = "setup"
    ROLL = "roll"
    MAIN = "main"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    END = "end"


ResourceBank = Dict[ResourceType, int]


@dataclass
class PlayerState:
    player_id: int
    resources: ResourceBank
    roads: Set[int] = field(default_factory=set)
    settlements: Set[int] = field(default_factory=set)
    cities: Set[int] = field(default_factory=set)
    victory_points: int = 0
    dev_cards: List[DevCardType] = field(default_factory=list)
    knights_played: int = 0


@dataclass
class GameState:
    board: Board
    players: Dict[int, PlayerState]
    bank: ResourceBank
    current_player: int
    phase: TurnPhase
    turn_index: int
    robber_tile: int
    vertex_occupancy: Dict[int, Tuple[int, BuildingType]] = field(default_factory=dict)
    edge_occupancy: Dict[int, int] = field(default_factory=dict)
    setup_round: int = 0
    setup_direction: int = 1
    pending_setup_vertex: int | None = None
    last_roll: int | None = None
    pending_discards: Dict[int, int] = field(default_factory=dict)
    robber_player: int | None = None
    winner: int | None = None
    dev_deck: List[DevCardType] = field(default_factory=list)
    dev_discard: List[DevCardType] = field(default_factory=list)
    played_dev_card_this_turn: bool = False
    new_dev_cards: Dict[int, List[DevCardType]] = field(default_factory=dict)
    last_bought_dev_card: DevCardType | None = None

    def legal_actions(self) -> List[Action]:
        from .rules import legal_actions

        return legal_actions(self)

    def apply(self, action: Action) -> "GameState":
        from .rules import apply_action

        return apply_action(self, action)


DEFAULT_BANK_COUNT = 19


def _empty_bank() -> ResourceBank:
    return {
        ResourceType.BRICK: DEFAULT_BANK_COUNT,
        ResourceType.LUMBER: DEFAULT_BANK_COUNT,
        ResourceType.ORE: DEFAULT_BANK_COUNT,
        ResourceType.GRAIN: DEFAULT_BANK_COUNT,
        ResourceType.WOOL: DEFAULT_BANK_COUNT,
    }


def _empty_resources() -> ResourceBank:
    return {
        ResourceType.BRICK: 0,
        ResourceType.LUMBER: 0,
        ResourceType.ORE: 0,
        ResourceType.GRAIN: 0,
        ResourceType.WOOL: 0,
    }


def initial_game_state(board: Board, num_players: int = 4) -> GameState:
    players: Dict[int, PlayerState] = {}
    for pid in range(num_players):
        players[pid] = PlayerState(player_id=pid, resources=_empty_resources())

    robber_tile = next(
        tile.tile_id for tile in board.tiles.values() if tile.resource == ResourceType.DESERT
    )

    dev_deck = (
        [DevCardType.KNIGHT] * 14
        + [DevCardType.VICTORY_POINT] * 5
        + [DevCardType.MONOPOLY] * 2
        + [DevCardType.YEAR_OF_PLENTY] * 2
        + [DevCardType.ROAD_BUILDING] * 2
    )
    import random

    rng = random.Random(42)
    rng.shuffle(dev_deck)
    new_dev_cards = {pid: [] for pid in range(num_players)}

    return GameState(
        board=board,
        players=players,
        bank=_empty_bank(),
        current_player=0,
        phase=TurnPhase.SETUP,
        turn_index=0,
        robber_tile=robber_tile,
        setup_round=0,
        setup_direction=1,
        winner=None,
        dev_deck=dev_deck,
        dev_discard=[],
        played_dev_card_this_turn=False,
        new_dev_cards=new_dev_cards,
        last_bought_dev_card=None,
    )
