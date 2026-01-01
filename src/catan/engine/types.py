from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple


class ResourceType(str, Enum):
    BRICK = "brick"
    LUMBER = "lumber"
    ORE = "ore"
    GRAIN = "grain"
    WOOL = "wool"
    DESERT = "desert"


class DevCardType(str, Enum):
    KNIGHT = "knight"
    MONOPOLY = "monopoly"
    YEAR_OF_PLENTY = "year_of_plenty"
    ROAD_BUILDING = "road_building"
    VICTORY_POINT = "victory_point"


class ActionType(str, Enum):
    PASS_TURN = "pass_turn"
    ROLL_DICE = "roll_dice"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_DEV_CARD = "play_dev_card"
    TRADE_BANK = "trade_bank"
    TRADE_PLAYER = "trade_player"
    MOVE_ROBBER = "move_robber"
    DISCARD = "discard"


class BuildingType(str, Enum):
    SETTLEMENT = "settlement"
    CITY = "city"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    payload: Dict[str, object]


@dataclass(frozen=True)
class Vertex:
    vertex_id: int
    coord: Tuple[int, int]


@dataclass(frozen=True)
class Edge:
    edge_id: int
    vertex_a: int
    vertex_b: int


@dataclass(frozen=True)
class HexTile:
    tile_id: int
    axial: Tuple[int, int]
    resource: ResourceType
    number_token: int | None
    has_robber: bool = False


@dataclass(frozen=True)
class BoardGraph:
    vertices: Dict[int, Vertex]
    edges: Dict[int, Edge]
    hex_to_vertices: Dict[int, List[int]]

    def vertex_ids(self) -> Iterable[int]:
        return self.vertices.keys()

    def edge_ids(self) -> Iterable[int]:
        return self.edges.keys()
