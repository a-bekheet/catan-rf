from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .types import BoardGraph, Edge, HexTile, ResourceType, Vertex

AXIAL_RADIUS = 2

STANDARD_RESOURCES = [
    ResourceType.BRICK,
    ResourceType.BRICK,
    ResourceType.BRICK,
    ResourceType.LUMBER,
    ResourceType.LUMBER,
    ResourceType.LUMBER,
    ResourceType.LUMBER,
    ResourceType.ORE,
    ResourceType.ORE,
    ResourceType.ORE,
    ResourceType.GRAIN,
    ResourceType.GRAIN,
    ResourceType.GRAIN,
    ResourceType.GRAIN,
    ResourceType.WOOL,
    ResourceType.WOOL,
    ResourceType.WOOL,
    ResourceType.WOOL,
    ResourceType.DESERT,
]

STANDARD_NUMBER_TOKENS = [
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
]

CORNER_OFFSETS = [
    (0, 4),
    (2, 2),
    (2, -2),
    (0, -4),
    (-2, -2),
    (-2, 2),
]

AXIAL_DIRECTIONS = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
]


@dataclass(frozen=True)
class NumberShuffleConstraints:
    no_adjacent_six_eight: bool = True
    no_adjacent_same_number: bool = False
    no_adjacent_two_twelve: bool = False
    max_attempts: int = 5000


@dataclass(frozen=True)
class Board:
    tiles: Dict[int, HexTile]
    graph: BoardGraph
    tile_neighbors: Dict[int, List[int]]

    def tile_ids(self) -> Iterable[int]:
        return self.tiles.keys()

    def edge_between(self, vertex_a: int, vertex_b: int) -> int | None:
        for edge in self.graph.edges.values():
            if (edge.vertex_a == vertex_a and edge.vertex_b == vertex_b) or (
                edge.vertex_a == vertex_b and edge.vertex_b == vertex_a
            ):
                return edge.edge_id
        return None

    def vertices_adjacent_to(self, vertex_id: int) -> List[int]:
        neighbors: List[int] = []
        for edge in self.graph.edges.values():
            if edge.vertex_a == vertex_id:
                neighbors.append(edge.vertex_b)
            elif edge.vertex_b == vertex_id:
                neighbors.append(edge.vertex_a)
        return neighbors

    def edges_for_vertex(self, vertex_id: int) -> List[int]:
        edges: List[int] = []
        for edge in self.graph.edges.values():
            if edge.vertex_a == vertex_id or edge.vertex_b == vertex_id:
                edges.append(edge.edge_id)
        return edges


def axial_coords(radius: int = AXIAL_RADIUS) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if -radius <= q + r <= radius:
                coords.append((q, r))
    return coords


def axial_center(q: int, r: int) -> Tuple[int, int]:
    size = 2
    return (size * (2 * q + r), size * (3 * r))


def build_tile_neighbors(coords: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    coord_to_id = {coord: tile_id for tile_id, coord in enumerate(coords)}
    neighbors: Dict[int, List[int]] = {}
    for tile_id, (q, r) in enumerate(coords):
        tile_neighbors: List[int] = []
        for dq, dr in AXIAL_DIRECTIONS:
            neighbor_coord = (q + dq, r + dr)
            if neighbor_coord in coord_to_id:
                tile_neighbors.append(coord_to_id[neighbor_coord])
        neighbors[tile_id] = tile_neighbors
    return neighbors


def build_board_graph(coords: List[Tuple[int, int]]) -> BoardGraph:
    vertex_map: Dict[Tuple[int, int], int] = {}
    vertices: Dict[int, Vertex] = {}
    edges: Dict[Tuple[int, int], int] = {}
    hex_to_vertices: Dict[int, List[int]] = {}

    def vertex_id_for(coord: Tuple[int, int]) -> int:
        if coord in vertex_map:
            return vertex_map[coord]
        vid = len(vertex_map)
        vertex_map[coord] = vid
        vertices[vid] = Vertex(vertex_id=vid, coord=coord)
        return vid

    for tile_id, (q, r) in enumerate(coords):
        cx, cy = axial_center(q, r)
        vertex_ids: List[int] = []
        for ox, oy in CORNER_OFFSETS:
            vid = vertex_id_for((cx + ox, cy + oy))
            vertex_ids.append(vid)
        hex_to_vertices[tile_id] = vertex_ids

        for i in range(6):
            a = vertex_ids[i]
            b = vertex_ids[(i + 1) % 6]
            edge_key = (min(a, b), max(a, b))
            if edge_key not in edges:
                edges[edge_key] = len(edges)

    edge_objs: Dict[int, Edge] = {
        eid: Edge(edge_id=eid, vertex_a=a, vertex_b=b)
        for (a, b), eid in edges.items()
    }

    return BoardGraph(vertices=vertices, edges=edge_objs, hex_to_vertices=hex_to_vertices)

def _numbers_valid(
    numbers_by_tile: Dict[int, int],
    neighbors: Dict[int, List[int]],
    constraints: NumberShuffleConstraints,
) -> bool:
    for tile_id, value in numbers_by_tile.items():
        for neighbor_id in neighbors[tile_id]:
            if neighbor_id not in numbers_by_tile:
                continue
            other = numbers_by_tile[neighbor_id]
            if constraints.no_adjacent_six_eight and value in (6, 8) and other in (6, 8):
                return False
            if constraints.no_adjacent_same_number and value == other:
                return False
            if constraints.no_adjacent_two_twelve and value in (2, 12) and other in (2, 12):
                return False
    return True


def _assign_numbers(
    tile_ids: List[int],
    numbers: List[int],
    neighbors: Dict[int, List[int]],
    rng,
    constraints: NumberShuffleConstraints,
) -> Dict[int, int]:
    for _ in range(constraints.max_attempts):
        rng.shuffle(numbers)
        numbers_by_tile = {tile_id: numbers[idx] for idx, tile_id in enumerate(tile_ids)}
        if _numbers_valid(numbers_by_tile, neighbors, constraints):
            return numbers_by_tile
    raise RuntimeError("Failed to assign numbers within constraints")


def standard_board(
    seed: int | None = None, constraints: NumberShuffleConstraints | None = None
) -> Board:
    import random

    rng = random.Random(seed)
    coords = axial_coords()
    resources = list(STANDARD_RESOURCES)
    numbers = list(STANDARD_NUMBER_TOKENS)

    rng.shuffle(resources)
    neighbors = build_tile_neighbors(coords)
    if constraints is None:
        constraints = NumberShuffleConstraints()

    tiles: Dict[int, HexTile] = {}
    non_desert_tiles = [
        tile_id for tile_id, resource in enumerate(resources) if resource != ResourceType.DESERT
    ]
    numbers_by_tile = _assign_numbers(non_desert_tiles, numbers, neighbors, rng, constraints)

    for tile_id, (q, r) in enumerate(coords):
        resource = resources[tile_id]
        if resource == ResourceType.DESERT:
            number_token = None
            has_robber = True
        else:
            number_token = numbers_by_tile[tile_id]
            has_robber = False

        tiles[tile_id] = HexTile(
            tile_id=tile_id,
            axial=(q, r),
            resource=resource,
            number_token=number_token,
            has_robber=has_robber,
        )

    graph = build_board_graph(coords)
    return Board(tiles=tiles, graph=graph, tile_neighbors=neighbors)
