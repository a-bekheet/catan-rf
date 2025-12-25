from catan.engine.board import standard_board
from catan.engine.types import ResourceType


def test_board_counts():
    board = standard_board(seed=42)
    assert len(board.tiles) == 19
    assert len(board.graph.vertices) == 54
    assert len(board.graph.edges) == 72


def test_desert_has_robber_and_no_number():
    board = standard_board(seed=7)
    deserts = [tile for tile in board.tiles.values() if tile.resource == ResourceType.DESERT]
    assert len(deserts) == 1
    assert deserts[0].has_robber is True
    assert deserts[0].number_token is None


def test_no_adjacent_six_or_eight():
    board = standard_board(seed=11)
    for tile_id, tile in board.tiles.items():
        if tile.number_token not in (6, 8):
            continue
        for neighbor_id in board.tile_neighbors[tile_id]:
            neighbor = board.tiles[neighbor_id]
            assert neighbor.number_token not in (6, 8)
