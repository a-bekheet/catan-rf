from catan.engine.board import standard_board
from catan.engine.game_state import TurnPhase, initial_game_state
from catan.engine.types import Action, ActionType, BuildingType, ResourceType


def test_setup_settlement_and_road_progresses_turn():
    board = standard_board(seed=1)
    state = initial_game_state(board)
    vertex_id = next(iter(board.graph.vertices.keys()))
    edge_id = board.edges_for_vertex(vertex_id)[0]

    state = state.apply(Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": vertex_id}))
    assert state.pending_setup_vertex == vertex_id

    state = state.apply(Action(ActionType.BUILD_ROAD, {"edge_id": edge_id}))
    assert state.pending_setup_vertex is None
    assert state.current_player == 1


def test_roll_distributes_resources_to_settlement():
    board = standard_board(seed=3)
    state = initial_game_state(board)

    target_tile = next(
        tile
        for tile in board.tiles.values()
        if tile.resource != ResourceType.DESERT and tile.number_token is not None
    )
    vertex_id = board.graph.hex_to_vertices[target_tile.tile_id][0]

    state.phase = TurnPhase.ROLL
    state.vertex_occupancy[vertex_id] = (0, BuildingType.SETTLEMENT)
    state.players[0].settlements.add(vertex_id)

    before = state.players[0].resources[target_tile.resource]
    state = state.apply(Action(ActionType.ROLL_DICE, {"roll": target_tile.number_token}))
    after = state.players[0].resources[target_tile.resource]

    assert after == before + 1
