from catan.engine.board import standard_board
from catan.engine.game_state import GameState, PlayerState, TurnPhase, initial_game_state
from catan.engine.types import Action, ActionType, DevCardType, ResourceType


def test_buy_dev_card():
    """Test buying a development card."""
    board = standard_board(seed=42)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give player resources to buy dev card
    state.players[0].resources[ResourceType.ORE] = 1
    state.players[0].resources[ResourceType.GRAIN] = 1
    state.players[0].resources[ResourceType.WOOL] = 1

    initial_deck_size = len(state.dev_deck)

    # Buy development card
    state = state.apply(Action(ActionType.BUY_DEV_CARD, {}))

    # Check resources were spent
    assert state.players[0].resources[ResourceType.ORE] == 0
    assert state.players[0].resources[ResourceType.GRAIN] == 0
    assert state.players[0].resources[ResourceType.WOOL] == 0

    # Check deck decreased
    assert len(state.dev_deck) == initial_deck_size - 1

    # Check card was added to new_dev_cards (can't be played this turn)
    assert len(state.new_dev_cards[0]) == 1


def test_victory_point_cards_immediate():
    """Test that victory point cards are immediately added to hand."""
    board = standard_board(seed=1)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN
    state.current_player = 0

    # Force a victory point card to be next
    state.dev_deck = [DevCardType.VICTORY_POINT]

    # Give player resources
    state.players[0].resources[ResourceType.ORE] = 1
    state.players[0].resources[ResourceType.GRAIN] = 1
    state.players[0].resources[ResourceType.WOOL] = 1

    initial_vp = state.players[0].victory_points

    # Buy the VP card
    state = state.apply(Action(ActionType.BUY_DEV_CARD, {}))

    # VP card should be in player's hand immediately
    assert DevCardType.VICTORY_POINT in state.players[0].dev_cards
    # But victory_points field shouldn't increase until win condition check
    assert state.players[0].victory_points == initial_vp


def test_knight_card_moves_robber():
    """Test playing a knight card moves the robber."""
    board = standard_board(seed=5)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give player a knight card
    state.players[0].dev_cards.append(DevCardType.KNIGHT)

    initial_robber = state.robber_tile
    target_tile = None
    for tile_id in state.board.tiles:
        if tile_id != initial_robber:
            target_tile = tile_id
            break

    # Play knight card
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.KNIGHT.value,
        "tile_id": target_tile
    }))

    # Check robber moved
    assert state.robber_tile == target_tile
    assert state.robber_tile != initial_robber

    # Check knight was played
    assert state.players[0].knights_played == 1
    assert state.played_dev_card_this_turn is True

    # Check card was removed from hand
    assert DevCardType.KNIGHT not in state.players[0].dev_cards


def test_monopoly_card():
    """Test monopoly card steals all of a resource from other players."""
    board = standard_board(seed=3)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give players resources
    state.players[0].resources[ResourceType.BRICK] = 1
    state.players[1].resources[ResourceType.BRICK] = 3
    state.players[2].resources[ResourceType.BRICK] = 2
    state.players[3].resources[ResourceType.BRICK] = 1

    # Give player monopoly card
    state.players[0].dev_cards.append(DevCardType.MONOPOLY)

    # Play monopoly on brick
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.MONOPOLY.value,
        "resource": ResourceType.BRICK.value
    }))

    # Player 0 should have all brick (1 original + 3+2+1 stolen)
    assert state.players[0].resources[ResourceType.BRICK] == 7

    # Other players should have no brick
    assert state.players[1].resources[ResourceType.BRICK] == 0
    assert state.players[2].resources[ResourceType.BRICK] == 0
    assert state.players[3].resources[ResourceType.BRICK] == 0


def test_year_of_plenty_card():
    """Test year of plenty card gives 2 resources from bank."""
    board = standard_board(seed=7)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give player year of plenty card
    state.players[0].dev_cards.append(DevCardType.YEAR_OF_PLENTY)

    initial_ore = state.players[0].resources[ResourceType.ORE]
    initial_wool = state.players[0].resources[ResourceType.WOOL]

    # Play year of plenty
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.YEAR_OF_PLENTY.value,
        "resource1": ResourceType.ORE.value,
        "resource2": ResourceType.WOOL.value
    }))

    # Should have gained the resources
    assert state.players[0].resources[ResourceType.ORE] == initial_ore + 1
    assert state.players[0].resources[ResourceType.WOOL] == initial_wool + 1


def test_road_building_card():
    """Test road building card places 2 free roads."""
    board = standard_board(seed=2)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Set up player with a settlement to build roads from
    vertex_id = list(board.graph.vertices.keys())[0]
    state.vertex_occupancy[vertex_id] = (0, state.players[0])
    state.players[0].settlements.add(vertex_id)

    # Give player road building card
    state.players[0].dev_cards.append(DevCardType.ROAD_BUILDING)

    # Find available edges
    available_edges = []
    for edge_id in board.edges_for_vertex(vertex_id):
        if edge_id not in state.edge_occupancy:
            available_edges.append(edge_id)

    road1, road2 = available_edges[:2]
    initial_roads = len(state.players[0].roads)

    # Play road building
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.ROAD_BUILDING.value,
        "roads": [road1, road2]
    }))

    # Should have built 2 roads for free
    assert len(state.players[0].roads) == initial_roads + 2
    assert road1 in state.players[0].roads
    assert road2 in state.players[0].roads


def test_largest_army_transfer():
    """Test largest army transfers immediately when surpassed."""
    board = standard_board(seed=9)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Player 0 has largest army with 3 knights
    state.players[0].knights_played = 3
    state.players[0].victory_points += 2  # Largest army VPs

    # Player 1 has 2 knights
    state.players[1].knights_played = 2

    # Give player 1 a knight card
    state.players[1].dev_cards.append(DevCardType.KNIGHT)

    # Find a tile to move robber to
    target_tile = None
    for tile_id in state.board.tiles:
        if tile_id != state.robber_tile:
            target_tile = tile_id
            break

    # Player 1 plays knight (now has 3 knights, tied)
    state.current_player = 1
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.KNIGHT.value,
        "tile_id": target_tile
    }))

    # Player 0 should still have largest army (tie goes to first holder)
    assert state.players[0].victory_points >= 2  # Still has largest army
    assert state.players[1].knights_played == 3

    # Give player 1 another knight
    state.players[1].dev_cards.append(DevCardType.KNIGHT)
    state.played_dev_card_this_turn = False  # Reset for next card

    # Find a different tile to move robber to (robber is now on target_tile)
    target_tile2 = None
    for tile_id in state.board.tiles:
        if tile_id != state.robber_tile:
            target_tile2 = tile_id
            break

    # Player 1 plays another knight (now has 4 knights, more than player 0)
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.KNIGHT.value,
        "tile_id": target_tile2
    }))

    # Largest army should transfer to player 1
    assert state.players[1].victory_points >= 2  # Now has largest army
    assert state.players[1].knights_played == 4


def test_cannot_play_card_bought_this_turn():
    """Test that cards bought this turn cannot be played."""
    board = standard_board(seed=4)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Force knight to be next card
    state.dev_deck = [DevCardType.KNIGHT]

    # Give player resources
    state.players[0].resources[ResourceType.ORE] = 1
    state.players[0].resources[ResourceType.GRAIN] = 1
    state.players[0].resources[ResourceType.WOOL] = 1

    # Buy dev card
    state = state.apply(Action(ActionType.BUY_DEV_CARD, {}))

    # Card should be in new_dev_cards, not in playable dev_cards
    assert DevCardType.KNIGHT not in state.players[0].dev_cards
    assert DevCardType.KNIGHT in state.new_dev_cards[0]


def test_cards_become_playable_next_turn():
    """Test that cards bought become playable next turn."""
    board = standard_board(seed=6)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Force knight to be next card
    state.dev_deck = [DevCardType.KNIGHT]

    # Give player resources
    state.players[0].resources[ResourceType.ORE] = 1
    state.players[0].resources[ResourceType.GRAIN] = 1
    state.players[0].resources[ResourceType.WOOL] = 1

    # Buy dev card
    state = state.apply(Action(ActionType.BUY_DEV_CARD, {}))

    # Pass turn to trigger card transfer
    state = state.apply(Action(ActionType.PASS_TURN, {}))

    # Card should now be in playable hand
    assert DevCardType.KNIGHT in state.players[0].dev_cards
    assert len(state.new_dev_cards[0]) == 0


def test_victory_with_dev_cards():
    """Test winning with victory point development cards."""
    board = standard_board(seed=8)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give player 9 visible victory points
    state.players[0].victory_points = 9

    # Give player a victory point card (should trigger win)
    state.players[0].dev_cards.append(DevCardType.VICTORY_POINT)

    # Check winner when turn ends or action happens
    from catan.engine.rules import _check_winner
    _check_winner(state)

    # Player should have won
    assert state.winner == 0
    assert state.phase == TurnPhase.END


def test_one_dev_card_per_turn_limit():
    """Test that only one dev card can be played per turn."""
    board = standard_board(seed=10)
    state = initial_game_state(board)
    state.phase = TurnPhase.MAIN

    # Give player multiple dev cards
    state.players[0].dev_cards.extend([DevCardType.KNIGHT, DevCardType.MONOPOLY])

    # Find a tile to move robber
    target_tile = None
    for tile_id in state.board.tiles:
        if tile_id != state.robber_tile:
            target_tile = tile_id
            break

    # Play first dev card (knight)
    state = state.apply(Action(ActionType.PLAY_DEV_CARD, {
        "dev_card": DevCardType.KNIGHT.value,
        "tile_id": target_tile
    }))

    # Should not be able to play second dev card
    legal_actions = state.legal_actions()
    play_dev_actions = [a for a in legal_actions if a.action_type == ActionType.PLAY_DEV_CARD]
    assert len(play_dev_actions) == 0

    assert state.played_dev_card_this_turn is True