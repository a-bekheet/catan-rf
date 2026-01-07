#!/usr/bin/env python3
"""Debug script to examine why settlement building isn't working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catan.agents.rl_agent import RLAgent
from catan.agents.random_agent import RandomAgent
from catan.engine.board import standard_board
from catan.engine.game_state import initial_game_state
from catan.engine.rules import COSTS, _can_afford, _settlement_distance_ok, _player_has_road_touching
from catan.engine.types import ActionType, ResourceType

def debug_settlement_building():
    """Debug why settlement building actions aren't generated."""
    print("🔧 DEBUGGING SETTLEMENT BUILDING")
    print("=" * 50)

    # Create agents and fast-forward to main game
    agents = [
        RLAgent(player_id=0, epsilon=0.0),
        RandomAgent(player_id=1),
        RandomAgent(player_id=2),
        RandomAgent(player_id=3),
    ]

    board = standard_board()
    state = initial_game_state(board, num_players=4)

    # Fast-forward through setup
    step = 0
    while state.phase.value == "setup" and step < 100:
        current_player = state.current_player
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)
        state = state.apply(action)
        step += 1

    # Now in main game - find a state where RL agent has resources
    print("🎲 Simulating to find resource-rich state...")
    main_step = 0
    while state.winner is None and main_step < 100:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            break

        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)
        state = state.apply(action)
        main_step += 1

        # Check RL agent resources
        if current_player == 0:
            player = state.players[0]
            total_resources = sum(player.resources.values())
            if total_resources >= 8:  # Likely has settlement resources
                break

    print(f"\n📊 SETTLEMENT BUILDING ANALYSIS (after {main_step} steps)")
    print("-" * 50)

    player = state.players[0]  # RL agent
    print(f"🤖 RL Agent (Player 0) Status:")
    print(f"  Victory Points: {player.victory_points}")
    print(f"  Settlements: {len(player.settlements)} (locations: {list(player.settlements)})")
    print(f"  Cities: {len(player.cities)}")
    print(f"  Roads: {len(player.roads)} (locations: {list(player.roads)})")

    print(f"\n💰 RESOURCES:")
    for resource, count in player.resources.items():
        print(f"  {resource.value}: {count}")

    print(f"  Total: {sum(player.resources.values())}")

    # Check settlement requirements
    settlement_cost = COSTS[ActionType.BUILD_SETTLEMENT]
    print(f"\n🏠 SETTLEMENT REQUIREMENTS:")
    print("  Cost:")
    for resource, cost in settlement_cost.items():
        if cost > 0:
            has = player.resources[resource]
            print(f"    {resource.value}: {cost} needed, {has} available ({'✅' if has >= cost else '❌'})")

    can_afford = _can_afford(player.resources, settlement_cost)
    print(f"  Can afford: {'✅' if can_afford else '❌'}")

    settlement_limit = len(player.settlements) < 5
    print(f"  Under limit (5): {'✅' if settlement_limit else '❌'} ({len(player.settlements)}/5)")

    # Check every vertex for settlement viability
    print(f"\n🗺️  VERTEX ANALYSIS:")
    valid_vertices = []

    for vertex_id in state.board.graph.vertices:
        # Check distance rule
        distance_ok = _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id)

        # Check road connection
        road_connected = _player_has_road_touching(state.board, player, vertex_id)

        if distance_ok and road_connected:
            valid_vertices.append(vertex_id)
            print(f"  Vertex {vertex_id}: ✅ Valid (distance: {distance_ok}, road: {road_connected})")
        elif distance_ok:
            print(f"  Vertex {vertex_id}: ❌ No road connection")
        elif road_connected:
            print(f"  Vertex {vertex_id}: ❌ Too close to settlement")

    print(f"\n📍 TOTAL VALID SETTLEMENT LOCATIONS: {len(valid_vertices)}")

    # Show what legal actions are actually generated
    legal_actions = state.legal_actions()
    settlement_actions = [a for a in legal_actions if a.action_type == ActionType.BUILD_SETTLEMENT]

    print(f"\n⚖️  LEGAL ACTIONS:")
    print(f"  Total: {len(legal_actions)}")
    print(f"  Settlement actions: {len(settlement_actions)}")

    action_types = {}
    for action in legal_actions:
        action_type = action.action_type.value
        action_types[action_type] = action_types.get(action_type, 0) + 1

    print("  By type:")
    for action_type, count in action_types.items():
        print(f"    {action_type}: {count}")

    if settlement_actions:
        print("  Settlement vertices available:")
        for action in settlement_actions[:5]:  # Show first 5
            vertex_id = action.payload["vertex_id"]
            print(f"    Vertex {vertex_id}")
    else:
        print("  ❌ NO SETTLEMENT ACTIONS GENERATED")

    # Manual settlement building check
    if can_afford and settlement_limit and valid_vertices:
        print(f"\n✅ SHOULD BE ABLE TO BUILD SETTLEMENTS!")
        print(f"   Player can afford: {can_afford}")
        print(f"   Under piece limit: {settlement_limit}")
        print(f"   Valid locations: {len(valid_vertices)}")
    else:
        print(f"\n❌ CANNOT BUILD SETTLEMENTS:")
        if not can_afford:
            print(f"   Missing resources")
        if not settlement_limit:
            print(f"   At piece limit ({len(player.settlements)}/5)")
        if not valid_vertices:
            print(f"   No valid locations")

    # Show board state for debugging
    print(f"\n🎲 GAME STATE:")
    print(f"  Phase: {state.phase.value}")
    print(f"  Current player: {state.current_player}")
    print(f"  Turn index: {state.turn_index}")

    return state

if __name__ == "__main__":
    debug_settlement_building()