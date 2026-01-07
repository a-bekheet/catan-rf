#!/usr/bin/env python3
"""Debug what the enhanced RL agent is actually doing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catan.agents.rl_agent import RLAgent
from catan.agents.random_agent import RandomAgent
from catan.engine.board import standard_board
from catan.engine.game_state import initial_game_state

def debug_enhanced_agent():
    """Debug the enhanced RL agent's decision making."""
    print("🔍 DEBUGGING ENHANCED RL AGENT DECISIONS")
    print("=" * 60)

    # Create enhanced RL agent
    rl_agent = RLAgent(player_id=0, epsilon=0.0)  # No exploration for debugging
    agents = [rl_agent, RandomAgent(1), RandomAgent(2), RandomAgent(3)]

    board = standard_board()
    state = initial_game_state(board, num_players=4)

    # Fast-forward through setup
    step = 0
    while state.phase.value == "setup" and step < 20:
        current_player = state.current_player
        legal_actions = state.legal_actions()
        if not legal_actions:
            break
        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)
        state = state.apply(action)
        step += 1

    print(f"Setup completed in {step} steps")

    # Monitor RL agent decisions in main game
    main_step = 0
    rl_decisions = []

    while state.winner is None and main_step < 30:  # Monitor first 30 main game steps
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            break

        agent = agents[current_player]

        if current_player == 0:  # RL agent
            player = state.players[0]

            print(f"\n🤖 RL AGENT TURN {main_step + 1}")
            print(f"   State: {rl_agent._state_key(state)}")
            print(f"   Resources: {dict(player.resources)} (total: {sum(player.resources.values())})")
            print(f"   VP: {player.victory_points}, Buildings: {len(player.settlements)} settlements, {len(player.cities)} cities, {len(player.roads)} roads")

            # Show legal action types
            action_types = {}
            for action in legal_actions:
                action_type = action.action_type.value
                action_types[action_type] = action_types.get(action_type, 0) + 1

            print(f"   Legal actions: {len(legal_actions)} total")
            print(f"   Action types: {action_types}")

            # Show settlement building capability
            from catan.engine.rules import COSTS, _can_afford
            from catan.engine.types import ActionType
            settlement_cost = COSTS[ActionType.BUILD_SETTLEMENT]
            can_afford_settlement = _can_afford(player.resources, settlement_cost)
            settlement_spots = rl_agent._count_valid_settlement_spots(state, player)

            print(f"   Settlement analysis:")
            print(f"     Can afford: {can_afford_settlement}")
            print(f"     Valid spots: {settlement_spots}")
            print(f"     Settlement actions available: {sum(1 for a in legal_actions if a.action_type == ActionType.BUILD_SETTLEMENT)}")

        action = agent.select_action(state, legal_actions)
        next_state = state.apply(action)

        # Track RL agent actions and rewards
        if current_player == 0:
            reward = rl_agent.compute_reward(state, next_state)
            action_type = action.action_type.value
            rl_decisions.append((action_type, reward))

            print(f"   → Action taken: {action_type}")
            print(f"   → Reward: {reward:.2f}")

            # Show state change
            prev_player = state.players[0]
            new_player = next_state.players[0]
            if new_player.victory_points != prev_player.victory_points:
                print(f"   🎯 VP changed: {prev_player.victory_points} → {new_player.victory_points}")

        state = next_state
        main_step += 1

    print(f"\n📊 RL AGENT DECISION SUMMARY (first {len(rl_decisions)} turns):")
    action_count = {}
    total_reward = 0
    for action_type, reward in rl_decisions:
        action_count[action_type] = action_count.get(action_type, 0) + 1
        total_reward += reward

    for action_type, count in sorted(action_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   {action_type}: {count} times")

    print(f"\n   Average reward per turn: {total_reward / len(rl_decisions) if rl_decisions else 0:.2f}")

    # Final state
    final_player = state.players[0]
    print(f"\n🏁 FINAL RL AGENT STATE:")
    print(f"   VP: {final_player.victory_points}")
    print(f"   Settlements: {len(final_player.settlements)} (locations: {list(final_player.settlements)})")
    print(f"   Cities: {len(final_player.cities)}")
    print(f"   Roads: {len(final_player.roads)} (locations: {list(final_player.roads)})")
    print(f"   Resources: {sum(final_player.resources.values())} total")

if __name__ == "__main__":
    debug_enhanced_agent()