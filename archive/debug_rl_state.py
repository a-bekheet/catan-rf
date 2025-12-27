#!/usr/bin/env python3
"""Debug script to examine RL agent state representation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catan.agents.rl_agent import RLAgent
from catan.agents.random_agent import RandomAgent
from catan.engine.board import standard_board
from catan.engine.game_state import initial_game_state

def debug_rl_state():
    """Examine what state information the RL agent sees."""
    print("🔍 DEBUGGING RL AGENT STATE REPRESENTATION")
    print("=" * 60)

    # Create a simple game setup
    board = standard_board()
    state = initial_game_state(board, num_players=4)

    # Create RL agent
    rl_agent = RLAgent(player_id=0, epsilon=0.0)  # No exploration for debugging

    print("\n📊 INITIAL STATE ANALYSIS")
    print("-" * 30)

    # Show what the RL agent sees in the initial state
    state_key = rl_agent._state_key(state)
    print(f"State representation: {state_key}")

    # Break down the state representation
    print("\n🔍 STATE BREAKDOWN:")
    parts = state_key.split("_")
    for part in parts:
        print(f"  {part}")

    # Show legal actions
    legal_actions = state.legal_actions()
    print(f"\n⚖️  Legal actions available: {len(legal_actions)}")

    action_keys = set()
    for action in legal_actions[:10]:  # Show first 10
        action_key = rl_agent._action_key(action)
        action_keys.add(action_key)
        print(f"  {action.action_type.value}: {action_key}")
        if action.payload:
            print(f"    Payload: {action.payload}")

    print(f"\n🎯 Unique action keys: {len(action_keys)} out of {len(legal_actions)} total actions")

    # Show Q-table state
    print(f"\n🧠 Q-TABLE STATUS:")
    print(f"  Total states learned: {len(rl_agent.q_table)}")

    if state_key in rl_agent.q_table:
        state_q_values = rl_agent.q_table[state_key]
        print(f"  Actions for this state: {len(state_q_values)}")
        print("  Top 5 Q-values:")
        sorted_actions = sorted(state_q_values.items(), key=lambda x: x[1], reverse=True)
        for action_key, q_val in sorted_actions[:5]:
            print(f"    {action_key}: {q_val:.3f}")
    else:
        print("  This state not yet in Q-table")

    # Simulate a few steps to see how state representation changes
    print(f"\n🎮 SIMULATING GAME STEPS:")
    print("-" * 30)

    step = 0
    agents = [rl_agent, RandomAgent(1), RandomAgent(2), RandomAgent(3)]

    while step < 10 and state.winner is None:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            break

        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)
        next_state = state.apply(action)

        if current_player == 0:  # RL agent
            print(f"\nStep {step + 1} - RL Agent Turn:")
            print(f"  Action: {action.action_type.value}")
            print(f"  Current state: {rl_agent._state_key(state)}")
            print(f"  Next state: {rl_agent._state_key(next_state)}")

            # Check if RL agent can win
            if hasattr(action, 'action_type'):
                reward = rl_agent.compute_reward(state, next_state)
                print(f"  Reward: {reward:.2f}")

        state = next_state
        step += 1

    print(f"\n📈 GAME STATUS AFTER {step} STEPS:")
    for i, player in state.players.items():
        print(f"  Player {i}: {player.victory_points} VP, {len(player.settlements)} settlements, {len(player.cities)} cities")

if __name__ == "__main__":
    debug_rl_state()