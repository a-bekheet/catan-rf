#!/usr/bin/env python3
"""Test the enhanced RL agent to see if it can now win games."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catan.agents.rl_agent import RLAgent
from catan.agents.random_agent import RandomAgent
from catan.engine.board import standard_board
from catan.engine.game_state import initial_game_state

def test_enhanced_agent():
    """Test if the enhanced RL agent can win games."""
    print("🧪 TESTING ENHANCED RL AGENT")
    print("=" * 50)

    wins = 0
    total_games = 10

    for game_num in range(total_games):
        print(f"\n🎮 Game {game_num + 1}/{total_games}")

        # Create agents - enhanced RL vs 3 random
        agents = [
            RLAgent(player_id=0, epsilon=0.1),  # Some exploration
            RandomAgent(player_id=1),
            RandomAgent(player_id=2),
            RandomAgent(player_id=3),
        ]

        # Initialize game
        board = standard_board()
        state = initial_game_state(board, num_players=4)

        # Reset agents
        for agent in agents:
            agent.reset()

        step_count = 0
        max_steps = 200  # Reasonable game length

        while step_count < max_steps and state.winner is None:
            current_player = state.current_player
            legal_actions = state.legal_actions()

            if not legal_actions:
                print(f"   ⚠️  No legal actions for player {current_player}")
                break

            try:
                agent = agents[current_player]
                action = agent.select_action(state, legal_actions)
                next_state = state.apply(action)

                # Update RL agent
                if hasattr(agent, 'compute_reward'):
                    reward = agent.compute_reward(state, next_state)
                    if hasattr(agent, 'update'):
                        agent.update(state, action, reward, next_state)

                state = next_state
                step_count += 1

                # Progress indicator
                if step_count % 50 == 0:
                    vps = [state.players[i].victory_points for i in range(4)]
                    print(f"   Step {step_count}: VP={vps}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                break

        # Check results
        if state.winner is not None:
            winner_name = ["RL Agent", "Random 1", "Random 2", "Random 3"][state.winner]
            print(f"   🏆 Winner: Player {state.winner} ({winner_name}) in {step_count} steps")

            if state.winner == 0:  # RL agent won
                wins += 1
                print("   ✅ RL AGENT VICTORY!")

            # Show final scores
            for i in range(4):
                player = state.players[i]
                agent_type = "RL" if i == 0 else "Random"
                print(f"   P{i} ({agent_type}): {player.victory_points} VP, "
                      f"{len(player.settlements)} settlements, {len(player.cities)} cities")
        else:
            print(f"   ⏰ Timeout after {step_count} steps")
            vps = [state.players[i].victory_points for i in range(4)]
            print(f"   Final VP: {vps}")

    print(f"\n📊 FINAL RESULTS:")
    print(f"   RL Agent wins: {wins}/{total_games} ({wins/total_games*100:.1f}%)")
    print(f"   Random agent wins: {total_games-wins}/{total_games} ({(total_games-wins)/total_games*100:.1f}%)")

    if wins > 0:
        print("   🎉 SUCCESS! RL agent can now win games!")
    else:
        print("   ❌ Still no wins. Need further improvements.")

    return wins > 0

if __name__ == "__main__":
    test_enhanced_agent()