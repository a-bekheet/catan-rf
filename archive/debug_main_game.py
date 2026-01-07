#!/usr/bin/env python3
"""Debug script to examine main game phase behavior."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catan.agents.rl_agent import RLAgent
from catan.agents.random_agent import RandomAgent
from catan.engine.board import standard_board
from catan.engine.game_state import initial_game_state

def debug_main_game():
    """Examine behavior during main game phase."""
    print("🎮 DEBUGGING MAIN GAME PHASE")
    print("=" * 50)

    # Create agents
    agents = [
        RLAgent(player_id=0, epsilon=0.1),
        RandomAgent(player_id=1),
        RandomAgent(player_id=2),
        RandomAgent(player_id=3),
    ]

    # Initialize game
    board = standard_board()
    state = initial_game_state(board, num_players=4)

    # Fast-forward through setup
    print("⚡ Fast-forwarding through setup phase...")
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

    print(f"Setup completed in {step} steps. Current phase: {state.phase.value}")

    # Now examine main game phase
    main_game_steps = 0
    dice_rolls = 0

    print(f"\n🎲 MAIN GAME ANALYSIS:")
    print("-" * 30)

    while state.phase.value != "setup" and state.winner is None and main_game_steps < 50:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            print(f"❌ No legal actions for player {current_player}")
            break

        # Show what RL agent sees
        if current_player == 0:
            state_key = agents[0]._state_key(state)
            print(f"\nStep {main_game_steps + 1} - RL Agent (P0):")
            print(f"  Phase: {state.phase.value}")
            print(f"  State: {state_key}")
            print(f"  Legal actions: {len(legal_actions)}")

            # Show action types available
            action_types = set(action.action_type.value for action in legal_actions)
            print(f"  Action types: {list(action_types)}")

            # Show current player resources
            player = state.players[current_player]
            resources = sum(player.resources.values())
            print(f"  Resources: {resources} total")
            print(f"  VP: {player.victory_points}")

        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)
        next_state = state.apply(action)

        # Track dice rolls
        if action.action_type.value == "roll_dice":
            dice_rolls += 1
            roll_value = next_state.last_roll if hasattr(next_state, 'last_roll') else "unknown"
            print(f"  🎲 Dice rolled: {roll_value}")

        # Check for wins
        if next_state.winner is not None:
            winner_type = "RL" if next_state.winner == 0 else "Random"
            print(f"🏆 GAME WON by Player {next_state.winner} ({winner_type})!")
            break

        state = next_state
        main_game_steps += 1

    print(f"\n📊 MAIN GAME SUMMARY:")
    print(f"  Steps in main game: {main_game_steps}")
    print(f"  Dice rolls: {dice_rolls}")
    print(f"  Winner: Player {state.winner if state.winner is not None else 'None'}")

    print(f"\n📈 FINAL PLAYER STATUS:")
    for i, player in state.players.items():
        agent_type = "RL" if i == 0 else "Random"
        print(f"  Player {i} ({agent_type}): {player.victory_points} VP")
        print(f"    Settlements: {len(player.settlements)}, Cities: {len(player.cities)}")
        print(f"    Roads: {len(player.roads)}, Dev cards: {len(player.dev_cards)}")
        print(f"    Resources: {sum(player.resources.values())}")

    # Check Q-table learning
    print(f"\n🧠 RL LEARNING STATUS:")
    rl_agent = agents[0]
    print(f"  Q-table size: {len(rl_agent.q_table)} states")
    print(f"  Current epsilon: {rl_agent.epsilon:.3f}")

if __name__ == "__main__":
    debug_main_game()