#!/usr/bin/env python3
"""Analyze games where the agent performs poorly."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_fast import SmartAgent, run_game
from catan_rl.core.game.agents.random_agent import RandomAgent
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import apply_action
from catan_rl.core.game.engine.types import ActionType, ResourceType

# Run 50 games and find the low-VP ones
smart_agent = SmartAgent(player_id=0, seed=42)
smart_agent.epsilon = 0

low_vp_games = []
for i in range(50):
    agents = [smart_agent]
    for j in range(1, 4):
        agents.append(RandomAgent(player_id=j, seed=5000 + i * 4 + j))

    winner, final_vps, turns = run_game(agents, max_turns=2000, seed=5000 + i)

    if final_vps[0] < 7:  # Low VP games
        low_vp_games.append({
            'seed': 5000 + i,
            'my_vp': final_vps[0],
            'winner': winner,
            'turns': turns
        })

print(f'Found {len(low_vp_games)} low-VP games out of 50')

# Analyze a specific low-VP game in detail
if low_vp_games:
    game_to_analyze = low_vp_games[0]
    print(f'\nAnalyzing game with seed {game_to_analyze["seed"]} (VP={game_to_analyze["my_vp"]})')
    print('=' * 60)

    board = standard_board(seed=game_to_analyze['seed'])
    state = initial_game_state(board, num_players=4)

    smart_agent = SmartAgent(player_id=0, seed=42)
    smart_agent.epsilon = 0
    agents = [smart_agent]
    for j in range(1, 4):
        agents.append(RandomAgent(player_id=j, seed=game_to_analyze['seed'] + j))

    # Track the game
    my_actions = []
    turns = 0

    while state.winner is None and turns < 500:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            break

        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)

        if current_player == 0:
            my_actions.append(action.action_type.value)

        state = apply_action(state, action)
        turns += 1

        # Print state at key moments
        if turns == 20:  # After setup
            print(f'After setup (turn {turns}):')
            for pid in range(4):
                p = state.players[pid]
                print(f'  P{pid}: VP={p.victory_points}, S={len(p.settlements)}, R={len(p.roads)}, Res={sum(p.resources.values())}')

        if turns == 200:
            print(f'\nMid-game (turn {turns}):')
            for pid in range(4):
                p = state.players[pid]
                print(f'  P{pid}: VP={p.victory_points}, S={len(p.settlements)}, C={len(p.cities)}, R={len(p.roads)}')

    print(f'\nFinal state (turn {turns}):')
    for pid in range(4):
        p = state.players[pid]
        print(f'  P{pid}: VP={p.victory_points}, S={len(p.settlements)}, C={len(p.cities)}, R={len(p.roads)}')

    print(f'\nMy action distribution:')
    action_counts = {}
    for a in my_actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f'  {action_type}: {count}')
