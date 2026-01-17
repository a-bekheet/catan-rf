#!/usr/bin/env python3
"""Analyze game outcomes."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_fast import SmartAgent, run_game
from catan_rl.core.game.agents.random_agent import RandomAgent

# Run 20 games and analyze results
smart_agent = SmartAgent(player_id=0, seed=42)
smart_agent.epsilon = 0  # No exploration

results = []
for i in range(20):
    agents = [smart_agent]
    for j in range(1, 4):
        agents.append(RandomAgent(player_id=j, seed=1000 + i * 4 + j))

    winner, final_vps, turns = run_game(agents, max_turns=2000, seed=1000 + i)
    results.append({
        'winner': winner,
        'my_vp': final_vps[0],
        'opponent_vps': [final_vps[j] for j in range(1, 4)],
        'turns': turns,
        'max_opp_vp': max(final_vps[j] for j in range(1, 4))
    })

print('Game Analysis (20 games):')
print('=' * 60)
wins = sum(1 for r in results if r['winner'] == 0)
print(f'Wins: {wins}/20 ({100*wins/20:.0f}%)')
print(f'Avg VP: {sum(r["my_vp"] for r in results)/20:.1f}')
print(f'Avg turns: {sum(r["turns"] for r in results)/20:.0f}')

print('\nVP Distribution:')
for r in results:
    status = 'WIN' if r['winner'] == 0 else 'LOSS' if r['winner'] else 'TIMEOUT'
    print(f'  Me:{r["my_vp"]:2d} vs Opp:{r["max_opp_vp"]:2d} | Turns:{r["turns"]:4d} | {status}')

# Why did we lose?
losses = [r for r in results if r['winner'] is not None and r['winner'] != 0]
timeouts = [r for r in results if r['winner'] is None]
print(f'\nLosses: {len(losses)}, Timeouts: {len(timeouts)}')
if losses:
    avg_loss_vp = sum(r['my_vp'] for r in losses) / len(losses)
    avg_winner_vp = sum(r['max_opp_vp'] for r in losses) / len(losses)
    print(f'In losses: Our avg VP={avg_loss_vp:.1f}, Winner avg VP={avg_winner_vp:.1f}')
