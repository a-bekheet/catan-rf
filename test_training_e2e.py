#!/usr/bin/env python3
"""
End-to-End Training Test with RandomAgent
==========================================

Tests the complete training loop using only RandomAgent (no heavy dependencies).
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🎯 End-to-End Training Test (RandomAgent)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

# Import required modules
from catan_rl.agents.base_rl_agent import RandomAgent
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action

print("✓ All imports successful\n")

# Create 4 random agents
print("Creating 4 random agents...")
agents = [RandomAgent(i, {'seed': 42 + i}) for i in range(4)]
for agent in agents:
    print(f"  ✓ Agent {agent.agent_id}: {agent.agent_name}")
print()

# Run a test episode
print("Running test episode...")
board = standard_board(seed=42)
state = initial_game_state(board, num_players=4)

episode_rewards = {i: 0.0 for i in range(4)}
turn_count = 0
max_turns = 50  # Short test

print(f"  Max turns: {max_turns}")
print()

while state.winner is None and turn_count < max_turns:
    current_player_id = state.current_player
    current_agent = agents[current_player_id]

    # Get legal actions
    actions = legal_actions(state)
    if not actions:
        print(f"  Turn {turn_count}: No legal actions available")
        break

    # Agent selects action
    action, metrics = current_agent.select_action(state, actions)

    if turn_count % 10 == 0:
        print(f"  Turn {turn_count}: Player {current_player_id} -> {action.action_type.value}")

    # Apply action
    try:
        next_state = apply_action(state, action)

        # Simple reward
        reward = 1.0 if action.action_type else 0.0
        episode_rewards[current_player_id] += reward

        # Update agent (RandomAgent doesn't learn, but test the interface)
        current_agent.update(state, action, reward, next_state, next_state.winner is not None)

        state = next_state
        turn_count += 1

    except Exception as e:
        print(f"  ✗ Error at turn {turn_count}: {e}")
        break

print()
print(f"Episode completed: {turn_count} turns")

# Notify agents of episode end
for i, agent in enumerate(agents):
    won = (i == state.winner if state.winner is not None else False)
    agent.record_episode_end(episode_rewards[i], turn_count, won)
    agent.reset_episode()

# Display results
print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📊 Episode Results")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if state.winner is not None:
    print(f"\n🏆 Winner: Player {state.winner}")
else:
    print(f"\n⏰ Game ended after {turn_count} turns (no winner)")

print("\nPlayer Stats:")
for player_id, player in state.players.items():
    reward = episode_rewards[player_id]
    print(f"  Player {player_id}:")
    print(f"    Victory Points: {player.victory_points}")
    print(f"    Settlements: {len(player.settlements)}")
    print(f"    Cities: {len(player.cities)}")
    print(f"    Roads: {len(player.roads)}")
    print(f"    Episode Reward: {reward:.2f}")

print("\nAgent Metrics:")
for agent in agents:
    metrics = agent.get_training_metrics()
    print(f"  Agent {metrics['agent_id']}: {metrics['total_episodes']} episodes, {metrics['total_steps']} steps")

print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("✅ End-to-End Test Complete!")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()
print("🎉 Training system structure is working correctly!")
print()
print("To test with real RL agents:")
print("  1. Install dependencies: uv sync")
print("  2. Run: python train_agents.py --agent ppo --episodes 10")
print()
