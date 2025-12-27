#!/usr/bin/env python3
"""
Demonstration script to verify DQN integration with existing game engine.

This script runs a quick test to ensure the integration is working correctly
and provides a simple example of how to use the new system.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.core.integration import DQNAgentAdapter
from catan_rl.agents.dqn_agent import DQNAgentFactory
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import apply_action


def test_dqn_agent_creation():
    """Test that all DQN agent types can be created."""
    print("Testing DQN agent creation...")

    try:
        # Test factory methods
        baseline = DQNAgentFactory.create_baseline_agent(0)
        spatial = DQNAgentFactory.create_spatial_agent(0)
        rainbow = DQNAgentFactory.create_rainbow_agent(0)

        print("✓ All agent types created successfully")

        # Test adapter
        adapter = DQNAgentAdapter(0, 'baseline')
        print("✓ DQN adapter created successfully")

        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False


def test_game_integration():
    """Test that DQN agents can play in the game engine."""
    print("\nTesting game integration...")

    try:
        # Create agents
        agents = [
            DQNAgentAdapter(0, 'baseline'),
            DQNAgentAdapter(1, 'spatial'),
            DQNAgentAdapter(2, 'rainbow'),
            DQNAgentAdapter(3, 'baseline')
        ]
        print("✓ Created 4 DQN agents")

        # Set up game
        board = standard_board(seed=42)
        game_state = initial_game_state(board, num_players=4)
        print("✓ Game state initialized")

        # Reset agents
        for agent in agents:
            agent.reset()
        print("✓ Agents reset")

        # Run a few turns
        turns_completed = 0
        max_turns = 20  # Just test a few turns

        while game_state.winner is None and turns_completed < max_turns:
            current_player_id = game_state.current_player
            current_agent = agents[current_player_id]

            # Get legal actions
            legal_actions = game_state.legal_actions()
            if not legal_actions:
                print(f"No legal actions for player {current_player_id}")
                break

            # Agent selects action
            action = current_agent.select_action(game_state, legal_actions)
            print(f"Turn {turns_completed}: Player {current_player_id} selected action {action.action_type.value}")

            # Apply action
            prev_state = game_state
            game_state = apply_action(game_state, action)

            # Update agent
            reward = current_agent.compute_reward(prev_state, game_state)
            current_agent.update(prev_state, action, reward, game_state)

            turns_completed += 1

        print(f"✓ Completed {turns_completed} turns successfully")

        # Check agent metrics
        for i, agent in enumerate(agents):
            metrics = agent.get_dqn_metrics()
            print(f"  Agent {i}: {len(agent.bridge.dqn_agent.replay_buffer)} experiences, "
                  f"epsilon={metrics.get('epsilon', 'N/A'):.3f}")

        return True
    except Exception as e:
        print(f"✗ Game integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_encoders():
    """Test different state encoders."""
    print("\nTesting state encoders...")

    try:
        from catan_rl.environments.state_encoders import StateEncoderFactory

        # Create a simple game state for testing
        board = standard_board(seed=42)
        game_state = initial_game_state(board, num_players=4)

        # Test feature encoder
        feature_encoder = StateEncoderFactory.create('feature')
        feature_state = feature_encoder.encode(game_state, 0)
        print(f"✓ Feature encoder: {feature_state.shape}")

        # Test spatial encoder
        spatial_encoder = StateEncoderFactory.create('spatial')
        spatial_state = spatial_encoder.encode(game_state, 0)
        print(f"✓ Spatial encoder: {spatial_state.shape}")

        return True
    except Exception as e:
        print(f"✗ State encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_space():
    """Test action space functionality."""
    print("\nTesting action space...")

    try:
        from catan_rl.environments.action_space import CatanActionSpace

        # Create action space
        action_space = CatanActionSpace()
        print(f"✓ Action space size: {action_space.get_action_space_size()}")

        # Create game state
        board = standard_board(seed=42)
        game_state = initial_game_state(board, num_players=4)

        # Get action mask
        action_mask = action_space.get_action_mask(game_state, 0)
        valid_actions = action_mask.sum().item()
        print(f"✓ Action mask: {valid_actions} valid actions out of {len(action_mask)}")

        # Test action encoding/decoding
        legal_actions = game_state.legal_actions()
        if legal_actions:
            first_action = legal_actions[0]
            encoded = action_space.encode_action(first_action)
            decoded = action_space.decode_action(encoded)
            print(f"✓ Action encoding/decoding: {first_action.action_type.value} -> {encoded} -> {decoded.action_type.value}")

        return True
    except Exception as e:
        print(f"✗ Action space test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_training_demo():
    """Run a quick training demonstration."""
    print("\nRunning quick training demo...")

    try:
        # Create a simple training setup
        agent = DQNAgentAdapter(0, 'baseline')

        # Create random opponents
        from catan_rl.core.game.agents.random_agent import RandomAgent
        opponents = [RandomAgent(i + 1) for i in range(3)]

        all_agents = [agent] + opponents

        episodes = 5  # Just a few episodes for demo
        print(f"Running {episodes} training episodes...")

        for episode in range(episodes):
            # Set up game
            board = standard_board(seed=episode)
            game_state = initial_game_state(board, num_players=4)

            # Reset agents
            for a in all_agents:
                a.reset()

            turn_count = 0
            max_turns = 100  # Limit turns per episode

            while game_state.winner is None and turn_count < max_turns:
                current_player_id = game_state.current_player
                current_agent = all_agents[current_player_id]

                legal_actions = game_state.legal_actions()
                if not legal_actions:
                    break

                action = current_agent.select_action(game_state, legal_actions)
                prev_state = game_state
                game_state = apply_action(game_state, action)

                # Update DQN agent
                if isinstance(current_agent, DQNAgentAdapter):
                    reward = current_agent.compute_reward(prev_state, game_state)
                    current_agent.update(prev_state, action, reward, game_state)

                turn_count += 1

            winner = game_state.winner
            winner_type = "DQN" if winner == 0 else f"Random {winner}" if winner else "None"
            print(f"  Episode {episode + 1}: Winner = {winner_type}, Turns = {turn_count}")

        # Check final metrics
        final_metrics = agent.get_dqn_metrics()
        print(f"✓ Training demo completed")
        print(f"  Final epsilon: {final_metrics.get('epsilon', 0):.3f}")
        print(f"  Replay buffer size: {final_metrics.get('replay_buffer_size', 0)}")
        print(f"  Total steps: {final_metrics.get('step_count', 0)}")

        return True
    except Exception as e:
        print(f"✗ Training demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("DQN Integration Verification")
    print("=" * 30)

    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run tests
    tests = [
        test_dqn_agent_creation,
        test_state_encoders,
        test_action_space,
        test_game_integration,
        run_quick_training_demo
    ]

    results = []
    for test in tests:
        success = test()
        results.append(success)

    # Summary
    print("\n" + "=" * 30)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 30)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train_dqn_agents.py --config baseline --episodes 100")
        print("2. Evaluate agents: python scripts/evaluate_dqn_agents.py --compare baseline spatial rainbow")
        print("3. Run experiments: python examples/run_dqn_experiment.py --config compare_all")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())