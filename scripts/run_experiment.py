#!/usr/bin/env python3
"""Clean experiment runner using the reorganized framework."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.experiments.experiment import ExperimentConfig, ExperimentRunner
from catan_rl.agents.random_agent import RandomAgent
from catan_rl.agents.legacy_qtable_agent import LegacyQTableAgent


def main():
    """Run a clean experiment comparing different agents."""
    print("🧪 CATAN RL EXPERIMENT FRAMEWORK")
    print("=" * 50)

    # Create experiment configuration
    config = ExperimentConfig(
        experiment_id="clean_framework_test_001",
        name="Clean Framework: Agent Comparison",
        description="Testing the cleaned framework with multiple agent types",
        agent_configs=[
            {"type": "RandomAgent", "params": {"seed": 42}},
            {"type": "LegacyQTableAgent", "params": {"epsilon": 0.2}},
            {"type": "RandomAgent", "params": {"seed": 123}},
            {"type": "RandomAgent", "params": {"seed": 456}}
        ],
        num_episodes=10,
        max_steps_per_game=300,  # Increased for full games
        evaluation_frequency=5,
        save_frequency=5,
        data_collection={
            "save_episodes": True,
            "track_decisions": True,
            "record_states": True
        },
        hyperparameters={
            "learning_enabled": True,
            "exploration_decay": True
        }
    )

    # Run experiment
    runner = ExperimentRunner(config)

    print(f"📋 Experiment: {config.name}")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   Max steps per game: {config.max_steps_per_game}")
    print(f"   Agents: {[ac['type'] for ac in config.agent_configs]}")

    try:
        runner.run_experiment()

        # Display comprehensive results
        stats = runner.tracker.get_summary_statistics()
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Win rates: {stats.get('win_rates', {})}")
        print(f"   Average game length: {stats.get('game_length_stats', {}).get('mean', 'N/A'):.1f} steps")
        print(f"   Average score: {stats.get('score_stats', {}).get('mean', 'N/A'):.1f}")
        print(f"   Runtime: {stats.get('runtime_hours', 0):.3f} hours")
        print(f"   Results saved to: {runner.tracker.experiment_path}")

        # Check if any games were won
        if stats.get('win_rates'):
            print("✅ SUCCESS: Agents are winning games!")
        else:
            print("⚠️ No wins recorded - games may be timing out")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()