#!/usr/bin/env python3
"""
Example script demonstrating the modular DQN framework for Catan RL.

This script shows how to:
1. Create different agent configurations
2. Run training experiments
3. Compare multiple approaches
4. Generate comprehensive evaluation reports

Usage:
    python examples/run_dqn_experiment.py --config baseline
    python examples/run_dqn_experiment.py --config spatial
    python examples/run_dqn_experiment.py --config rainbow
    python examples/run_dqn_experiment.py --config compare_all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.agents.dqn_agent import DQNAgent, DQNAgentFactory
from catan_rl.evaluation.evaluator import AgentEvaluator, ConfigurationComparator, run_comprehensive_evaluation
from catan_rl.experiments.experiment import ExperimentConfig, ExperimentRunner


def create_custom_agent(agent_id: int) -> DQNAgent:
    """Create custom agent configuration for experimentation."""
    config = {
        'state_encoder': {
            'type': 'feature',  # Options: 'feature', 'spatial', 'hybrid'
            'params': {}
        },
        'network': {
            'type': 'mlp',  # Options: 'mlp', 'conv', 'dueling_mlp', 'dueling_conv', 'hybrid', 'attention_fusion', 'ensemble'
            'params': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3,
                'activation': 'relu',
                'batch_norm': True
            }
        },
        'algorithm': {
            'type': 'double',  # Options: 'vanilla', 'double', 'dueling', 'rainbow'
            'params': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'target_update_frequency': 1000,
                'batch_size': 32,
                'optimizer': 'adam',
                'use_lr_scheduler': True,
                'lr_step_size': 5000,
                'lr_decay': 0.9
            }
        },
        'replay_buffer': {
            'type': 'prioritized',  # Options: 'uniform', 'prioritized', 'episodic', 'n_step'
            'capacity': 50000,
            'params': {
                'alpha': 0.6,
                'beta_start': 0.4,
                'beta_steps': 100000
            }
        },
        'epsilon_start': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.9995,
        'update_frequency': 4,
        'warmup_steps': 1000
    }
    return DQNAgent(agent_id, config)


def run_single_experiment(config_name: str, num_episodes: int = 1000):
    """Run a single training experiment."""
    print(f"Running {config_name} experiment for {num_episodes} episodes")

    # Create agent based on configuration
    if config_name == 'baseline':
        agent = DQNAgentFactory.create_baseline_agent(0)
    elif config_name == 'spatial':
        agent = DQNAgentFactory.create_spatial_agent(0)
    elif config_name == 'rainbow':
        agent = DQNAgentFactory.create_rainbow_agent(0)
    elif config_name == 'custom':
        agent = create_custom_agent(0)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Create opponents
    opponents = [DQNAgentFactory.create_baseline_agent(i + 1) for i in range(3)]

    # Set up experiment
    experiment_config = ExperimentConfig(
        name=f"{config_name}_experiment",
        agents=[agent] + opponents,
        num_episodes=num_episodes,
        num_players=4,
        save_frequency=max(1, num_episodes // 10),
        log_frequency=max(1, num_episodes // 50),
        output_dir=f"experiments/{config_name}"
    )

    # Run experiment
    experiment_runner = ExperimentRunner(experiment_config)
    results = experiment_runner.run()

    print(f"\nExperiment Results:")
    print(f"Total episodes: {len(results['episode_data'])}")

    # Analyze agent performance
    agent_wins = sum(1 for episode in results['episode_data']
                    if episode['player_id'] == 0 and episode['won'])
    agent_games = sum(1 for episode in results['episode_data']
                     if episode['player_id'] == 0)

    win_rate = agent_wins / agent_games if agent_games > 0 else 0.0
    print(f"Agent win rate: {win_rate:.2%} ({agent_wins}/{agent_games})")

    # Print final agent metrics
    final_metrics = agent.get_metrics()
    print(f"Final epsilon: {final_metrics['epsilon']:.3f}")
    print(f"Replay buffer size: {final_metrics['replay_buffer_size']}")

    return results


def run_evaluation_experiment(config_name: str):
    """Run comprehensive evaluation of a configuration."""
    print(f"Running comprehensive evaluation for {config_name}")

    # Create factory function
    if config_name == 'baseline':
        factory_func = DQNAgentFactory.create_baseline_agent
    elif config_name == 'spatial':
        factory_func = DQNAgentFactory.create_spatial_agent
    elif config_name == 'rainbow':
        factory_func = DQNAgentFactory.create_rainbow_agent
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Run evaluation
    evaluator = AgentEvaluator()
    result = evaluator.evaluate_agent(
        factory_func,
        config_name,
        num_episodes=500,
        num_evaluation_games=100,
        save_checkpoints=True,
        checkpoint_dir=f"evaluation/{config_name}"
    )

    print(f"\nEvaluation Results for {config_name}:")
    print(f"Win rate: {result.win_rate:.2%}")
    print(f"Average score: {result.average_score:.2f}")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Network parameters: {result.network_parameters:,}")
    if result.convergence_episode:
        print(f"Converged at episode: {result.convergence_episode}")

    return result


def run_comparison_experiment():
    """Compare multiple configurations."""
    print("Running comprehensive comparison of all configurations")

    configurations = {
        'baseline_mlp': DQNAgentFactory.create_baseline_agent,
        'spatial_cnn': DQNAgentFactory.create_spatial_agent,
        'rainbow_dqn': DQNAgentFactory.create_rainbow_agent,
        'custom_config': create_custom_agent
    }

    comparator = ConfigurationComparator()
    results = comparator.compare_configurations(
        configurations,
        num_episodes=300,  # Reduced for demonstration
        num_evaluation_games=50,
        num_runs_per_config=2
    )

    # Generate comprehensive report
    comparator.generate_report(results, "comparison_results")

    print(f"\nComparison Results:")
    print(f"Best configuration: {results.best_config}")
    print("\nPerformance Ranking:")
    for i, (config, win_rate) in enumerate(results.performance_ranking):
        print(f"  {i+1}. {config}: {win_rate:.2%}")

    return results


def demonstrate_modular_components():
    """Demonstrate the modularity of the framework."""
    print("Demonstrating modular component combinations:")

    # Test different state encoders
    state_encoders = ['feature', 'spatial']
    network_types = ['mlp', 'conv', 'dueling_mlp']
    algorithms = ['vanilla', 'double', 'rainbow']

    print("\nTesting different component combinations:")

    for encoder in state_encoders:
        for network in network_types:
            for algorithm in algorithms:
                # Skip invalid combinations
                if encoder == 'spatial' and network == 'mlp':
                    continue
                if encoder == 'feature' and network == 'conv':
                    continue

                try:
                    config = {
                        'state_encoder': {'type': encoder, 'params': {}},
                        'network': {'type': network, 'params': {'hidden_dims': [256, 128]}},
                        'algorithm': {'type': algorithm, 'params': {}},
                        'replay_buffer': {'type': 'uniform', 'capacity': 1000, 'params': {}},
                        'epsilon_start': 1.0, 'epsilon_min': 0.1, 'epsilon_decay': 0.995,
                        'update_frequency': 4, 'warmup_steps': 100
                    }

                    agent = DQNAgent(0, config)
                    info = agent.get_network_info()

                    print(f"  ✓ {encoder} + {network} + {algorithm}: "
                          f"{info['network']['total_parameters']:,} parameters")

                except Exception as e:
                    print(f"  ✗ {encoder} + {network} + {algorithm}: {str(e)}")

    print("\nAll valid combinations created successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run DQN experiments for Catan RL')
    parser.add_argument('--config', choices=['baseline', 'spatial', 'rainbow', 'custom',
                                            'evaluate', 'compare_all', 'demo_modular'],
                       default='baseline', help='Experiment configuration to run')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.config in ['baseline', 'spatial', 'rainbow', 'custom']:
        # Run single configuration experiment
        results = run_single_experiment(args.config, args.episodes)

    elif args.config == 'evaluate':
        # Run evaluation experiment
        config_to_evaluate = 'baseline'  # Could be made configurable
        results = run_evaluation_experiment(config_to_evaluate)

    elif args.config == 'compare_all':
        # Run comprehensive comparison
        results = run_comparison_experiment()

    elif args.config == 'demo_modular':
        # Demonstrate modular framework
        demonstrate_modular_components()
        return

    print(f"\nExperiment completed! Results saved to {args.output_dir}")
    print("\nTo analyze results, check the generated logs, checkpoints, and evaluation reports.")


if __name__ == '__main__':
    main()