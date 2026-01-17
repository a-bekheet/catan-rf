#!/usr/bin/env python3
"""
Comprehensive training script for DQN agents in Catan.

This script provides multiple training configurations and comprehensive
logging and analysis capabilities.

Usage:
    python scripts/train_dqn_agents.py --config baseline --episodes 1000
    python scripts/train_dqn_agents.py --config comparative --episodes 500
    python scripts/train_dqn_agents.py --config analysis --episodes 2000
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.core.integration import DQNAgentAdapter, ExperimentRunner
from catan_rl.agents.dqn_agent import DQNAgentFactory
from catan_rl.evaluation import AgentEvaluator, ConfigurationComparator


class TrainingConfig:
    """Training configuration for different experiment types."""

    @staticmethod
    def baseline_config() -> Dict[str, Any]:
        """Simple baseline training configuration."""
        return {
            'name': 'baseline_training',
            'agents': [
                {'type': 'baseline', 'player_id': 0},
                {'type': 'random', 'player_id': 1},
                {'type': 'random', 'player_id': 2},
                {'type': 'random', 'player_id': 3}
            ],
            'num_episodes': 1000,
            'save_frequency': 100,
            'log_frequency': 50,
            'evaluation_frequency': 200,
            'evaluation_games': 50,
            'output_dir': 'experiments/baseline'
        }

    @staticmethod
    def comparative_config() -> Dict[str, Any]:
        """Configuration for comparing different agent types."""
        return {
            'name': 'comparative_training',
            'agents': [
                {'type': 'baseline', 'player_id': 0},
                {'type': 'spatial', 'player_id': 1},
                {'type': 'rainbow', 'player_id': 2},
                {'type': 'random', 'player_id': 3}
            ],
            'num_episodes': 1000,
            'save_frequency': 100,
            'log_frequency': 25,
            'evaluation_frequency': 200,
            'evaluation_games': 50,
            'output_dir': 'experiments/comparative'
        }

    @staticmethod
    def analysis_config() -> Dict[str, Any]:
        """Configuration for detailed analysis and research."""
        return {
            'name': 'analysis_training',
            'agents': [
                {'type': 'rainbow', 'player_id': 0},
                {'type': 'rainbow', 'player_id': 1},
                {'type': 'baseline', 'player_id': 2},
                {'type': 'baseline', 'player_id': 3}
            ],
            'num_episodes': 2000,
            'save_frequency': 50,
            'log_frequency': 20,
            'evaluation_frequency': 100,
            'evaluation_games': 100,
            'output_dir': 'experiments/analysis',
            'detailed_logging': True,
            'save_game_replays': True
        }


class DQNTrainer:
    """Comprehensive DQN training system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.training_results = {}
        self.evaluation_results = []
        self.training_curves = {
            'episodes': [],
            'win_rates': {},
            'avg_rewards': {},
            'q_values': {},
            'losses': {},
            'epsilon_values': {}
        }

        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def create_agents(self) -> List[DQNAgentAdapter]:
        """Create agents based on configuration."""
        agents = []

        for agent_config in self.config['agents']:
            if agent_config['type'] == 'random':
                # Create random agent (placeholder - would need proper implementation)
                from catan_rl.core.game.agents.random_agent import RandomAgent
                agents.append(RandomAgent(agent_config['player_id']))
            else:
                # Create DQN agent
                agent = DQNAgentAdapter(
                    player_id=agent_config['player_id'],
                    dqn_config_name=agent_config['type']
                )
                agents.append(agent)

        return agents

    def run_training(self) -> Dict[str, Any]:
        """Run complete training process."""
        print(f"Starting training: {self.config['name']}")
        print(f"Output directory: {self.output_dir}")

        # Create agents
        agents = self.create_agents()
        dqn_agents = [agent for agent in agents if isinstance(agent, DQNAgentAdapter)]

        print(f"Created {len(agents)} agents ({len(dqn_agents)} DQN agents)")

        # Run training
        start_time = time.time()
        training_results = self._run_training_loop(dqn_agents)
        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Final evaluation
        print("Running final evaluation...")
        final_eval_results = self._run_final_evaluation(dqn_agents)

        # Compile results
        results = {
            'config': self.config,
            'training_time': training_time,
            'training_results': training_results,
            'evaluation_results': final_eval_results,
            'training_curves': self.training_curves
        }

        # Save results
        self._save_results(results)
        self._generate_plots()

        return results

    def _run_training_loop(self, agents: List[DQNAgentAdapter]) -> Dict[str, Any]:
        """Run the main training loop."""
        from catan_rl.core.game.engine.board import standard_board
        from catan_rl.core.game.engine.game_state import initial_game_state
        from catan_rl.core.game.engine.rules import apply_action

        num_episodes = self.config['num_episodes']
        save_frequency = self.config.get('save_frequency', 100)
        log_frequency = self.config.get('log_frequency', 50)
        eval_frequency = self.config.get('evaluation_frequency', 200)

        # Initialize tracking
        wins_per_agent = {agent.player_id: 0 for agent in agents}
        episode_data = []
        recent_episode_lengths = []

        print(f"Training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Reset environment
            board = standard_board(seed=episode)
            game_state = initial_game_state(board, num_players=len(agents))

            # Reset agents
            for agent in agents:
                agent.reset()

            # Run episode
            episode_result = self._run_episode(game_state, agents)
            episode_data.append(episode_result)
            recent_episode_lengths.append(episode_result['episode_length'])

            # Track wins
            if episode_result['winner'] is not None:
                wins_per_agent[episode_result['winner']] += 1

            # Update training curves
            self._update_training_curves(episode, episode_result, agents, wins_per_agent)

            # Logging
            if episode % log_frequency == 0:
                self._log_progress(episode, wins_per_agent, recent_episode_lengths[-log_frequency:])

            # Save checkpoints
            if save_frequency > 0 and episode % save_frequency == 0:
                self._save_checkpoints(agents, episode)

            # Run evaluation
            if eval_frequency > 0 and episode % eval_frequency == 0 and episode > 0:
                eval_result = self._run_evaluation(agents, episode)
                self.evaluation_results.append(eval_result)

        return {
            'episodes': episode_data,
            'wins_per_agent': wins_per_agent,
            'final_win_rates': {aid: wins_per_agent[aid] / num_episodes for aid in wins_per_agent}
        }

    def _run_episode(self, game_state, agents) -> Dict[str, Any]:
        """Run a single episode and return results."""
        from catan_rl.core.game.engine.rules import apply_action

        episode_length = 0
        max_turns = 1000  # Reduced from 1000 to encourage faster games
        episode_rewards = {agent.player_id: 0.0 for agent in agents if isinstance(agent, DQNAgentAdapter)}

        while game_state.winner is None and episode_length < max_turns:
            current_player_id = game_state.current_player
            current_agent = agents[current_player_id]

            # Get legal actions
            legal_actions = game_state.legal_actions()
            if not legal_actions:
                break

            # Agent selects action
            action = current_agent.select_action(game_state, legal_actions)

            # Apply action
            prev_state = game_state
            game_state = apply_action(game_state, action)

            # Check if truncated (hit max_turns without winner)
            truncated = (episode_length >= max_turns - 1 and game_state.winner is None)

            # Update DQN agents
            if isinstance(current_agent, DQNAgentAdapter):
                reward = current_agent.compute_reward(prev_state, game_state, truncated=truncated)
                current_agent.update(prev_state, action, reward, game_state)
                episode_rewards[current_player_id] += reward

            episode_length += 1

        return {
            'winner': game_state.winner,
            'episode_length': episode_length,
            'episode_rewards': episode_rewards,
            'final_scores': {pid: player.victory_points for pid, player in game_state.players.items()}
        }

    def _update_training_curves(self, episode, episode_result, agents, wins_per_agent):
        """Update training curves with current episode data."""
        self.training_curves['episodes'].append(episode)

        # Calculate rolling win rates (last 50 episodes)
        window = min(50, episode + 1)
        for agent in agents:
            if isinstance(agent, DQNAgentAdapter):
                agent_id = agent.player_id
                if agent_id not in self.training_curves['win_rates']:
                    self.training_curves['win_rates'][agent_id] = []

                recent_wins = wins_per_agent[agent_id] - sum(
                    1 for i in range(max(0, episode - window), episode)
                    if i < len(self.training_curves['episodes']) and
                    episode_result.get('winner') == agent_id
                )
                win_rate = wins_per_agent[agent_id] / (episode + 1)
                self.training_curves['win_rates'][agent_id].append(win_rate)

                # Get agent metrics
                metrics = agent.get_dqn_metrics()

                # Track other metrics
                if agent_id not in self.training_curves['avg_rewards']:
                    self.training_curves['avg_rewards'][agent_id] = []
                    self.training_curves['q_values'][agent_id] = []
                    self.training_curves['losses'][agent_id] = []
                    self.training_curves['epsilon_values'][agent_id] = []

                self.training_curves['avg_rewards'][agent_id].append(
                    episode_result['episode_rewards'].get(agent_id, 0.0)
                )
                self.training_curves['q_values'][agent_id].append(
                    metrics.get('avg_q_value', 0.0)
                )
                self.training_curves['losses'][agent_id].append(
                    metrics.get('avg_loss', 0.0)
                )
                self.training_curves['epsilon_values'][agent_id].append(
                    metrics.get('epsilon', 0.0)
                )

    def _log_progress(self, episode, wins_per_agent, recent_lengths):
        """Log training progress."""
        print(f"\nEpisode {episode}:")
        for agent_id, wins in wins_per_agent.items():
            win_rate = wins / (episode + 1) if episode > 0 else 0
            print(f"  Agent {agent_id}: {wins}/{episode + 1} wins ({win_rate:.3f} win rate)")

        if recent_lengths:
            avg_length = np.mean(recent_lengths)
            print(f"  Average episode length (last {len(recent_lengths)}): {avg_length:.1f}")

    def _save_checkpoints(self, agents, episode):
        """Save agent checkpoints."""
        checkpoint_dir = self.output_dir / 'checkpoints' / f'episode_{episode}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for agent in agents:
            if isinstance(agent, DQNAgentAdapter):
                checkpoint_path = checkpoint_dir / f'agent_{agent.player_id}'
                agent.save_model()

    def _run_evaluation(self, agents, episode) -> Dict[str, Any]:
        """Run evaluation games to assess current performance."""
        print(f"  Running evaluation at episode {episode}...")

        # Set agents to evaluation mode
        for agent in agents:
            if isinstance(agent, DQNAgentAdapter):
                agent.set_evaluation_mode(True)

        eval_games = self.config.get('evaluation_games', 50)
        eval_wins = {agent.player_id: 0 for agent in agents}

        # Run evaluation games
        for game in range(eval_games):
            from catan_rl.core.game.engine.board import standard_board
            from catan_rl.core.game.engine.game_state import initial_game_state

            # Use different seed for evaluation
            # Handle string episode names (e.g., 'final') by using hash
            episode_num = episode if isinstance(episode, int) else hash(str(episode)) % 10000
            board = standard_board(seed=10000 + episode_num * 1000 + game)
            game_state = initial_game_state(board, num_players=len(agents))

            # Reset agents
            for agent in agents:
                agent.reset()

            # Run game
            result = self._run_episode(game_state, agents)
            if result['winner'] is not None:
                eval_wins[result['winner']] += 1

        # Restore training mode
        for agent in agents:
            if isinstance(agent, DQNAgentAdapter):
                agent.set_evaluation_mode(False)

        eval_result = {
            'episode': episode,
            'eval_games': eval_games,
            'eval_wins': eval_wins,
            'eval_win_rates': {aid: wins / eval_games for aid, wins in eval_wins.items()}
        }

        print(f"  Evaluation results: {eval_result['eval_win_rates']}")
        return eval_result

    def _run_final_evaluation(self, agents) -> Dict[str, Any]:
        """Run comprehensive final evaluation."""
        print("Running final evaluation...")

        final_eval_games = 200
        return self._run_evaluation(agents, 'final')

    def _save_results(self, results):
        """Save comprehensive results."""
        # Save main results
        with open(self.output_dir / 'results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {self.output_dir}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj

    def _generate_plots(self):
        """Generate training analysis plots."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Win rate plots
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        for agent_id, win_rates in self.training_curves['win_rates'].items():
            plt.plot(win_rates, label=f'Agent {agent_id}')
        plt.title('Win Rates Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.legend()

        plt.subplot(2, 2, 2)
        for agent_id, rewards in self.training_curves['avg_rewards'].items():
            # Smooth with moving average
            smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
            plt.plot(smoothed, label=f'Agent {agent_id}')
        plt.title('Average Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()

        plt.subplot(2, 2, 3)
        for agent_id, q_values in self.training_curves['q_values'].items():
            if q_values:  # Only plot if we have data
                smoothed = np.convolve(q_values, np.ones(20)/20, mode='valid')
                plt.plot(smoothed, label=f'Agent {agent_id}')
        plt.title('Q-Values')
        plt.xlabel('Episode')
        plt.ylabel('Average Q-Value')
        plt.legend()

        plt.subplot(2, 2, 4)
        for agent_id, epsilon_values in self.training_curves['epsilon_values'].items():
            plt.plot(epsilon_values, label=f'Agent {agent_id}')
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()

        plt.tight_layout()
        plt.savefig(plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to {plots_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train DQN agents for Catan')
    parser.add_argument('--config', choices=['baseline', 'comparative', 'analysis'],
                       default='baseline', help='Training configuration')
    parser.add_argument('--episodes', type=int, help='Number of training episodes (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto',
                       help='Device to use for training')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Get configuration
    if args.config == 'baseline':
        config = TrainingConfig.baseline_config()
    elif args.config == 'comparative':
        config = TrainingConfig.comparative_config()
    elif args.config == 'analysis':
        config = TrainingConfig.analysis_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Override configuration with command line arguments
    if args.episodes:
        config['num_episodes'] = args.episodes
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Run training
    trainer = DQNTrainer(config)
    results = trainer.run_training()

    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)

    final_win_rates = results['training_results']['final_win_rates']
    for agent_id, win_rate in final_win_rates.items():
        print(f"Agent {agent_id}: {win_rate:.3f} win rate")

    print(f"\nTotal training time: {results['training_time']:.2f} seconds")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()