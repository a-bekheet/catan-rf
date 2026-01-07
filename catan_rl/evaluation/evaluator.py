"""Comprehensive evaluation framework for Catan RL agents."""

from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

from ..agents.dqn_agent import DQNAgent, DQNAgentFactory
from ..experiments.experiment import ExperimentConfig, ExperimentRunner
from catan_rl.core.game.engine.game_state import GameState


@dataclass
class EvaluationResult:
    """Results from evaluating an agent configuration."""
    agent_config_name: str
    total_games: int
    wins: int
    win_rate: float
    average_score: float
    average_game_length: float
    average_decision_time: float
    convergence_episode: Optional[int]
    final_epsilon: float
    network_parameters: int
    training_time: float
    additional_metrics: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Results from comparing multiple agent configurations."""
    configurations: List[str]
    results: List[EvaluationResult]
    best_config: str
    performance_ranking: List[Tuple[str, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    convergence_analysis: Dict[str, Any]


class AgentEvaluator:
    """Evaluates individual agent configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate_agent(
        self,
        agent_factory_func: callable,
        agent_config_name: str,
        num_episodes: int = 1000,
        num_evaluation_games: int = 100,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "checkpoints"
    ) -> EvaluationResult:
        """
        Evaluate a single agent configuration.

        Args:
            agent_factory_func: Function that creates the agent
            agent_config_name: Name identifier for this configuration
            num_episodes: Number of training episodes
            num_evaluation_games: Number of games for final evaluation
            save_checkpoints: Whether to save training checkpoints
            checkpoint_dir: Directory to save checkpoints

        Returns:
            EvaluationResult with comprehensive metrics
        """
        print(f"Evaluating agent configuration: {agent_config_name}")

        # Create agent
        agent = agent_factory_func(0)

        # Training phase
        print(f"Training for {num_episodes} episodes...")
        training_start_time = time.time()
        training_metrics = self._train_agent(agent, num_episodes, save_checkpoints, checkpoint_dir, agent_config_name)
        training_time = time.time() - training_start_time

        # Evaluation phase
        print(f"Evaluating with {num_evaluation_games} games...")
        agent.set_evaluation_mode(True)
        evaluation_metrics = self._evaluate_agent(agent, num_evaluation_games)

        # Analyze convergence
        convergence_episode = self._find_convergence_point(training_metrics['win_rates'])

        # Compile results
        result = EvaluationResult(
            agent_config_name=agent_config_name,
            total_games=num_episodes + num_evaluation_games,
            wins=evaluation_metrics['wins'],
            win_rate=evaluation_metrics['win_rate'],
            average_score=evaluation_metrics['average_score'],
            average_game_length=evaluation_metrics['average_game_length'],
            average_decision_time=evaluation_metrics['average_decision_time'],
            convergence_episode=convergence_episode,
            final_epsilon=agent.epsilon,
            network_parameters=sum(p.numel() for p in agent.q_network.parameters()),
            training_time=training_time,
            additional_metrics={
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'network_info': agent.get_network_info()
            }
        )

        print(f"Evaluation complete. Win rate: {result.win_rate:.2%}")
        return result

    def _train_agent(
        self,
        agent: DQNAgent,
        num_episodes: int,
        save_checkpoints: bool,
        checkpoint_dir: str,
        config_name: str
    ) -> Dict[str, Any]:
        """Train agent and collect metrics."""
        experiment_config = ExperimentConfig(
            name=f"training_{config_name}",
            agents=[agent],
            num_episodes=num_episodes,
            num_players=4,
            save_frequency=max(1, num_episodes // 10) if save_checkpoints else 0,
            log_frequency=max(1, num_episodes // 100),
            output_dir=checkpoint_dir
        )

        experiment_runner = ExperimentRunner(experiment_config)
        results = experiment_runner.run()

        # Extract training metrics
        metrics = {
            'win_rates': [],
            'average_scores': [],
            'game_lengths': [],
            'decision_times': [],
            'losses': [],
            'q_values': [],
            'epsilons': []
        }

        # Process results to extract metrics over time
        for episode_data in results['episode_data']:
            if episode_data['player_id'] == agent.agent_id:
                metrics['win_rates'].append(1.0 if episode_data['won'] else 0.0)
                metrics['average_scores'].append(episode_data['final_score'])
                metrics['game_lengths'].append(episode_data['game_length'])

        # Get agent-specific metrics
        agent_metrics = agent.get_metrics()
        metrics['losses'] = agent_metrics.get('avg_loss', 0.0)
        metrics['q_values'] = agent_metrics.get('avg_q_value', 0.0)
        metrics['epsilons'] = agent_metrics.get('epsilon', 0.0)

        return metrics

    def _evaluate_agent(self, agent: DQNAgent, num_games: int) -> Dict[str, Any]:
        """Evaluate trained agent performance."""
        # Run evaluation games
        wins = 0
        scores = []
        game_lengths = []
        decision_times = []

        for game_idx in range(num_games):
            if game_idx % 20 == 0:
                print(f"Evaluation game {game_idx}/{num_games}")

            # Create experiment for single game
            experiment_config = ExperimentConfig(
                name=f"eval_game_{game_idx}",
                agents=[agent] + [self._create_baseline_opponent(i + 1) for i in range(3)],
                num_episodes=1,
                num_players=4,
                save_frequency=0,
                log_frequency=0
            )

            experiment_runner = ExperimentRunner(experiment_config)
            game_results = experiment_runner.run()

            # Extract results for evaluated agent
            for episode_data in game_results['episode_data']:
                if episode_data['player_id'] == agent.agent_id:
                    if episode_data['won']:
                        wins += 1
                    scores.append(episode_data['final_score'])
                    game_lengths.append(episode_data['game_length'])
                    decision_times.append(episode_data.get('avg_decision_time', 0.0))

        return {
            'wins': wins,
            'win_rate': wins / num_games,
            'average_score': np.mean(scores),
            'average_game_length': np.mean(game_lengths),
            'average_decision_time': np.mean(decision_times),
            'score_std': np.std(scores),
            'scores': scores
        }

    def _create_baseline_opponent(self, agent_id: int) -> DQNAgent:
        """Create baseline opponent for evaluation."""
        return DQNAgentFactory.create_baseline_agent(agent_id)

    def _find_convergence_point(self, win_rates: List[float], window_size: int = 50) -> Optional[int]:
        """Find episode where performance converged."""
        if len(win_rates) < window_size * 2:
            return None

        # Calculate moving averages
        moving_avg = []
        for i in range(window_size, len(win_rates)):
            avg = np.mean(win_rates[i-window_size:i])
            moving_avg.append(avg)

        # Find where variance becomes small
        variances = []
        for i in range(window_size, len(moving_avg)):
            var = np.var(moving_avg[i-window_size:i])
            variances.append(var)

        # Convergence when variance drops below threshold
        threshold = 0.01
        for i, var in enumerate(variances):
            if var < threshold:
                return i + window_size * 2

        return None


class ConfigurationComparator:
    """Compares multiple agent configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.evaluator = AgentEvaluator(config)

    def compare_configurations(
        self,
        configurations: Dict[str, callable],
        num_episodes: int = 1000,
        num_evaluation_games: int = 100,
        num_runs_per_config: int = 3,
        parallel: bool = False
    ) -> ComparisonResult:
        """
        Compare multiple agent configurations.

        Args:
            configurations: Dict mapping config names to factory functions
            num_episodes: Training episodes per configuration
            num_evaluation_games: Evaluation games per configuration
            num_runs_per_config: Number of independent runs per config
            parallel: Whether to run evaluations in parallel

        Returns:
            ComparisonResult with detailed comparison analysis
        """
        print(f"Comparing {len(configurations)} configurations with {num_runs_per_config} runs each")

        all_results = []

        for config_name, factory_func in configurations.items():
            config_results = []

            for run in range(num_runs_per_config):
                print(f"Configuration: {config_name}, Run: {run + 1}/{num_runs_per_config}")

                result = self.evaluator.evaluate_agent(
                    factory_func,
                    f"{config_name}_run_{run}",
                    num_episodes,
                    num_evaluation_games
                )

                config_results.append(result)

            # Average results across runs
            averaged_result = self._average_results(config_results, config_name)
            all_results.append(averaged_result)

        # Analyze results
        best_config = self._find_best_configuration(all_results)
        performance_ranking = self._rank_configurations(all_results)
        statistical_significance = self._compute_statistical_significance(all_results)
        convergence_analysis = self._analyze_convergence(all_results)

        return ComparisonResult(
            configurations=list(configurations.keys()),
            results=all_results,
            best_config=best_config,
            performance_ranking=performance_ranking,
            statistical_significance=statistical_significance,
            convergence_analysis=convergence_analysis
        )

    def _average_results(self, results: List[EvaluationResult], config_name: str) -> EvaluationResult:
        """Average results across multiple runs."""
        if not results:
            raise ValueError("No results to average")

        # Average numerical metrics
        averaged = EvaluationResult(
            agent_config_name=config_name,
            total_games=results[0].total_games,
            wins=int(np.mean([r.wins for r in results])),
            win_rate=np.mean([r.win_rate for r in results]),
            average_score=np.mean([r.average_score for r in results]),
            average_game_length=np.mean([r.average_game_length for r in results]),
            average_decision_time=np.mean([r.average_decision_time for r in results]),
            convergence_episode=int(np.mean([r.convergence_episode for r in results if r.convergence_episode])) if any(r.convergence_episode for r in results) else None,
            final_epsilon=np.mean([r.final_epsilon for r in results]),
            network_parameters=results[0].network_parameters,
            training_time=np.mean([r.training_time for r in results]),
            additional_metrics={
                'std_win_rate': np.std([r.win_rate for r in results]),
                'std_score': np.std([r.average_score for r in results]),
                'individual_results': results
            }
        )

        return averaged

    def _find_best_configuration(self, results: List[EvaluationResult]) -> str:
        """Find best configuration based on win rate."""
        best_result = max(results, key=lambda r: r.win_rate)
        return best_result.agent_config_name

    def _rank_configurations(self, results: List[EvaluationResult]) -> List[Tuple[str, float]]:
        """Rank configurations by performance."""
        ranking = sorted(results, key=lambda r: r.win_rate, reverse=True)
        return [(r.agent_config_name, r.win_rate) for r in ranking]

    def _compute_statistical_significance(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
        """Compute statistical significance between configurations."""
        from scipy import stats

        significance = {}
        for i, result_i in enumerate(results):
            significance[result_i.agent_config_name] = {}
            for j, result_j in enumerate(results):
                if i != j:
                    # Get individual run results
                    scores_i = [r.win_rate for r in result_i.additional_metrics['individual_results']]
                    scores_j = [r.win_rate for r in result_j.additional_metrics['individual_results']]

                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scores_i, scores_j)
                    significance[result_i.agent_config_name][result_j.agent_config_name] = p_value

        return significance

    def _analyze_convergence(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze convergence properties."""
        convergence_episodes = [r.convergence_episode for r in results if r.convergence_episode]

        return {
            'converged_configs': len(convergence_episodes),
            'average_convergence_episode': np.mean(convergence_episodes) if convergence_episodes else None,
            'fastest_convergence': min(convergence_episodes) if convergence_episodes else None,
            'slowest_convergence': max(convergence_episodes) if convergence_episodes else None
        }

    def generate_report(self, comparison_result: ComparisonResult, output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save raw results
        results_data = [asdict(result) for result in comparison_result.results]
        with open(output_path / "results.json", 'w') as f:
            json.dump({
                'comparison_result': asdict(comparison_result),
                'detailed_results': results_data
            }, f, indent=2)

        # Generate visualizations
        self._create_performance_plots(comparison_result, output_path)
        self._create_convergence_plots(comparison_result, output_path)

        # Generate summary report
        self._create_summary_report(comparison_result, output_path)

        print(f"Evaluation report saved to {output_path}")

    def _create_performance_plots(self, comparison_result: ComparisonResult, output_path: Path):
        """Create performance visualization plots."""
        # Win rate comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Win rates
        config_names = [r.agent_config_name for r in comparison_result.results]
        win_rates = [r.win_rate for r in comparison_result.results]
        std_rates = [r.additional_metrics.get('std_win_rate', 0) for r in comparison_result.results]

        axes[0, 0].bar(config_names, win_rates, yerr=std_rates, capsize=5)
        axes[0, 0].set_title('Win Rates by Configuration')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Average scores
        avg_scores = [r.average_score for r in comparison_result.results]
        axes[0, 1].bar(config_names, avg_scores)
        axes[0, 1].set_title('Average Scores by Configuration')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Training time vs performance
        training_times = [r.training_time for r in comparison_result.results]
        axes[1, 0].scatter(training_times, win_rates)
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_title('Training Time vs Performance')

        # Model complexity vs performance
        param_counts = [r.network_parameters for r in comparison_result.results]
        axes[1, 1].scatter(param_counts, win_rates)
        axes[1, 1].set_xlabel('Network Parameters')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_title('Model Complexity vs Performance')

        plt.tight_layout()
        plt.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_convergence_plots(self, comparison_result: ComparisonResult, output_path: Path):
        """Create convergence analysis plots."""
        fig, ax = plt.subplots(figsize=(10, 6))

        convergence_episodes = []
        config_names = []

        for result in comparison_result.results:
            if result.convergence_episode:
                convergence_episodes.append(result.convergence_episode)
                config_names.append(result.agent_config_name)

        if convergence_episodes:
            ax.bar(config_names, convergence_episodes)
            ax.set_title('Convergence Speed by Configuration')
            ax.set_ylabel('Episodes to Convergence')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_path / "convergence_analysis.png", dpi=300, bbox_inches='tight')

        plt.close()

    def _create_summary_report(self, comparison_result: ComparisonResult, output_path: Path):
        """Create text summary report."""
        with open(output_path / "summary_report.txt", 'w') as f:
            f.write("CATAN RL AGENT EVALUATION REPORT\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Configurations Evaluated: {len(comparison_result.configurations)}\n")
            f.write(f"Best Configuration: {comparison_result.best_config}\n\n")

            f.write("PERFORMANCE RANKING:\n")
            f.write("-" * 20 + "\n")
            for i, (config, win_rate) in enumerate(comparison_result.performance_ranking):
                f.write(f"{i+1}. {config}: {win_rate:.2%} win rate\n")

            f.write("\n\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in comparison_result.results:
                f.write(f"\nConfiguration: {result.agent_config_name}\n")
                f.write(f"  Win Rate: {result.win_rate:.2%}\n")
                f.write(f"  Average Score: {result.average_score:.2f}\n")
                f.write(f"  Training Time: {result.training_time:.2f}s\n")
                f.write(f"  Network Parameters: {result.network_parameters:,}\n")
                if result.convergence_episode:
                    f.write(f"  Convergence Episode: {result.convergence_episode}\n")

            f.write("\n\nCONVERGENCE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            conv_analysis = comparison_result.convergence_analysis
            f.write(f"Configs that converged: {conv_analysis['converged_configs']}\n")
            if conv_analysis['average_convergence_episode']:
                f.write(f"Average convergence episode: {conv_analysis['average_convergence_episode']:.0f}\n")
                f.write(f"Fastest convergence: {conv_analysis['fastest_convergence']}\n")
                f.write(f"Slowest convergence: {conv_analysis['slowest_convergence']}\n")


def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all agent configurations."""
    # Define configurations to compare
    configurations = {
        'baseline_mlp': DQNAgentFactory.create_baseline_agent,
        'spatial_cnn': DQNAgentFactory.create_spatial_agent,
        'rainbow_dqn': DQNAgentFactory.create_rainbow_agent
    }

    # Run comparison
    comparator = ConfigurationComparator()
    results = comparator.compare_configurations(
        configurations,
        num_episodes=500,  # Reduced for testing
        num_evaluation_games=50,
        num_runs_per_config=2
    )

    # Generate report
    comparator.generate_report(results)

    print("\nEvaluation Summary:")
    print(f"Best configuration: {results.best_config}")
    for config, win_rate in results.performance_ranking:
        print(f"  {config}: {win_rate:.2%}")

    return results