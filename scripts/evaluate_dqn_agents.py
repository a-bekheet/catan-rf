#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained DQN agents.

This script provides detailed analysis and comparison of trained models,
including statistical testing and performance breakdowns.

Usage:
    python scripts/evaluate_dqn_agents.py --model baseline --games 100
    python scripts/evaluate_dqn_agents.py --compare baseline spatial rainbow
    python scripts/evaluate_dqn_agents.py --tournament --rounds 5
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.core.integration import DQNAgentAdapter
from catan_rl.evaluation import ConfigurationComparator, AgentEvaluator


class DQNEvaluator:
    """Comprehensive evaluation system for trained DQN agents."""

    def __init__(self, output_dir: str = 'evaluation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        self.tournament_results = {}

    def load_trained_agent(self, model_path: str, agent_type: str, player_id: int = 0) -> DQNAgentAdapter:
        """Load a trained DQN agent from checkpoint."""
        agent = DQNAgentAdapter(
            player_id=player_id,
            dqn_config_name=agent_type,
            checkpoint_path=model_path
        )
        return agent

    def evaluate_single_agent(
        self,
        agent: DQNAgentAdapter,
        num_games: int = 100,
        opponent_types: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single agent against various opponents."""
        if opponent_types is None:
            opponent_types = ['random', 'baseline', 'spatial']

        print(f"Evaluating agent {agent.player_id} against {len(opponent_types)} opponent types...")

        results = {
            'agent_info': self._get_agent_info(agent),
            'evaluations': {}
        }

        for opp_type in opponent_types:
            print(f"  vs {opp_type} opponents...")
            eval_result = self._run_evaluation_games(agent, opp_type, num_games)
            results['evaluations'][opp_type] = eval_result

        # Compute overall statistics
        results['overall'] = self._compute_overall_stats(results['evaluations'])

        return results

    def compare_multiple_agents(
        self,
        agent_configs: List[Tuple[str, str]],  # [(model_path, agent_type), ...]
        num_games: int = 100
    ) -> Dict[str, Any]:
        """Compare multiple trained agents."""
        print(f"Comparing {len(agent_configs)} agents with {num_games} games each...")

        agents = []
        for i, (model_path, agent_type) in enumerate(agent_configs):
            agent = self.load_trained_agent(model_path, agent_type, i)
            agents.append((agent, agent_type))

        results = {
            'agent_types': [agent_type for _, agent_type in agents],
            'head_to_head': {},
            'round_robin': {},
            'statistical_analysis': {}
        }

        # Head-to-head comparisons
        results['head_to_head'] = self._run_head_to_head(agents, num_games)

        # Round-robin tournament
        results['round_robin'] = self._run_round_robin(agents, num_games)

        # Statistical analysis
        results['statistical_analysis'] = self._compute_statistical_significance(
            results['round_robin']['match_results']
        )

        return results

    def run_tournament(
        self,
        agent_configs: List[Tuple[str, str]],
        rounds: int = 5,
        games_per_round: int = 50
    ) -> Dict[str, Any]:
        """Run comprehensive tournament evaluation."""
        print(f"Running {rounds}-round tournament with {len(agent_configs)} agents...")

        agents = []
        for i, (model_path, agent_type) in enumerate(agent_configs):
            agent = self.load_trained_agent(model_path, agent_type, i)
            agents.append((agent, agent_type))

        tournament_results = {
            'rounds': [],
            'cumulative_scores': {agent_type: [] for _, agent_type in agents},
            'final_rankings': [],
            'consistency_analysis': {}
        }

        cumulative_wins = {agent_type: 0 for _, agent_type in agents}

        for round_num in range(rounds):
            print(f"\nRound {round_num + 1}/{rounds}")
            round_result = self._run_tournament_round(agents, games_per_round)

            tournament_results['rounds'].append(round_result)

            # Update cumulative scores
            for agent_type in cumulative_wins:
                cumulative_wins[agent_type] += round_result['wins'].get(agent_type, 0)
                tournament_results['cumulative_scores'][agent_type].append(
                    cumulative_wins[agent_type]
                )

        # Final rankings
        final_scores = [(agent_type, wins) for agent_type, wins in cumulative_wins.items()]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        tournament_results['final_rankings'] = final_scores

        # Consistency analysis
        tournament_results['consistency_analysis'] = self._analyze_consistency(
            tournament_results['rounds']
        )

        return tournament_results

    def _get_agent_info(self, agent: DQNAgentAdapter) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        metrics = agent.get_dqn_metrics()
        network_info = agent.dqn_agent.get_network_info()

        return {
            'agent_id': agent.player_id,
            'network_info': network_info,
            'training_metrics': metrics,
            'parameter_count': network_info['network']['total_parameters']
        }

    def _run_evaluation_games(
        self,
        agent: DQNAgentAdapter,
        opponent_type: str,
        num_games: int
    ) -> Dict[str, Any]:
        """Run evaluation games against specific opponent type."""
        from catan_rl.core.game.engine.board import standard_board
        from catan_rl.core.game.engine.game_state import initial_game_state
        from catan_rl.core.game.engine.rules import apply_action

        # Set agent to evaluation mode
        agent.set_evaluation_mode(True)

        wins = 0
        game_lengths = []
        scores = []
        decision_times = []

        for game in range(num_games):
            if game % 20 == 0:
                print(f"    Game {game}/{num_games}")

            # Create opponents
            opponents = []
            for i in range(3):
                if opponent_type == 'random':
                    from catan_rl.core.game.agents.random_agent import RandomAgent
                    opponents.append(RandomAgent(i + 1))
                else:
                    opp_agent = DQNAgentAdapter(
                        player_id=i + 1,
                        dqn_config_name=opponent_type
                    )
                    opp_agent.set_evaluation_mode(True)
                    opponents.append(opp_agent)

            all_agents = [agent] + opponents

            # Run game
            board = standard_board(seed=game)
            game_state = initial_game_state(board, num_players=4)

            for a in all_agents:
                a.reset()

            episode_length = 0
            max_turns = 1000

            game_start_time = time.time()
            agent_decision_times = []

            while game_state.winner is None and episode_length < max_turns:
                current_player_id = game_state.current_player
                current_agent = all_agents[current_player_id]

                legal_actions = game_state.legal_actions()
                if not legal_actions:
                    break

                # Measure decision time for main agent
                if current_player_id == 0:
                    decision_start = time.time()

                action = current_agent.select_action(game_state, legal_actions)

                if current_player_id == 0:
                    agent_decision_times.append(time.time() - decision_start)

                prev_state = game_state
                game_state = apply_action(game_state, action)

                # Update DQN agents
                if hasattr(current_agent, 'update') and hasattr(current_agent, 'compute_reward'):
                    reward = current_agent.compute_reward(prev_state, game_state)
                    current_agent.update(prev_state, action, reward, game_state)

                episode_length += 1

            # Record results
            if game_state.winner == 0:  # Main agent won
                wins += 1

            game_lengths.append(episode_length)
            if game_state.players:
                scores.append(game_state.players[0].victory_points)
            if agent_decision_times:
                decision_times.append(np.mean(agent_decision_times))

        return {
            'num_games': num_games,
            'wins': wins,
            'win_rate': wins / num_games,
            'avg_game_length': np.mean(game_lengths),
            'avg_score': np.mean(scores),
            'avg_decision_time': np.mean(decision_times) if decision_times else 0,
            'score_std': np.std(scores),
            'game_length_std': np.std(game_lengths)
        }

    def _run_head_to_head(self, agents, num_games) -> Dict[str, Any]:
        """Run head-to-head comparisons between all agent pairs."""
        results = {}

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, type1 = agents[i]
                agent2, type2 = agents[j]

                print(f"  {type1} vs {type2}...")

                matchup_result = self._run_agent_matchup(agent1, agent2, num_games)
                results[f"{type1}_vs_{type2}"] = matchup_result

        return results

    def _run_agent_matchup(self, agent1, agent2, num_games) -> Dict[str, Any]:
        """Run games between two specific agents."""
        from catan_rl.core.game.engine.board import standard_board
        from catan_rl.core.game.engine.game_state import initial_game_state
        from catan_rl.core.game.engine.rules import apply_action
        from catan_rl.core.game.agents.random_agent import RandomAgent

        agent1.set_evaluation_mode(True)
        agent2.set_evaluation_mode(True)

        wins = {0: 0, 1: 0, 'other': 0}  # agent1, agent2, others
        game_lengths = []

        for game in range(num_games):
            # Create agents (alternate positions)
            if game % 2 == 0:
                all_agents = [
                    agent1, agent2,
                    RandomAgent(2), RandomAgent(3)
                ]
            else:
                all_agents = [
                    agent2, agent1,
                    RandomAgent(2), RandomAgent(3)
                ]

            # Run game
            board = standard_board(seed=game)
            game_state = initial_game_state(board, num_players=4)

            for a in all_agents:
                a.reset()

            episode_length = 0
            max_turns = 1000

            while game_state.winner is None and episode_length < max_turns:
                current_player_id = game_state.current_player
                current_agent = all_agents[current_player_id]

                legal_actions = game_state.legal_actions()
                if not legal_actions:
                    break

                action = current_agent.select_action(game_state, legal_actions)
                prev_state = game_state
                game_state = apply_action(game_state, action)

                if hasattr(current_agent, 'update') and hasattr(current_agent, 'compute_reward'):
                    reward = current_agent.compute_reward(prev_state, game_state)
                    current_agent.update(prev_state, action, reward, game_state)

                episode_length += 1

            # Record winner
            winner = game_state.winner
            if winner == 0:
                wins[0 if game % 2 == 0 else 1] += 1
            elif winner == 1:
                wins[1 if game % 2 == 0 else 0] += 1
            else:
                wins['other'] += 1

            game_lengths.append(episode_length)

        return {
            'num_games': num_games,
            'wins': wins,
            'win_rates': {k: v / num_games for k, v in wins.items()},
            'avg_game_length': np.mean(game_lengths)
        }

    def _run_round_robin(self, agents, num_games) -> Dict[str, Any]:
        """Run round-robin tournament."""
        results = {
            'total_wins': {agent_type: 0 for _, agent_type in agents},
            'match_results': [],
            'win_matrix': np.zeros((len(agents), len(agents)))
        }

        # Run games with different agent configurations
        for game in range(num_games):
            # Shuffle agent positions
            import random
            agent_positions = list(range(len(agents)))
            random.shuffle(agent_positions)

            # Select 4 agents (if more than 4, rotate through them)
            selected_agents = [agents[agent_positions[i % len(agents)]] for i in range(4)]

            # Run game
            winner, game_data = self._run_single_tournament_game(selected_agents, game)

            if winner is not None and winner < len(selected_agents):
                winning_agent, winning_type = selected_agents[winner]
                results['total_wins'][winning_type] += 1

                # Update win matrix
                winner_idx = next(i for i, (_, t) in enumerate(agents) if t == winning_type)
                results['win_matrix'][winner_idx, winner_idx] += 1

            results['match_results'].append({
                'game': game,
                'agents': [agent_type for _, agent_type in selected_agents],
                'winner': winner,
                'winner_type': selected_agents[winner][1] if winner is not None and winner < len(selected_agents) else None,
                'game_data': game_data
            })

        return results

    def _run_single_tournament_game(self, agents, game_seed) -> Tuple[int, Dict[str, Any]]:
        """Run a single tournament game."""
        from catan_rl.core.game.engine.board import standard_board
        from catan_rl.core.game.engine.game_state import initial_game_state
        from catan_rl.core.game.engine.rules import apply_action

        board = standard_board(seed=game_seed)
        game_state = initial_game_state(board, num_players=len(agents))

        # Set agents to evaluation mode and reset
        for agent, _ in agents:
            if hasattr(agent, 'set_evaluation_mode'):
                agent.set_evaluation_mode(True)
            agent.reset()

        episode_length = 0
        max_turns = 1000

        while game_state.winner is None and episode_length < max_turns:
            current_player_id = game_state.current_player
            current_agent, _ = agents[current_player_id]

            legal_actions = game_state.legal_actions()
            if not legal_actions:
                break

            action = current_agent.select_action(game_state, legal_actions)
            prev_state = game_state
            game_state = apply_action(game_state, action)

            if hasattr(current_agent, 'update'):
                reward = current_agent.compute_reward(prev_state, game_state)
                current_agent.update(prev_state, action, reward, game_state)

            episode_length += 1

        game_data = {
            'episode_length': episode_length,
            'final_scores': {i: game_state.players[i].victory_points for i in game_state.players}
        }

        return game_state.winner, game_data

    def _run_tournament_round(self, agents, games_per_round) -> Dict[str, Any]:
        """Run a single tournament round."""
        round_results = self._run_round_robin(agents, games_per_round)

        return {
            'wins': round_results['total_wins'],
            'win_rates': {
                agent_type: wins / games_per_round
                for agent_type, wins in round_results['total_wins'].items()
            },
            'match_results': round_results['match_results']
        }

    def _compute_overall_stats(self, evaluations) -> Dict[str, Any]:
        """Compute overall statistics across all evaluations."""
        total_games = sum(eval_data['num_games'] for eval_data in evaluations.values())
        total_wins = sum(eval_data['wins'] for eval_data in evaluations.values())
        overall_win_rate = total_wins / total_games if total_games > 0 else 0

        avg_scores = [eval_data['avg_score'] for eval_data in evaluations.values()]
        avg_game_lengths = [eval_data['avg_game_length'] for eval_data in evaluations.values()]

        return {
            'total_games': total_games,
            'total_wins': total_wins,
            'overall_win_rate': overall_win_rate,
            'avg_score': np.mean(avg_scores),
            'avg_game_length': np.mean(avg_game_lengths),
            'performance_by_opponent': {
                opp_type: eval_data['win_rate']
                for opp_type, eval_data in evaluations.items()
            }
        }

    def _compute_statistical_significance(self, match_results) -> Dict[str, Any]:
        """Compute statistical significance of results."""
        # Extract win data for each agent type
        agent_types = set()
        for result in match_results:
            agent_types.update(result['agents'])

        agent_types = list(agent_types)
        win_data = {agent_type: [] for agent_type in agent_types}

        for result in match_results:
            winner_type = result.get('winner_type')
            for agent_type in agent_types:
                if agent_type in result['agents']:
                    win_data[agent_type].append(1 if winner_type == agent_type else 0)

        # Compute pairwise t-tests
        p_values = {}
        for i in range(len(agent_types)):
            for j in range(i + 1, len(agent_types)):
                type1, type2 = agent_types[i], agent_types[j]
                if len(win_data[type1]) > 0 and len(win_data[type2]) > 0:
                    t_stat, p_value = stats.ttest_ind(win_data[type1], win_data[type2])
                    p_values[f"{type1}_vs_{type2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

        return {
            'win_data': {k: np.mean(v) for k, v in win_data.items()},
            'pairwise_tests': p_values
        }

    def _analyze_consistency(self, rounds) -> Dict[str, Any]:
        """Analyze consistency of agent performance across rounds."""
        agent_types = set()
        for round_result in rounds:
            agent_types.update(round_result['wins'].keys())

        consistency_metrics = {}
        for agent_type in agent_types:
            win_rates = [round_result['win_rates'].get(agent_type, 0) for round_result in rounds]
            consistency_metrics[agent_type] = {
                'mean_win_rate': np.mean(win_rates),
                'std_win_rate': np.std(win_rates),
                'coefficient_variation': np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) > 0 else float('inf'),
                'min_win_rate': np.min(win_rates),
                'max_win_rate': np.max(win_rates)
            }

        return consistency_metrics

    def generate_report(self, results: Dict[str, Any], report_type: str):
        """Generate comprehensive evaluation report."""
        report_dir = self.output_dir / f"{report_type}_report"
        report_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(report_dir / "results.json", 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)

        # Generate plots
        self._generate_evaluation_plots(results, report_dir, report_type)

        # Generate text report
        self._generate_text_report(results, report_dir, report_type)

        print(f"Report generated: {report_dir}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj

    def _generate_evaluation_plots(self, results, report_dir, report_type):
        """Generate evaluation plots."""
        if report_type == 'comparison':
            self._plot_comparison_results(results, report_dir)
        elif report_type == 'tournament':
            self._plot_tournament_results(results, report_dir)
        else:
            self._plot_single_agent_results(results, report_dir)

    def _plot_comparison_results(self, results, report_dir):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Win rates by agent type
        agent_types = results['agent_types']
        round_robin = results['round_robin']

        win_rates = [round_robin['total_wins'].get(agent_type, 0) /
                    len(round_robin['match_results']) for agent_type in agent_types]

        axes[0, 0].bar(agent_types, win_rates)
        axes[0, 0].set_title('Overall Win Rates')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Head-to-head matrix
        if 'head_to_head' in results:
            h2h = results['head_to_head']
            matrix_data = np.zeros((len(agent_types), len(agent_types)))

            for matchup, data in h2h.items():
                if '_vs_' in matchup:
                    type1, type2 = matchup.split('_vs_')
                    if type1 in agent_types and type2 in agent_types:
                        i, j = agent_types.index(type1), agent_types.index(type2)
                        matrix_data[i, j] = data['win_rates'].get(0, 0)
                        matrix_data[j, i] = data['win_rates'].get(1, 0)

            sns.heatmap(matrix_data, annot=True, xticklabels=agent_types,
                       yticklabels=agent_types, ax=axes[0, 1])
            axes[0, 1].set_title('Head-to-Head Win Rates')

        plt.tight_layout()
        plt.savefig(report_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tournament_results(self, results, report_dir):
        """Plot tournament results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cumulative scores over rounds
        for agent_type, scores in results['cumulative_scores'].items():
            axes[0, 0].plot(scores, label=agent_type, marker='o')
        axes[0, 0].set_title('Cumulative Wins Over Tournament Rounds')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Cumulative Wins')
        axes[0, 0].legend()

        # Final rankings
        rankings = results['final_rankings']
        agents, scores = zip(*rankings)
        axes[0, 1].bar(agents, scores)
        axes[0, 1].set_title('Final Tournament Rankings')
        axes[0, 1].set_ylabel('Total Wins')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Consistency analysis
        consistency = results['consistency_analysis']
        agent_types = list(consistency.keys())
        cv_values = [consistency[agent]['coefficient_variation'] for agent in agent_types]

        axes[1, 0].bar(agent_types, cv_values)
        axes[1, 0].set_title('Performance Consistency (Lower = More Consistent)')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(report_dir / 'tournament_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_single_agent_results(self, results, report_dir):
        """Plot single agent evaluation results."""
        evaluations = results['evaluations']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Win rates by opponent type
        opponent_types = list(evaluations.keys())
        win_rates = [evaluations[opp]['win_rate'] for opp in opponent_types]

        axes[0, 0].bar(opponent_types, win_rates)
        axes[0, 0].set_title('Win Rates vs Different Opponents')
        axes[0, 0].set_ylabel('Win Rate')

        # Average scores vs different opponents
        avg_scores = [evaluations[opp]['avg_score'] for opp in opponent_types]
        axes[0, 1].bar(opponent_types, avg_scores)
        axes[0, 1].set_title('Average Scores vs Different Opponents')
        axes[0, 1].set_ylabel('Average Score')

        # Game length distribution
        game_lengths = [evaluations[opp]['avg_game_length'] for opp in opponent_types]
        axes[1, 0].bar(opponent_types, game_lengths)
        axes[1, 0].set_title('Average Game Length vs Different Opponents')
        axes[1, 0].set_ylabel('Game Length')

        # Decision time
        decision_times = [evaluations[opp]['avg_decision_time'] for opp in opponent_types]
        axes[1, 1].bar(opponent_types, decision_times)
        axes[1, 1].set_title('Average Decision Time vs Different Opponents')
        axes[1, 1].set_ylabel('Decision Time (seconds)')

        plt.tight_layout()
        plt.savefig(report_dir / 'single_agent_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_text_report(self, results, report_dir, report_type):
        """Generate text summary report."""
        with open(report_dir / 'summary.txt', 'w') as f:
            f.write(f"DQN AGENT EVALUATION REPORT\n")
            f.write(f"Report Type: {report_type.upper()}\n")
            f.write("=" * 50 + "\n\n")

            if report_type == 'comparison':
                self._write_comparison_summary(results, f)
            elif report_type == 'tournament':
                self._write_tournament_summary(results, f)
            else:
                self._write_single_agent_summary(results, f)

    def _write_comparison_summary(self, results, f):
        """Write comparison summary."""
        f.write("AGENT COMPARISON RESULTS\n")
        f.write("-" * 25 + "\n\n")

        agent_types = results['agent_types']
        round_robin = results['round_robin']

        f.write("Overall Performance:\n")
        for agent_type in agent_types:
            wins = round_robin['total_wins'].get(agent_type, 0)
            total_games = len(round_robin['match_results'])
            win_rate = wins / total_games if total_games > 0 else 0
            f.write(f"  {agent_type}: {wins}/{total_games} wins ({win_rate:.3f} win rate)\n")

        if 'statistical_analysis' in results:
            f.write("\nStatistical Significance:\n")
            for comparison, stats in results['statistical_analysis']['pairwise_tests'].items():
                significance = "significant" if stats['significant'] else "not significant"
                f.write(f"  {comparison}: {significance} (p={stats['p_value']:.4f})\n")

    def _write_tournament_summary(self, results, f):
        """Write tournament summary."""
        f.write("TOURNAMENT RESULTS\n")
        f.write("-" * 17 + "\n\n")

        f.write("Final Rankings:\n")
        for i, (agent_type, total_wins) in enumerate(results['final_rankings']):
            f.write(f"  {i+1}. {agent_type}: {total_wins} wins\n")

        f.write("\nConsistency Analysis:\n")
        consistency = results['consistency_analysis']
        for agent_type, metrics in consistency.items():
            f.write(f"  {agent_type}:\n")
            f.write(f"    Mean win rate: {metrics['mean_win_rate']:.3f}\n")
            f.write(f"    Std deviation: {metrics['std_win_rate']:.3f}\n")
            f.write(f"    Consistency: {1/metrics['coefficient_variation']:.2f}\n")

    def _write_single_agent_summary(self, results, f):
        """Write single agent summary."""
        f.write("SINGLE AGENT EVALUATION\n")
        f.write("-" * 23 + "\n\n")

        overall = results['overall']
        f.write(f"Overall Performance:\n")
        f.write(f"  Total games: {overall['total_games']}\n")
        f.write(f"  Total wins: {overall['total_wins']}\n")
        f.write(f"  Win rate: {overall['overall_win_rate']:.3f}\n")
        f.write(f"  Average score: {overall['avg_score']:.2f}\n")

        f.write(f"\nPerformance by Opponent Type:\n")
        for opp_type, win_rate in overall['performance_by_opponent'].items():
            f.write(f"  vs {opp_type}: {win_rate:.3f}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agents')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--agent-type', choices=['baseline', 'spatial', 'rainbow'],
                       default='baseline', help='Type of agent to evaluate')
    parser.add_argument('--compare', nargs='+', help='Compare multiple agent types')
    parser.add_argument('--tournament', action='store_true', help='Run tournament evaluation')
    parser.add_argument('--games', type=int, default=100, help='Number of evaluation games')
    parser.add_argument('--rounds', type=int, default=5, help='Number of tournament rounds')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    evaluator = DQNEvaluator(args.output_dir)

    if args.compare:
        # Compare multiple agents
        agent_configs = [(f"checkpoints/{agent_type}/final", agent_type)
                        for agent_type in args.compare]
        results = evaluator.compare_multiple_agents(agent_configs, args.games)
        evaluator.generate_report(results, 'comparison')

        print("\nComparison Results:")
        round_robin = results['round_robin']
        for agent_type in args.compare:
            wins = round_robin['total_wins'].get(agent_type, 0)
            win_rate = wins / len(round_robin['match_results'])
            print(f"  {agent_type}: {win_rate:.3f} win rate")

    elif args.tournament:
        # Run tournament
        agent_types = ['baseline', 'spatial', 'rainbow']
        agent_configs = [(f"checkpoints/{agent_type}/final", agent_type)
                        for agent_type in agent_types]
        results = evaluator.run_tournament(agent_configs, args.rounds, args.games)
        evaluator.generate_report(results, 'tournament')

        print("\nTournament Results:")
        for i, (agent_type, wins) in enumerate(results['final_rankings']):
            print(f"  {i+1}. {agent_type}: {wins} total wins")

    else:
        # Single agent evaluation
        if not args.model:
            print("Error: --model required for single agent evaluation")
            return

        agent = evaluator.load_trained_agent(args.model, args.agent_type)
        results = evaluator.evaluate_single_agent(agent, args.games)
        evaluator.generate_report(results, 'single_agent')

        print(f"\nEvaluation Results for {args.agent_type}:")
        overall = results['overall']
        print(f"  Overall win rate: {overall['overall_win_rate']:.3f}")
        print(f"  Average score: {overall['avg_score']:.2f}")


if __name__ == '__main__':
    main()