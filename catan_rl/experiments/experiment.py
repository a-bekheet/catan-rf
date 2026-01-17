"""Experiment management and data collection system."""

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from catan_rl.experiments.agent import BaseAgent, AgentMetrics, GameMetrics
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.types import Action


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_id: str
    name: str
    description: str
    agent_configs: List[Dict[str, Any]]
    num_episodes: int
    max_steps_per_game: int
    evaluation_frequency: int
    save_frequency: int
    data_collection: Dict[str, bool]
    hyperparameters: Dict[str, Any]


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode_id: int
    game_metrics: GameMetrics
    training_metrics: Dict[str, Dict[str, float]]  # agent_id -> metrics
    evaluation_metrics: Optional[Dict[str, Any]] = None


class ExperimentTracker:
    """Tracks and stores experimental data."""

    def __init__(self, experiment_config: ExperimentConfig, base_path: str = "experiments"):
        self.config = experiment_config
        self.base_path = Path(base_path)
        self.experiment_path = self.base_path / "results" / experiment_config.experiment_id

        # Create directories
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        (self.experiment_path / "models").mkdir(exist_ok=True)
        (self.experiment_path / "logs").mkdir(exist_ok=True)
        (self.experiment_path / "analysis").mkdir(exist_ok=True)

        # Initialize tracking
        self.episode_results: List[EpisodeResult] = []
        self.start_time = time.time()

        # Save experiment config
        self._save_config()

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_path = self.experiment_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def record_episode(self, episode_result: EpisodeResult) -> None:
        """Record results from a completed episode."""
        self.episode_results.append(episode_result)

        # Save episode data if configured
        if self.config.data_collection.get("save_episodes", True):
            episode_path = self.experiment_path / "logs" / f"episode_{episode_result.episode_id:06d}.json"
            with open(episode_path, 'w') as f:
                json.dump(self._serialize_episode(episode_result), f, indent=2)

    def _serialize_episode(self, episode: EpisodeResult) -> Dict[str, Any]:
        """Convert episode result to JSON-serializable format."""
        return {
            "episode_id": episode.episode_id,
            "game_metrics": asdict(episode.game_metrics),
            "training_metrics": episode.training_metrics,
            "evaluation_metrics": episode.evaluation_metrics
        }

    def get_summary_statistics(self, last_n_episodes: int = 100) -> Dict[str, Any]:
        """Get summary statistics for recent episodes."""
        if not self.episode_results:
            return {}

        recent_episodes = self.episode_results[-last_n_episodes:]

        # Win rates by agent type
        win_counts = {}
        total_games = len(recent_episodes)

        for episode in recent_episodes:
            winner = episode.game_metrics.winner
            if winner is not None:
                agent_type = episode.game_metrics.agent_types[winner]
                win_counts[agent_type] = win_counts.get(agent_type, 0) + 1

        win_rates = {agent: count / total_games for agent, count in win_counts.items()}

        # Game length statistics
        game_lengths = [episode.game_metrics.game_length for episode in recent_episodes]

        # Score statistics
        all_scores = []
        for episode in recent_episodes:
            all_scores.extend(episode.game_metrics.final_scores)

        return {
            "episodes_analyzed": len(recent_episodes),
            "win_rates": win_rates,
            "game_length_stats": {
                "mean": np.mean(game_lengths),
                "std": np.std(game_lengths),
                "min": min(game_lengths),
                "max": max(game_lengths)
            },
            "score_stats": {
                "mean": np.mean(all_scores),
                "std": np.std(all_scores),
                "min": min(all_scores),
                "max": max(all_scores)
            },
            "runtime_hours": (time.time() - self.start_time) / 3600
        }

    def save_checkpoint(self, agents: List[BaseAgent]) -> None:
        """Save model checkpoints for all agents."""
        checkpoint_dir = self.experiment_path / "models" / f"episode_{len(self.episode_results):06d}"
        checkpoint_dir.mkdir(exist_ok=True)

        for agent in agents:
            model_path = checkpoint_dir / f"{agent.agent_id}.pkl"
            agent.save_model(str(model_path))

    def save_summary_report(self) -> None:
        """Save comprehensive summary report."""
        summary = {
            "experiment_config": asdict(self.config),
            "total_episodes": len(self.episode_results),
            "summary_statistics": self.get_summary_statistics(),
            "final_statistics": self.get_summary_statistics(len(self.episode_results)),
            "generated_at": datetime.now().isoformat()
        }

        report_path = self.experiment_path / "summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)


class ExperimentRunner:
    """Runs and manages RL experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tracker = ExperimentTracker(config)
        self.agents: List[BaseAgent] = []

    def setup_agents(self) -> None:
        """Initialize agents based on configuration."""
        from rl_framework.base.agent import AgentFactory

        self.agents = []
        for i, agent_config in enumerate(self.config.agent_configs):
            agent = AgentFactory.create_agent(
                agent_config["type"],
                player_id=i,
                **agent_config.get("params", {})
            )
            self.agents.append(agent)

    def run_episode(self, episode_id: int) -> EpisodeResult:
        """Run a single episode and collect data."""
        # Initialize game
        board = standard_board()
        state = initial_game_state(board, num_players=len(self.agents))

        # Reset agents
        for agent in self.agents:
            agent.reset()

        # Game tracking
        turn_metrics: List[AgentMetrics] = []
        step_count = 0
        setup_steps = 0

        game_start_time = time.time()

        while step_count < self.config.max_steps_per_game and state.winner is None:
            current_player = state.current_player
            legal_actions = state.legal_actions()

            if not legal_actions:
                break

            # Get action from agent
            agent = self.agents[current_player]
            start_time = time.time()

            try:
                action, decision_metrics = agent.select_action(state, legal_actions)
                decision_metrics.turn_id = step_count

                # Apply action
                next_state = state.apply(action)

                # Update agents with experience
                training_metrics = {}
                for i, train_agent in enumerate(self.agents):
                    if hasattr(train_agent, 'compute_reward'):
                        reward = train_agent.compute_reward(state, next_state) if hasattr(train_agent, 'compute_reward') else 0.0
                        agent_training_metrics = train_agent.update(state, action, reward, next_state)
                        training_metrics[train_agent.agent_id] = agent_training_metrics

                # Track setup vs main game
                if state.phase.value == "setup":
                    setup_steps += 1

                # Store metrics
                turn_metrics.append(decision_metrics)
                state = next_state
                step_count += 1

            except Exception as e:
                print(f"Error in step {step_count}: {e}")
                break

        # Create game metrics
        game_end_time = time.time()
        game_duration = int(game_end_time - game_start_time)

        game_metrics = GameMetrics(
            game_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            agent_types=[agent.agent_type for agent in self.agents],
            game_length=step_count,
            winner=state.winner,
            final_scores=[state.players[i].victory_points for i in range(len(self.agents))],
            setup_duration=setup_steps,
            main_game_duration=step_count - setup_steps,
            turn_metrics=turn_metrics,
            strategic_summary=self._compute_strategic_summary(state)
        )

        # Collect training metrics
        episode_training_metrics = {}
        for agent in self.agents:
            episode_training_metrics[agent.agent_id] = agent._get_internal_metrics()

        return EpisodeResult(
            episode_id=episode_id,
            game_metrics=game_metrics,
            training_metrics=episode_training_metrics
        )

    def _compute_strategic_summary(self, final_state) -> Dict[str, Any]:
        """Compute strategic analysis of the completed game."""
        summary = {
            "total_buildings_built": 0,
            "total_resources_collected": 0,
            "agent_strategies": {}
        }

        for i, agent in enumerate(self.agents):
            strategic_metrics = agent.compute_strategic_metrics(final_state)
            summary["agent_strategies"][agent.agent_id] = strategic_metrics
            summary["total_buildings_built"] += (
                strategic_metrics["settlements"] +
                strategic_metrics["cities"] +
                strategic_metrics["roads"]
            )

        return summary

    def run_experiment(self) -> None:
        """Run the complete experiment."""
        print(f"🚀 Starting experiment: {self.config.name}")
        print(f"   Episodes: {self.config.num_episodes}")
        print(f"   Agents: {[agent.agent_type for agent in self.agents]}")

        self.setup_agents()

        for episode in range(self.config.num_episodes):
            episode_result = self.run_episode(episode)
            self.tracker.record_episode(episode_result)

            # Progress reporting
            if (episode + 1) % 10 == 0:
                stats = self.tracker.get_summary_statistics(last_n_episodes=10)
                print(f"Episode {episode + 1}: Win rates = {stats.get('win_rates', {})}")

            # Model checkpointing
            if (episode + 1) % self.config.save_frequency == 0:
                self.tracker.save_checkpoint(self.agents)

        # Final summary
        self.tracker.save_summary_report()
        print(f"✅ Experiment complete! Results saved to: {self.tracker.experiment_path}")