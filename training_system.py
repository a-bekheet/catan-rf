#!/usr/bin/env python3
"""
Enhanced Training System for Catan RL
=====================================

Comprehensive training infrastructure with visualization, model management,
and advanced training loops.
"""

import os
import sys
import time
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import queue

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from catan_rl.agents.dqn_agent import DQNAgentFactory
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action


@dataclass
class TrainingConfig:
    """Configuration for training sessions."""
    episodes: int = 1000
    save_frequency: int = 100
    eval_frequency: int = 50
    max_turns_per_episode: int = 200
    learning_start: int = 100
    target_update_frequency: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"
    visualize: bool = True
    agent_configs: Dict[str, Any] = None


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode: int
    winner: Optional[int]
    turns: int
    duration: float
    rewards: Dict[int, float]
    final_scores: Dict[int, int]
    invalid_actions: Dict[int, int]


@dataclass
class TrainingStats:
    """Overall training statistics."""
    total_episodes: int = 0
    total_time: float = 0.0
    win_rates: Dict[int, List[float]] = None
    avg_episode_length: List[float] = None
    avg_rewards: Dict[int, List[float]] = None
    learning_curves: Dict[int, List[float]] = None
    invalid_action_rates: Dict[int, List[float]] = None

    def __post_init__(self):
        if self.win_rates is None:
            self.win_rates = {}
        if self.avg_episode_length is None:
            self.avg_episode_length = []
        if self.avg_rewards is None:
            self.avg_rewards = {}
        if self.learning_curves is None:
            self.learning_curves = {}
        if self.invalid_action_rates is None:
            self.invalid_action_rates = {}


class ProgressBar:
    """ASCII progress bar for training visualization."""

    def __init__(self, total: int, width: int = 50, desc: str = "Progress"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, amount: int = 1):
        self.current += amount
        self._display()

    def _display(self):
        if self.total == 0:
            return

        progress = self.current / self.total
        filled_width = int(self.width * progress)
        bar = '█' * filled_width + '-' * (self.width - filled_width)

        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed_time * (self.total - self.current) / self.current
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "∞"

        print(f'\r{self.desc}: |{bar}| {self.current}/{self.total} '
              f'({progress:.1%}) ETA: {eta_str}', end='', flush=True)

        if self.current >= self.total:
            print()  # New line when complete


class TrainingVisualizer:
    """Real-time training visualization."""

    def __init__(self, num_agents: int = 4):
        self.num_agents = num_agents
        self.stats_history = []
        self.live_plotting = MATPLOTLIB_AVAILABLE

        if self.live_plotting:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Catan RL Training Progress')
            plt.ion()

    def update(self, stats: TrainingStats, episode: int):
        """Update visualization with new statistics."""
        if not self.live_plotting:
            return

        self.axes[0, 0].clear()
        self.axes[0, 1].clear()
        self.axes[1, 0].clear()
        self.axes[1, 1].clear()

        # Win rates
        for agent_id in range(self.num_agents):
            if agent_id in stats.win_rates:
                episodes = list(range(len(stats.win_rates[agent_id])))
                self.axes[0, 0].plot(episodes, stats.win_rates[agent_id],
                                   label=f'Agent {agent_id}')
        self.axes[0, 0].set_title('Win Rates Over Time')
        self.axes[0, 0].set_xlabel('Evaluation Period')
        self.axes[0, 0].set_ylabel('Win Rate')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)

        # Average episode length
        if stats.avg_episode_length:
            episodes = list(range(len(stats.avg_episode_length)))
            self.axes[0, 1].plot(episodes, stats.avg_episode_length)
            self.axes[0, 1].set_title('Average Episode Length')
            self.axes[0, 1].set_xlabel('Evaluation Period')
            self.axes[0, 1].set_ylabel('Turns')
            self.axes[0, 1].grid(True)

        # Average rewards
        for agent_id in range(self.num_agents):
            if agent_id in stats.avg_rewards:
                episodes = list(range(len(stats.avg_rewards[agent_id])))
                self.axes[1, 0].plot(episodes, stats.avg_rewards[agent_id],
                                   label=f'Agent {agent_id}')
        self.axes[1, 0].set_title('Average Rewards')
        self.axes[1, 0].set_xlabel('Evaluation Period')
        self.axes[1, 0].set_ylabel('Reward')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)

        # Invalid action rates
        for agent_id in range(self.num_agents):
            if agent_id in stats.invalid_action_rates:
                episodes = list(range(len(stats.invalid_action_rates[agent_id])))
                self.axes[1, 1].plot(episodes, stats.invalid_action_rates[agent_id],
                                   label=f'Agent {agent_id}')
        self.axes[1, 1].set_title('Invalid Action Rates')
        self.axes[1, 1].set_xlabel('Evaluation Period')
        self.axes[1, 1].set_ylabel('Invalid Rate')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)

        plt.tight_layout()
        plt.pause(0.01)

    def save(self, filepath: str):
        """Save current plot to file."""
        if self.live_plotting:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')


class CatanTrainer:
    """Main training system for Catan RL agents."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.agents = []
        self.stats = TrainingStats()
        self.visualizer = TrainingVisualizer()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize agents
        self._initialize_agents()

        # Training state
        self.training_active = False
        self.pause_requested = False

    def _initialize_agents(self):
        """Initialize training agents."""
        print("🤖 Initializing DQN agents...")

        agent_types = ['baseline', 'spatial', 'rainbow']
        for i in range(4):  # 4 players
            agent_type = agent_types[i % len(agent_types)]

            if agent_type == 'baseline':
                agent = DQNAgentFactory.create_baseline_agent(i)
            elif agent_type == 'spatial':
                agent = DQNAgentFactory.create_spatial_agent(i)
            else:  # rainbow
                agent = DQNAgentFactory.create_rainbow_agent(i)

            self.agents.append(agent)
            print(f"  ✓ Agent {i}: {agent_type} DQN")

    def run_episode(self, episode_num: int, max_turns: int = 200) -> EpisodeStats:
        """Run a single training episode."""
        start_time = time.time()

        # Initialize game
        board = standard_board(seed=42 + episode_num)
        game_state = initial_game_state(board, num_players=len(self.agents))

        # Reset agents
        for agent in self.agents:
            agent.reset()

        # Episode tracking
        episode_rewards = {i: 0.0 for i in range(len(self.agents))}
        invalid_actions = {i: 0 for i in range(len(self.agents))}
        turn = 0

        while not game_state.winner and turn < max_turns:
            if self.pause_requested:
                return None

            current_player = game_state.current_player
            agent = self.agents[current_player]

            # Get legal actions
            legal_actions_list = legal_actions(game_state)
            if not legal_actions_list:
                break

            # Agent selects action
            try:
                action, metrics = agent.select_action(game_state, legal_actions_list)

                # Apply action
                new_state = apply_action(game_state, action)

                # Calculate reward
                reward = self._calculate_reward(game_state, new_state, action, current_player)
                episode_rewards[current_player] += reward

                # Update agent
                is_terminal = (new_state.winner is not None) or (turn >= max_turns - 1)
                agent.observe_reward(reward, new_state, is_terminal)

                game_state = new_state

            except ValueError:
                # Invalid action
                invalid_actions[current_player] += 1
                agent.observe_reward(-1.0, game_state, False)

                # Force a valid action (fallback to pass if possible)
                try:
                    from catan_rl.core.game.engine.types import Action, ActionType
                    pass_action = Action(ActionType.PASS_TURN, {})
                    if pass_action in legal_actions_list:
                        game_state = apply_action(game_state, pass_action)
                    else:
                        break
                except:
                    break

            turn += 1

        # End episode for all agents
        for i, agent in enumerate(self.agents):
            final_reward = 10.0 if game_state.winner == i else 0.0
            agent.observe_reward(final_reward, game_state, True)
            episode_rewards[i] += final_reward

        duration = time.time() - start_time
        final_scores = {i: player.victory_points for i, player in game_state.players.items()}

        return EpisodeStats(
            episode=episode_num,
            winner=game_state.winner,
            turns=turn,
            duration=duration,
            rewards=episode_rewards,
            final_scores=final_scores,
            invalid_actions=invalid_actions
        )

    def _calculate_reward(self, old_state, new_state, action, player_id):
        """Calculate reward for an action."""
        old_player = old_state.players[player_id]
        new_player = new_state.players[player_id]

        reward = 0.0

        # Victory points reward
        vp_diff = new_player.victory_points - old_player.victory_points
        reward += vp_diff * 5.0

        # Building rewards
        settlement_diff = len(new_player.settlements) - len(old_player.settlements)
        city_diff = len(new_player.cities) - len(old_player.cities)
        road_diff = len(new_player.roads) - len(old_player.roads)

        reward += settlement_diff * 1.0
        reward += city_diff * 2.0
        reward += road_diff * 0.5

        # Resource management reward
        old_resources = sum(old_player.resources.values())
        new_resources = sum(new_player.resources.values())
        if old_resources > 7 and new_resources <= 7:
            reward += 1.0  # Reward for avoiding discard

        # Small positive reward for valid actions
        reward += 0.1

        return reward

    def evaluate_agents(self, num_eval_episodes: int = 20) -> Dict[str, Any]:
        """Evaluate current agent performance."""
        print(f"\n📊 Evaluating agents ({num_eval_episodes} episodes)...")

        eval_stats = {
            'wins': {i: 0 for i in range(len(self.agents))},
            'total_rewards': {i: 0.0 for i in range(len(self.agents))},
            'invalid_actions': {i: 0 for i in range(len(self.agents))},
            'episode_lengths': [],
            'total_episodes': 0
        }

        # Set agents to evaluation mode
        for agent in self.agents:
            agent.set_evaluation_mode(True)

        eval_bar = ProgressBar(num_eval_episodes, desc="Evaluating")

        for episode in range(num_eval_episodes):
            stats = self.run_episode(episode, max_turns=100)
            if stats is None:  # Paused
                break

            eval_stats['total_episodes'] += 1
            eval_stats['episode_lengths'].append(stats.turns)

            if stats.winner is not None:
                eval_stats['wins'][stats.winner] += 1

            for agent_id in range(len(self.agents)):
                eval_stats['total_rewards'][agent_id] += stats.rewards[agent_id]
                eval_stats['invalid_actions'][agent_id] += stats.invalid_actions[agent_id]

            eval_bar.update()

        # Return to training mode
        for agent in self.agents:
            agent.set_evaluation_mode(False)

        return eval_stats

    def train(self):
        """Main training loop."""
        print(f"\n🎯 Starting training for {self.config.episodes} episodes...")
        print(f"📁 Checkpoints will be saved to: {self.checkpoint_dir}")

        if MATPLOTLIB_AVAILABLE:
            print("📈 Live visualization enabled")
        else:
            print("⚠️  Install matplotlib for live visualization")

        self.training_active = True
        start_time = time.time()

        progress_bar = ProgressBar(self.config.episodes, desc="Training")

        # Training loop
        for episode in range(self.config.episodes):
            if self.pause_requested:
                print("\n⏸️  Training paused by user")
                break

            # Run episode
            episode_stats = self.run_episode(episode)
            if episode_stats is None:  # Paused
                break

            progress_bar.update()

            # Evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                eval_results = self.evaluate_agents()
                self._update_training_stats(eval_results, episode)
                self.visualizer.update(self.stats, episode)

            # Save checkpoint
            if episode % self.config.save_frequency == 0 and episode > 0:
                self.save_checkpoint(episode)

            # Print progress
            if episode % 50 == 0 and episode > 0:
                self._print_progress(episode, episode_stats)

        total_time = time.time() - start_time
        self.stats.total_episodes = episode + 1
        self.stats.total_time = total_time

        print(f"\n✅ Training completed in {total_time:.1f}s")
        self._print_final_results()

        self.training_active = False

    def _update_training_stats(self, eval_results: Dict[str, Any], episode: int):
        """Update training statistics with evaluation results."""
        if eval_results['total_episodes'] == 0:
            return

        # Win rates
        for agent_id in range(len(self.agents)):
            if agent_id not in self.stats.win_rates:
                self.stats.win_rates[agent_id] = []

            win_rate = eval_results['wins'][agent_id] / eval_results['total_episodes']
            self.stats.win_rates[agent_id].append(win_rate)

        # Average episode length
        if eval_results['episode_lengths']:
            avg_length = np.mean(eval_results['episode_lengths'])
            self.stats.avg_episode_length.append(avg_length)

        # Average rewards
        for agent_id in range(len(self.agents)):
            if agent_id not in self.stats.avg_rewards:
                self.stats.avg_rewards[agent_id] = []

            avg_reward = eval_results['total_rewards'][agent_id] / eval_results['total_episodes']
            self.stats.avg_rewards[agent_id].append(avg_reward)

        # Invalid action rates
        for agent_id in range(len(self.agents)):
            if agent_id not in self.stats.invalid_action_rates:
                self.stats.invalid_action_rates[agent_id] = []

            total_actions = eval_results['total_episodes'] * 50  # Rough estimate
            if total_actions > 0:
                invalid_rate = eval_results['invalid_actions'][agent_id] / total_actions
                self.stats.invalid_action_rates[agent_id].append(invalid_rate)

    def _print_progress(self, episode: int, stats: EpisodeStats):
        """Print training progress."""
        print(f"\n📈 Episode {episode}")
        print(f"   Winner: Player {stats.winner if stats.winner is not None else 'None'}")
        print(f"   Turns: {stats.turns}")
        print(f"   Duration: {stats.duration:.2f}s")

        for i, agent in enumerate(self.agents):
            metrics = agent.get_metrics()
            print(f"   Agent {i}: ε={metrics['epsilon']:.3f}, "
                  f"wins={metrics['wins']}, "
                  f"invalid={stats.invalid_actions[i]}")

    def _print_final_results(self):
        """Print final training results."""
        print("\n" + "="*60)
        print("🏆 FINAL TRAINING RESULTS")
        print("="*60)

        for i, agent in enumerate(self.agents):
            metrics = agent.get_metrics()
            print(f"\nAgent {i} ({agent.__class__.__name__}):")
            print(f"  Episodes: {metrics['episode_count']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Total Wins: {metrics['wins']}")
            print(f"  Final ε: {metrics['epsilon']:.4f}")
            print(f"  Steps: {metrics['step_count']:,}")

            if 'avg_loss' in metrics:
                print(f"  Avg Loss: {metrics['avg_loss']:.4f}")
            if 'avg_q_value' in metrics:
                print(f"  Avg Q-Value: {metrics['avg_q_value']:.4f}")

        print(f"\nTotal Training Time: {self.stats.total_time:.1f}s")
        print(f"Episodes Completed: {self.stats.total_episodes}")
        print(f"Avg Episode Time: {self.stats.total_time/self.stats.total_episodes:.2f}s")

    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}_{timestamp}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save agents
        for i, agent in enumerate(self.agents):
            agent_path = checkpoint_path / f"agent_{i}.pth"
            agent.save_checkpoint(str(agent_path))

        # Save training stats
        stats_path = checkpoint_path / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2, default=str)

        # Save config
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save visualization
        if MATPLOTLIB_AVAILABLE:
            plot_path = checkpoint_path / "training_progress.png"
            self.visualizer.save(str(plot_path))

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load agents
        for i, agent in enumerate(self.agents):
            agent_path = checkpoint_dir / f"agent_{i}.pth"
            if agent_path.exists():
                agent.load_checkpoint(str(agent_path))

        # Load training stats
        stats_path = checkpoint_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
                # Reconstruct TrainingStats object
                # This is simplified - would need proper deserialization
                print(f"📥 Loaded training stats from: {stats_path}")

        print(f"✅ Checkpoint loaded: {checkpoint_path}")

    def pause_training(self):
        """Pause training loop."""
        self.pause_requested = True

    def resume_training(self):
        """Resume training loop."""
        self.pause_requested = False


def create_training_config(mode: str = "quick") -> TrainingConfig:
    """Create training configuration for different modes."""
    configs = {
        "quick": TrainingConfig(
            episodes=100,
            save_frequency=50,
            eval_frequency=10,
            max_turns_per_episode=100
        ),
        "standard": TrainingConfig(
            episodes=1000,
            save_frequency=100,
            eval_frequency=50,
            max_turns_per_episode=200
        ),
        "full": TrainingConfig(
            episodes=10000,
            save_frequency=500,
            eval_frequency=100,
            max_turns_per_episode=300
        ),
        "demo": TrainingConfig(
            episodes=20,
            save_frequency=10,
            eval_frequency=5,
            max_turns_per_episode=50
        )
    }

    return configs.get(mode, configs["standard"])


def main():
    """Main training interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Catan RL Training System")
    parser.add_argument("--mode", choices=["demo", "quick", "standard", "full"],
                       default="quick", help="Training mode")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Enable visualization")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                       help="Checkpoint directory")

    args = parser.parse_args()

    # Create config
    config = create_training_config(args.mode)
    if args.episodes:
        config.episodes = args.episodes
    config.visualize = args.visualize
    config.checkpoint_dir = args.checkpoint_dir

    print(f"🎯 Starting Catan RL Training")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {config.episodes}")
    print(f"Checkpoints: {config.checkpoint_dir}")

    # Create and run trainer
    trainer = CatanTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
        trainer.pause_training()


if __name__ == "__main__":
    main()