#!/usr/bin/env python3
"""
Multi-Agent RL Training CLI
============================

Beautiful command-line interface for training multiple RL agents to play Catan.
Supports Ray RLlib (PPO), TorchRL (SAC), and LangGraph (LLM).

Usage:
    python train_agents.py                    # Interactive menu
    python train_agents.py --agent ppo        # Train PPO agent
    python train_agents.py --agent sac        # Train SAC agent
    python train_agents.py --agent llm        # Train LLM agent
    python train_agents.py --agent all        # Train all agents
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich not installed. Install with: pip install rich")

from catan_rl.agents.base_rl_agent import BaseRLAgent, RandomAgent
from catan_rl.agents.rllib_ppo_agent import RLlibPPOAgent
from catan_rl.agents.torchrl_sac_agent import TorchRLSACAgent
from catan_rl.agents.langgraph_llm_agent import LangGraphLLMAgent
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action


class TrainingSession:
    """Manages training session for multiple agents."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.checkpoint_dir = Path("checkpoints") / f"multi_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training metrics
        self.episode_count = 0
        self.start_time = None
        self.episode_times = []
        self.win_counts = {}
        self.total_rewards = {}

    def create_agent(self, agent_type: str, agent_id: int, config: Dict[str, Any]) -> BaseRLAgent:
        """Create an agent of specified type."""
        if agent_type == 'ppo':
            return RLlibPPOAgent(agent_id, config)
        elif agent_type == 'sac':
            return TorchRLSACAgent(agent_id, config)
        elif agent_type == 'llm':
            return LangGraphLLMAgent(agent_id, config)
        elif agent_type == 'random':
            return RandomAgent(agent_id, config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def train_single_agent(
        self,
        agent_type: str,
        num_episodes: int = 100,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Train a single agent against random opponents."""
        self.console.print(f"\n[bold cyan]Training {agent_type.upper()} Agent[/bold cyan]\n")

        # Create agents
        config = agent_config or self._get_default_config(agent_type)
        agent = self.create_agent(agent_type, agent_id=0, config=config)
        opponents = [RandomAgent(i+1, {}) for i in range(3)]
        agents = [agent] + opponents

        # Initialize tracking
        self.win_counts = {i: 0 for i in range(4)}
        self.total_rewards = {i: 0.0 for i in range(4)}
        self.start_time = time.time()

        # Training loop with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Training {agent_type.upper()}...",
                total=num_episodes
            )

            for episode in range(num_episodes):
                episode_start = time.time()

                # Run episode
                winner, rewards, turns = self._run_episode(agents)

                # Update metrics
                self.episode_count += 1
                self.win_counts[winner] += 1
                for agent_id, reward in rewards.items():
                    self.total_rewards[agent_id] += reward
                self.episode_times.append(time.time() - episode_start)

                # Update progress
                progress.update(task, advance=1)

                # Display stats every 10 episodes
                if (episode + 1) % 10 == 0:
                    self._display_training_stats(agent_type, episode + 1, num_episodes)

                # Save checkpoint every 50 episodes
                if (episode + 1) % 50 == 0:
                    checkpoint_path = self.checkpoint_dir / f"{agent_type}_ep{episode+1}"
                    agent.save_checkpoint(checkpoint_path)
                    self.console.print(f"[green]✓ Checkpoint saved: {checkpoint_path}[/green]")

        # Final save
        final_path = self.checkpoint_dir / f"{agent_type}_final"
        agent.save_checkpoint(final_path)
        self.console.print(f"\n[bold green]✓ Training complete! Final model saved to: {final_path}[/bold green]")

        # Display final stats
        self._display_final_stats(agent_type, num_episodes)

    def train_all_agents(
        self,
        num_episodes: int = 100,
        tournament_episodes: int = 50
    ) -> None:
        """Train all three agents, then run a tournament."""
        self.console.print(Panel.fit(
            "[bold cyan]Multi-Agent Training Program[/bold cyan]\n\n"
            "Training sequence:\n"
            "  1. Ray RLlib PPO Agent\n"
            "  2. TorchRL SAC Agent\n"
            "  3. LangGraph LLM Agent\n"
            f"  4. Tournament ({tournament_episodes} episodes)\n",
            border_style="cyan"
        ))

        # Train each agent
        for agent_type in ['ppo', 'sac', 'llm']:
            self.train_single_agent(agent_type, num_episodes)
            self.console.print("\n" + "="*80 + "\n")

        # Run tournament with all trained agents
        self.console.print(Panel.fit(
            "[bold yellow]Tournament Mode[/bold yellow]\n\n"
            "All trained agents compete against each other!",
            border_style="yellow"
        ))
        self._run_tournament(tournament_episodes)

    def _run_episode(self, agents: List[BaseRLAgent]) -> tuple:
        """Run a single game episode."""
        # Initialize game
        board = standard_board()
        state = initial_game_state(board, num_players=4)

        # Episode tracking
        episode_rewards = {i: 0.0 for i in range(4)}
        turn_count = 0
        max_turns = 500  # Prevent infinite games

        # Game loop
        while state.winner is None and turn_count < max_turns:
            current_player_id = state.current_player
            current_agent = agents[current_player_id]

            # Get legal actions
            actions = legal_actions(state)
            if not actions:
                break

            # Agent selects action
            action, metrics = current_agent.select_action(state, actions)

            # Apply action and get next state
            next_state = apply_action(state, action)

            # Compute reward
            reward = self._compute_reward(state, next_state, current_player_id)
            episode_rewards[current_player_id] += reward

            # Update agent
            current_agent.update(state, action, reward, next_state, next_state.winner is not None)

            # Move to next state
            state = next_state
            turn_count += 1

        # Episode ended
        winner = state.winner if state.winner is not None else 0

        # Notify agents of episode end
        for i, agent in enumerate(agents):
            won = (i == winner)
            agent.record_episode_end(episode_rewards[i], turn_count, won)
            agent.reset_episode()

        return winner, episode_rewards, turn_count

    def _compute_reward(
        self,
        state,
        next_state,
        player_id: int
    ) -> float:
        """Compute reward for state transition."""
        player = next_state.players[player_id]
        prev_player = state.players[player_id]

        # Win bonus
        if next_state.winner == player_id:
            return 100.0

        # VP progress
        vp_gain = player.victory_points - prev_player.victory_points
        reward = vp_gain * 10.0

        # Building rewards
        new_settlements = len(player.settlements) - len(prev_player.settlements)
        new_cities = len(player.cities) - len(prev_player.cities)
        reward += new_settlements * 5.0 + new_cities * 8.0

        # Small time penalty to encourage winning
        reward -= 0.01

        return reward

    def _display_training_stats(self, agent_type: str, episode: int, total_episodes: int) -> None:
        """Display current training statistics."""
        if not RICH_AVAILABLE:
            return

        # Calculate metrics
        win_rate = self.win_counts[0] / episode * 100
        avg_reward = self.total_rewards[0] / episode
        avg_time = sum(self.episode_times[-10:]) / min(10, len(self.episode_times))

        # Estimate time remaining
        remaining_episodes = total_episodes - episode
        est_time_remaining = avg_time * remaining_episodes
        eta = datetime.now() + timedelta(seconds=est_time_remaining)

        # Create stats table
        table = Table(title=f"{agent_type.upper()} Agent - Episode {episode}/{total_episodes}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Avg Reward", f"{avg_reward:.2f}")
        table.add_row("Avg Episode Time", f"{avg_time:.2f}s")
        table.add_row("ETA", eta.strftime("%H:%M:%S"))

        self.console.print(table)

    def _display_final_stats(self, agent_type: str, num_episodes: int) -> None:
        """Display final training statistics."""
        if not RICH_AVAILABLE:
            return

        training_time = time.time() - self.start_time
        win_rate = self.win_counts[0] / num_episodes * 100
        avg_reward = self.total_rewards[0] / num_episodes

        # Create summary panel
        summary = f"""
[bold]Training Summary for {agent_type.upper()} Agent[/bold]

Total Episodes: {num_episodes}
Training Time: {training_time/60:.1f} minutes
Win Rate: {win_rate:.1f}%
Average Reward: {avg_reward:.2f}

Agent vs Random Opponents:
  Agent Wins: {self.win_counts[0]} ({win_rate:.1f}%)
  Opponent Wins: {sum(self.win_counts[i] for i in range(1, 4))} ({100-win_rate:.1f}%)
"""

        self.console.print(Panel(summary, border_style="green", box=box.DOUBLE))

    def _run_tournament(self, num_episodes: int) -> None:
        """Run tournament with all trained agents."""
        # Load trained agents
        agents = []
        for agent_id, agent_type in enumerate(['ppo', 'sac', 'llm', 'random']):
            if agent_type != 'random':
                config = self._get_default_config(agent_type)
                agent = self.create_agent(agent_type, agent_id, config)

                # Load final checkpoint
                checkpoint_path = self.checkpoint_dir / f"{agent_type}_final"
                if checkpoint_path.exists():
                    agent.load_checkpoint(checkpoint_path)
                    agent.set_training_mode(False)  # Evaluation mode

                agents.append(agent)
            else:
                agents.append(RandomAgent(agent_id, {}))

        # Run tournament
        tournament_wins = {i: 0 for i in range(4)}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[yellow]Tournament in progress...", total=num_episodes)

            for episode in range(num_episodes):
                winner, _, _ = self._run_episode(agents)
                tournament_wins[winner] += 1
                progress.update(task, advance=1)

        # Display tournament results
        self._display_tournament_results(tournament_wins, num_episodes)

    def _display_tournament_results(self, wins: Dict[int, int], total: int) -> None:
        """Display tournament results."""
        if not RICH_AVAILABLE:
            return

        table = Table(title="Tournament Results", box=box.DOUBLE, border_style="yellow")
        table.add_column("Player", style="cyan")
        table.add_column("Agent Type", style="magenta")
        table.add_column("Wins", style="green")
        table.add_column("Win Rate", style="yellow")

        agent_names = ['PPO', 'SAC', 'LLM', 'Random']
        for player_id in range(4):
            win_count = wins[player_id]
            win_rate = win_count / total * 100
            table.add_row(
                f"Player {player_id}",
                agent_names[player_id],
                str(win_count),
                f"{win_rate:.1f}%"
            )

        self.console.print("\n")
        self.console.print(table)

    def _get_default_config(self, agent_type: str) -> Dict[str, Any]:
        """Get default configuration for agent type."""
        if agent_type == 'ppo':
            return {
                'state_encoder': {'type': 'feature'},
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'num_workers': 2,
            }
        elif agent_type == 'sac':
            return {
                'state_encoder': {'type': 'feature'},
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'alpha': 0.2,
                'batch_size': 256,
            }
        elif agent_type == 'llm':
            return {
                'llm_provider': 'openai',
                'model_name': 'gpt-4-turbo-preview',
                'temperature': 0.7,
            }
        else:
            return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RL agents to play Catan")
    parser.add_argument(
        '--agent',
        choices=['ppo', 'sac', 'llm', 'all'],
        help="Which agent to train (ppo, sac, llm, or all)"
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help="Number of training episodes (default: 100)"
    )
    parser.add_argument(
        '--tournament',
        type=int,
        default=50,
        help="Number of tournament episodes when training all agents (default: 50)"
    )

    args = parser.parse_args()

    # Initialize console
    console = Console() if RICH_AVAILABLE else None
    session = TrainingSession(console)

    # Display welcome banner
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]Catan Multi-Agent RL Training System[/bold cyan]\n\n"
            "Train AI agents using cutting-edge RL frameworks:\n"
            "  • Ray RLlib (PPO) - Distributed policy optimization\n"
            "  • TorchRL (SAC) - Soft actor-critic with entropy\n"
            "  • LangGraph (LLM) - Strategic reasoning with LLMs\n",
            border_style="cyan",
            box=box.DOUBLE
        ))

    # Interactive mode if no arguments
    if args.agent is None:
        if RICH_AVAILABLE:
            choice = Prompt.ask(
                "\n[cyan]Select training mode[/cyan]",
                choices=['1', '2', '3', '4'],
                default='1'
            )
            choice_map = {'1': 'ppo', '2': 'sac', '3': 'llm', '4': 'all'}
            args.agent = choice_map[choice]
        else:
            print("\n1. Train PPO Agent")
            print("2. Train SAC Agent")
            print("3. Train LLM Agent")
            print("4. Train All Agents")
            choice = input("\nSelect (1-4): ").strip()
            choice_map = {'1': 'ppo', '2': 'sac', '3': 'llm', '4': 'all'}
            args.agent = choice_map.get(choice, 'ppo')

    # Train agents
    try:
        if args.agent == 'all':
            session.train_all_agents(args.episodes, args.tournament)
        else:
            session.train_single_agent(args.agent, args.episodes)

    except KeyboardInterrupt:
        if console:
            console.print("\n\n[yellow]Training interrupted by user.[/yellow]")
        else:
            print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        if console:
            console.print(f"\n[red]Error: {e}[/red]")
        else:
            print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
