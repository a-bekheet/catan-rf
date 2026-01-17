#!/usr/bin/env python3
"""Training script for RL bots playing against each other."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from catan_rl.core.game.agents.random_agent import RandomAgent
from catan_rl.core.game.agents.rl_agent import RLAgent
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import GameState, initial_game_state, TurnPhase
from catan_rl.core.game.engine.types import Action


class MultiAgentTrainer:
    """Manages multi-agent training sessions."""

    def __init__(self, num_episodes: int = 1000, verbose: bool = False):
        self.num_episodes = num_episodes
        self.verbose = verbose
        self.progress_line = 0  # Track which line the progress bar is on

        # Initialize agents
        self.agents = [
            RLAgent(player_id=0, epsilon=0.3, seed=42),
            RLAgent(player_id=1, epsilon=0.3, seed=43),
            RandomAgent(player_id=2, seed=44),
            RandomAgent(player_id=3, seed=45),
        ]

        # Training stats
        self.episode_stats = {
            "wins": [0, 0, 0, 0],
            "total_rewards": [0.0, 0.0, 0.0, 0.0],
            "episodes_completed": 0,
            "total_turns": 0,
        }

        # Detailed game analysis stats
        self.detailed_game_logs = []
        self.strategy_analysis = {
            "RL": {"vp": [], "settlements": [], "cities": [], "roads": [], "dev_cards": [], "knights": []},
            "Random": {"vp": [], "settlements": [], "cities": [], "roads": [], "dev_cards": [], "knights": []}
        }

    def run_episode(self) -> Dict[str, object]:
        """Run a single game episode."""
        # Reset environment
        board = standard_board()
        state = initial_game_state(board, num_players=4)

        # Reset agents
        for agent in self.agents:
            agent.reset()

        episode_rewards = [0.0, 0.0, 0.0, 0.0]
        step_count = 0
        max_steps = 1000

        while step_count < max_steps:
            # Check if game is over
            if state.winner is not None:
                break

            current_player = state.current_player
            legal_actions = state.legal_actions()

            if not legal_actions:
                if self.verbose:
                    print(f"No legal actions for player {current_player} in phase {state.phase}")
                break

            # Get action from current agent
            agent = self.agents[current_player]

            try:
                action = agent.select_action(state, legal_actions)
                next_state = state.apply(action)
            except Exception as e:
                if self.verbose:
                    print(f"Error applying action: {e}")
                break

            # Compute rewards for all agents
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'compute_reward'):
                    reward = agent.compute_reward(state, next_state)
                    episode_rewards[i] += reward

                    # Update RL agents
                    if hasattr(agent, 'update'):
                        agent.update(state, action, reward, next_state)

            state = next_state
            step_count += 1

            if self.verbose and step_count % 100 == 0:
                print(f"Step {step_count}, Phase: {state.phase}, Player: {state.current_player}")

        # Episode completed
        self.episode_stats["episodes_completed"] += 1
        self.episode_stats["total_turns"] += step_count

        if state.winner is not None:
            self.episode_stats["wins"][state.winner] += 1

        for i in range(4):
            self.episode_stats["total_rewards"][i] += episode_rewards[i]

        # Collect detailed final stats for analysis
        final_stats = {}
        for i in range(4):
            player = state.players[i]
            final_stats[f"player_{i}"] = {
                "victory_points": player.victory_points,
                "settlements": len(player.settlements),
                "cities": len(player.cities),
                "roads": len(player.roads),
                "dev_cards_bought": len(player.dev_cards),
                "knights_played": player.knights_played,
                "total_resources": sum(player.resources.values()),
                "agent_type": "RL" if hasattr(self.agents[i], 'epsilon') else "Random"
            }

        return {
            "winner": state.winner,
            "steps": step_count,
            "rewards": episode_rewards,
            "final_vps": [state.players[i].victory_points for i in range(4)],
            "detailed_stats": final_stats,
        }

    def train(self):
        """Run training loop."""
        print(f"Starting training for {self.num_episodes} episodes...")

        # Setup persistent progress bar if in verbose mode
        self._setup_persistent_progress()

        start_time = time.time()

        for episode in range(self.num_episodes):
            episode_result = self.run_episode()

            # Log detailed game statistics
            self._log_detailed_stats(episode_result)

            # Decay exploration for RL agents
            for agent in self.agents:
                if hasattr(agent, 'decay_epsilon'):
                    agent.decay_epsilon()

            # Print progress
            if self.verbose:
                # Frequent progress updates in verbose mode
                if (episode + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    self.print_progress_bar(episode + 1, elapsed, episode_result)

                # Full stats every 100 episodes
                if (episode + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    # Move cursor to a safe area above progress bar for stats
                    if self.verbose:
                        terminal_height = self._get_terminal_height()
                        print(f"\033[s", end="")  # Save cursor
                        print(f"\033[{terminal_height-3};1H", end="")  # Move above progress bar
                    self.print_stats(episode + 1, elapsed)
                    if self.verbose:
                        print(f"\033[u", end="")  # Restore cursor
            else:
                # Normal progress updates
                if (episode + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    self.print_stats(episode + 1, elapsed)

        # Clear persistent progress bar and show completion
        if self.verbose:
            terminal_height = self._get_terminal_height()
            print(f"\033[{terminal_height};1H\033[K", end="")  # Clear bottom line

        print("\nTraining completed!")
        self.print_final_stats()
        self.print_strategy_analysis()
        self.save_models()

    def print_stats(self, episode: int, elapsed_time: float):
        """Print training statistics."""
        wins = self.episode_stats["wins"]
        rewards = self.episode_stats["total_rewards"]

        print(f"\n--- Episode {episode} ---")
        print(f"Time elapsed: {elapsed_time:.1f}s")
        print("Win rates:")
        for i in range(4):
            agent_type = "RL" if hasattr(self.agents[i], 'epsilon') else "Random"
            win_rate = wins[i] / episode * 100 if episode > 0 else 0
            avg_reward = rewards[i] / episode if episode > 0 else 0
            print(f"  Player {i} ({agent_type}): {win_rate:.1f}% wins, avg reward: {avg_reward:.2f}")

    def _setup_persistent_progress(self):
        """Setup terminal for persistent progress bar at bottom."""
        if self.verbose:
            # Clear screen and save cursor position
            print("\033[2J\033[H", end="")  # Clear screen, move to top
            # Reserve space for progress bar at bottom
            print("\n" * (self._get_terminal_height() - 2))  # Move to near bottom
            print("Initializing training...", flush=True)

    def _get_terminal_height(self):
        """Get terminal height or default."""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 24  # Default fallback

    def _update_persistent_progress(self, episode: int, elapsed_time: float, episode_result: Dict):
        """Update the persistent progress bar at bottom of screen."""
        # Create progress bar
        progress = episode / self.num_episodes
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Calculate stats
        total_turns = self.episode_stats["total_turns"]
        avg_turns_per_game = total_turns / episode if episode > 0 else 0
        games_per_minute = (episode / elapsed_time * 60) if elapsed_time > 0 else 0

        # Winner info for this episode
        winner_info = ""
        if episode_result["winner"] is not None:
            agent_type = "RL" if episode_result["winner"] < 2 else "Random"
            winner_info = f" | Winner: P{episode_result['winner']} ({agent_type})"
        else:
            winner_info = " | No winner (timeout)"

        # Current RL agent epsilon values
        epsilon_info = ""
        rl_epsilons = []
        for agent in self.agents:
            if hasattr(agent, 'epsilon'):
                rl_epsilons.append(f"{agent.epsilon:.3f}")
        if rl_epsilons:
            epsilon_info = f" | ε: {'/'.join(rl_epsilons)}"

        # Build progress line
        progress_text = (f"[{bar}] {episode}/{self.num_episodes} ({progress*100:.1f}%) | "
                        f"⏱️ {elapsed_time:.1f}s | 🎮 {games_per_minute:.1f} games/min | "
                        f"🎲 {total_turns} total turns ({avg_turns_per_game:.1f}/game){winner_info}{epsilon_info}")

        # Save cursor, move to bottom, update progress, restore cursor
        terminal_height = self._get_terminal_height()
        print(f"\033[s", end="")  # Save cursor position
        print(f"\033[{terminal_height};1H", end="")  # Move to bottom line
        print(f"\033[K{progress_text}", end="", flush=True)  # Clear line and write progress
        print(f"\033[u", end="")  # Restore cursor position

    def print_progress_bar(self, episode: int, elapsed_time: float, episode_result: Dict):
        """Print progress bar (delegates to persistent version in verbose mode)."""
        if self.verbose:
            self._update_persistent_progress(episode, elapsed_time, episode_result)
        else:
            # Fallback to simple progress for non-verbose mode
            progress = episode / self.num_episodes
            print(f"\rProgress: {episode}/{self.num_episodes} ({progress*100:.1f}%)", end="", flush=True)

    def print_final_stats(self):
        """Print final training statistics."""
        print("\n" + "="*50)
        print("FINAL TRAINING RESULTS")
        print("="*50)

        episodes = self.episode_stats["episodes_completed"]
        wins = self.episode_stats["wins"]
        rewards = self.episode_stats["total_rewards"]
        total_turns = self.episode_stats["total_turns"]

        print(f"Total episodes: {episodes}")
        print(f"Total turns played: {total_turns:,}")
        print(f"Average turns per game: {total_turns/episodes:.1f}" if episodes > 0 else "Average turns per game: 0")
        print("\nFinal standings:")

        # Sort by win rate
        standings = []
        for i in range(4):
            agent_type = "RL Agent" if hasattr(self.agents[i], 'epsilon') else "Random Agent"
            win_rate = wins[i] / episodes * 100 if episodes > 0 else 0
            avg_reward = rewards[i] / episodes if episodes > 0 else 0
            standings.append((i, agent_type, win_rate, avg_reward, wins[i]))

        standings.sort(key=lambda x: x[2], reverse=True)

        for rank, (player_id, agent_type, win_rate, avg_reward, total_wins) in enumerate(standings, 1):
            print(f"{rank}. Player {player_id} ({agent_type}): "
                  f"{win_rate:.1f}% win rate ({total_wins} wins), "
                  f"avg reward: {avg_reward:.2f}")

    def _log_detailed_stats(self, episode_result):
        """Log detailed game statistics for strategy analysis."""
        if "detailed_stats" not in episode_result:
            return

        game_log = {
            "episode": self.episode_stats["episodes_completed"],
            "winner": episode_result["winner"],
            "steps": episode_result["steps"],
            "players": episode_result["detailed_stats"]
        }
        self.detailed_game_logs.append(game_log)

        # Aggregate stats by agent type
        for player_id, stats in episode_result["detailed_stats"].items():
            agent_type = stats["agent_type"]
            self.strategy_analysis[agent_type]["vp"].append(stats["victory_points"])
            self.strategy_analysis[agent_type]["settlements"].append(stats["settlements"])
            self.strategy_analysis[agent_type]["cities"].append(stats["cities"])
            self.strategy_analysis[agent_type]["roads"].append(stats["roads"])
            self.strategy_analysis[agent_type]["dev_cards"].append(stats["dev_cards_bought"])
            self.strategy_analysis[agent_type]["knights"].append(stats["knights_played"])

    def print_strategy_analysis(self):
        """Print detailed strategy analysis comparing RL vs Random agents."""
        print("\n" + "="*60)
        print("📊 DETAILED STRATEGY ANALYSIS")
        print("="*60)

        if not self.detailed_game_logs:
            print("No detailed game logs available.")
            return

        # Calculate averages for each agent type
        for agent_type in ["RL", "Random"]:
            stats = self.strategy_analysis[agent_type]
            if not stats["vp"]:  # No data for this agent type
                continue

            print(f"\n🤖 {agent_type} Agents Performance:")
            print(f"   Games played: {len(stats['vp'])}")

            avg_stats = {
                "Victory Points": sum(stats["vp"]) / len(stats["vp"]),
                "Settlements": sum(stats["settlements"]) / len(stats["settlements"]),
                "Cities": sum(stats["cities"]) / len(stats["cities"]),
                "Roads": sum(stats["roads"]) / len(stats["roads"]),
                "Dev Cards": sum(stats["dev_cards"]) / len(stats["dev_cards"]),
                "Knights Played": sum(stats["knights"]) / len(stats["knights"])
            }

            for metric, value in avg_stats.items():
                print(f"   {metric:15}: {value:.1f}")

        # Game completion analysis
        total_games = len(self.detailed_game_logs)
        wins_by_type = {"RL": 0, "Random": 0, "None": 0}

        for game in self.detailed_game_logs:
            if game["winner"] is not None:
                winner_type = game["players"][f"player_{game['winner']}"]["agent_type"]
                wins_by_type[winner_type] += 1
            else:
                wins_by_type["None"] += 1

        print(f"\n🏆 Win Analysis ({total_games} games):")
        for agent_type, wins in wins_by_type.items():
            win_rate = (wins / total_games) * 100 if total_games > 0 else 0
            print(f"   {agent_type:10}: {wins:3d} wins ({win_rate:5.1f}%)")

        # Strategy insights
        print(f"\n🔍 Strategy Insights:")
        if self.strategy_analysis["RL"]["vp"] and self.strategy_analysis["Random"]["vp"]:
            rl_avg_vp = sum(self.strategy_analysis["RL"]["vp"]) / len(self.strategy_analysis["RL"]["vp"])
            random_avg_vp = sum(self.strategy_analysis["Random"]["vp"]) / len(self.strategy_analysis["Random"]["vp"])

            if rl_avg_vp > random_avg_vp:
                vp_diff = rl_avg_vp - random_avg_vp
                print(f"   ✅ RL agents average {vp_diff:.1f} more VPs per game")
            else:
                vp_diff = random_avg_vp - rl_avg_vp
                print(f"   ❌ Random agents average {vp_diff:.1f} more VPs per game")

            # Building strategy comparison
            rl_settlements = sum(self.strategy_analysis["RL"]["settlements"]) / len(self.strategy_analysis["RL"]["settlements"])
            random_settlements = sum(self.strategy_analysis["Random"]["settlements"]) / len(self.strategy_analysis["Random"]["settlements"])

            if rl_settlements > random_settlements:
                print(f"   🏘️  RL agents build more settlements ({rl_settlements:.1f} vs {random_settlements:.1f})")
            else:
                print(f"   🏘️  Random agents build more settlements ({random_settlements:.1f} vs {rl_settlements:.1f})")

            rl_cities = sum(self.strategy_analysis["RL"]["cities"]) / len(self.strategy_analysis["RL"]["cities"])
            random_cities = sum(self.strategy_analysis["Random"]["cities"]) / len(self.strategy_analysis["Random"]["cities"])

            if rl_cities > random_cities:
                print(f"   🏙️  RL agents build more cities ({rl_cities:.1f} vs {random_cities:.1f})")
            else:
                print(f"   🏙️  Random agents build more cities ({random_cities:.1f} vs {rl_cities:.1f})")

    def save_models(self):
        """Save trained RL agent models."""
        print("\n💾 Saving trained models...")
        saved_count = 0
        for agent in self.agents:
            if hasattr(agent, 'save_model'):
                agent.save_model()
                print(f"   Saved: {agent.model_path}")
                saved_count += 1

        if saved_count > 0:
            print(f"✅ {saved_count} models saved successfully!")
        else:
            print("No RL agents to save")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train RL bots to play Catan")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    trainer = MultiAgentTrainer(
        num_episodes=args.episodes,
        verbose=args.verbose
    )

    trainer.train()


if __name__ == "__main__":
    main()