#!/usr/bin/env python3
"""
Unified CLI for Catan RL System
================================

Interactive menu to run different components of the Catan RL system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print the welcome banner."""
    print(f"""
{Colors.HEADER}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                        CATAN RL SYSTEM                      ║
║                    Interactive Launcher                     ║
╚══════════════════════════════════════════════════════════════╝
{Colors.ENDC}
{Colors.OKCYAN}🎯 Deep Reinforcement Learning for Settlers of Catan{Colors.ENDC}
{Colors.OKBLUE}🤖 PyTorch DQN Framework Successfully Integrated{Colors.ENDC}
""")


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import uvicorn
        print(f"{Colors.OKGREEN}✓ Dependencies OK (PyTorch {torch.__version__}){Colors.ENDC}")
        return True
    except ImportError as e:
        print(f"{Colors.FAIL}✗ Missing dependency: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Run: pip install torch uvicorn{Colors.ENDC}")
        return False


def run_web_server():
    """Start the web server."""
    print(f"{Colors.OKBLUE}🌐 Starting web server...{Colors.ENDC}")
    print(f"{Colors.OKCYAN}   → Open http://localhost:8001 in your browser{Colors.ENDC}")
    print(f"{Colors.WARNING}   → Press Ctrl+C to stop{Colors.ENDC}")

    # Check if server is already running
    try:
        import urllib.request
        response = urllib.request.urlopen('http://localhost:8001/api/state', timeout=1)
        print(f"{Colors.WARNING}⚠️  Server already running at http://localhost:8001{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   → Open the URL above in your browser{Colors.ENDC}")
        input(f"{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        return
    except:
        pass

    # Set PYTHONPATH to include current directory
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)

    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'catan_rl.core.game.web.server:app',
            '--host', '127.0.0.1',
            '--port', '8001',
            '--reload'
        ], env=env)
    except KeyboardInterrupt:
        print(f"\n{Colors.OKGREEN}✓ Server stopped{Colors.ENDC}")
    except FileNotFoundError as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Try: pip install uvicorn{Colors.ENDC}")


def run_dqn_demo():
    """Run DQN integration demo."""
    print(f"{Colors.OKBLUE}🤖 Running DQN integration demo...{Colors.ENDC}")
    try:
        result = subprocess.run([sys.executable, 'scripts/demo_integration.py'],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"{Colors.WARNING}Warnings:{Colors.ENDC}")
            print(result.stderr)
    except Exception as e:
        print(f"{Colors.FAIL}Error running demo: {e}{Colors.ENDC}")


def run_simple_game():
    """Run a simple CLI game."""
    print(f"{Colors.OKBLUE}🎲 Starting simple CLI game...{Colors.ENDC}")

    # Create a simple game script
    game_script = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action
from catan_rl.core.game.agents.random_agent import RandomAgent

def main():
    print("🎲 Simple Catan CLI Game")
    print("=" * 50)

    # Initialize game
    board = standard_board(seed=42)
    game_state = initial_game_state(board, num_players=2)

    # Create agents
    agents = {
        0: RandomAgent(0),
        1: RandomAgent(1)
    }

    turn = 0
    max_turns = 20  # Limit for demo

    while not game_state.winner and turn < max_turns:
        current_player = game_state.current_player
        agent = agents[current_player]

        # Get legal actions
        legal_actions_list = legal_actions(game_state)
        if not legal_actions_list:
            break

        # Agent selects action
        action = agent.select_action(game_state, legal_actions_list)

        print(f"Turn {turn}: Player {current_player} -> {action.action_type.value}")

        # Apply action
        try:
            game_state = apply_action(game_state, action)
        except ValueError as e:
            print(f"  Invalid action: {e}")
            break

        turn += 1

        # Brief pause for readability
        import time
        time.sleep(0.5)

    if game_state.winner:
        print(f"\\n🏆 Player {game_state.winner} wins!")
    else:
        print(f"\\n⏰ Game ended after {turn} turns")

    print("\\nGame Summary:")
    for player_id, player in game_state.players.items():
        print(f"  Player {player_id}: {player.victory_points} VP, "
              f"{len(player.settlements)} settlements, {len(player.cities)} cities")

if __name__ == "__main__":
    main()
"""

    # Write and run the game script
    script_path = Path("temp_game.py")
    try:
        script_path.write_text(game_script)
        subprocess.run([sys.executable, str(script_path)])
    finally:
        if script_path.exists():
            script_path.unlink()


def run_training_demo():
    """Run a training demonstration."""
    print(f"{Colors.OKBLUE}🎯 Running training demo...{Colors.ENDC}")
    print(f"{Colors.WARNING}Note: This will show untrained DQN agents (expect poor play){Colors.ENDC}")

    demo_script = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catan_rl.agents.dqn_agent import DQNAgentFactory
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action

def main():
    print("🎯 DQN Training Demo")
    print("=" * 50)

    # Create DQN agents
    agents = [DQNAgentFactory.create_baseline_agent(i) for i in range(2)]

    for episode in range(3):
        print(f"\\nEpisode {episode + 1}/3:")

        # Initialize game
        board = standard_board(seed=42 + episode)
        game_state = initial_game_state(board, num_players=2)

        # Reset agents
        for agent in agents:
            agent.reset()

        turn = 0
        max_turns = 10  # Short demo

        while not game_state.winner and turn < max_turns:
            current_player = game_state.current_player
            agent = agents[current_player]

            # Get legal actions
            legal_actions_list = legal_actions(game_state)
            if not legal_actions_list:
                break

            # Agent selects action
            try:
                action, metrics = agent.select_action(game_state, legal_actions_list)
                print(f"  Turn {turn}: Player {current_player} -> {action.action_type.value} "
                      f"(ε={metrics.exploration_rate:.3f})")

                # Apply action
                new_state = apply_action(game_state, action)

                # Give reward (simple: +1 for valid action)
                agent.observe_reward(1.0, new_state, False)
                game_state = new_state

            except ValueError as e:
                print(f"  Turn {turn}: Player {current_player} -> INVALID ACTION")
                # Give negative reward for invalid action
                agent.observe_reward(-1.0, game_state, False)
                break

            turn += 1

        # End episode
        for i, agent in enumerate(agents):
            final_reward = 10.0 if game_state.winner == i else 0.0
            agent.observe_reward(final_reward, game_state, True)

        if game_state.winner is not None:
            print(f"  🏆 Player {game_state.winner} wins!")
        else:
            print(f"  ⏰ Episode ended after {turn} turns")

    print(f"\\n📊 Training Summary:")
    for i, agent in enumerate(agents):
        metrics = agent.get_metrics()
        print(f"  Agent {i}: ε={metrics['epsilon']:.3f}, "
              f"steps={metrics['step_count']}, win_rate={metrics['win_rate']:.2f}")

if __name__ == "__main__":
    main()
"""

    # Write and run the demo script
    script_path = Path("temp_training.py")
    try:
        script_path.write_text(demo_script)
        subprocess.run([sys.executable, str(script_path)])
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Training interrupted{Colors.ENDC}")
    finally:
        if script_path.exists():
            script_path.unlink()


def show_system_info():
    """Show system information."""
    print(f"{Colors.OKBLUE}📊 System Information{Colors.ENDC}")
    print("=" * 50)

    # Python version
    print(f"Python: {sys.version}")

    # Dependencies
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("PyTorch: Not installed")

    # Project structure
    print(f"\nProject Structure:")
    important_paths = [
        "src/catan_rl/agents/",
        "src/catan_rl/environments/",
        "src/catan_rl/core/game/",
        "scripts/",
        "src/catan/web/"
    ]

    for path in important_paths:
        if Path(path).exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path}")

    # Integration status
    print(f"\n🔗 Integration Status:")
    try:
        # Test imports
        from catan_rl.agents.dqn_agent import DQNAgentFactory
        from catan_rl.environments.action_space import CatanActionSpace
        print(f"  ✓ DQN Framework")

        agent = DQNAgentFactory.create_baseline_agent(0)
        state_shape = agent.state_encoder.get_state_shape()
        action_space_size = agent.action_space.get_action_space_size()
        print(f"  ✓ State Encoder: {state_shape}")
        print(f"  ✓ Action Space: {action_space_size} actions")
    except Exception as e:
        print(f"  ✗ Integration Error: {e}")


def run_advanced_training():
    """Run the advanced training system."""
    print(f"{Colors.OKBLUE}🚀 Advanced Training System{Colors.ENDC}")
    print("=" * 50)

    print("Available training modes:")
    print(f"{Colors.OKCYAN}1.{Colors.ENDC} 🏃 Demo (20 episodes)")
    print(f"{Colors.OKCYAN}2.{Colors.ENDC} ⚡ Quick (100 episodes)")
    print(f"{Colors.OKCYAN}3.{Colors.ENDC} 🎯 Standard (1,000 episodes)")
    print(f"{Colors.OKCYAN}4.{Colors.ENDC} 🔥 Full (10,000 episodes)")
    print(f"{Colors.OKCYAN}5.{Colors.ENDC} 🛠️  Custom")

    try:
        mode_choice = input(f"\n{Colors.BOLD}Choose training mode (1-5): {Colors.ENDC}").strip()

        mode_map = {
            '1': 'demo',
            '2': 'quick',
            '3': 'standard',
            '4': 'full',
            '5': 'custom'
        }

        mode = mode_map.get(mode_choice, 'quick')

        if mode == 'custom':
            try:
                episodes = int(input("Number of episodes: "))
                eval_freq = int(input("Evaluation frequency: "))
                save_freq = int(input("Save frequency: "))

                training_script = f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_system import CatanTrainer, TrainingConfig

config = TrainingConfig(
    episodes={episodes},
    eval_frequency={eval_freq},
    save_frequency={save_freq},
    max_turns_per_episode=200
)

print("🚀 Starting custom training...")
trainer = CatanTrainer(config)

try:
    trainer.train()
except KeyboardInterrupt:
    print("\\n⏸️  Training interrupted")
"""
            except ValueError:
                print(f"{Colors.WARNING}Invalid input, using quick mode{Colors.ENDC}")
                mode = 'quick'

        if mode != 'custom':
            training_script = f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_system import CatanTrainer, create_training_config

print("🚀 Starting {mode} training...")
config = create_training_config('{mode}')
trainer = CatanTrainer(config)

try:
    trainer.train()
except KeyboardInterrupt:
    print("\\n⏸️  Training interrupted")
"""

        # Write and run the training script
        script_path = Path("temp_advanced_training.py")
        try:
            script_path.write_text(training_script)
            subprocess.run([sys.executable, str(script_path)])
        except Exception as e:
            print(f"{Colors.FAIL}Training failed: {e}{Colors.ENDC}")
        finally:
            if script_path.exists():
                script_path.unlink()

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Training cancelled{Colors.ENDC}")


def run_training_tournament():
    """Run a tournament between different agent types."""
    print(f"{Colors.OKBLUE}🏆 Training Tournament{Colors.ENDC}")
    print("=" * 50)

    tournament_script = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catan_rl.agents.dqn_agent import DQNAgentFactory
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state
from catan_rl.core.game.engine.rules import legal_actions, apply_action
import time

def run_tournament_match(agents, num_games=10):
    print(f"🏆 Tournament: {num_games} games")

    wins = {i: 0 for i in range(len(agents))}
    total_scores = {i: 0 for i in range(len(agents))}

    for game in range(num_games):
        print(f"\\rGame {game+1}/{num_games}", end="", flush=True)

        # Initialize game
        board = standard_board(seed=42 + game)
        game_state = initial_game_state(board, num_players=len(agents))

        # Reset agents
        for agent in agents:
            agent.reset()
            agent.set_evaluation_mode(True)

        turn = 0
        max_turns = 100

        while not game_state.winner and turn < max_turns:
            current_player = game_state.current_player
            agent = agents[current_player]

            legal_actions_list = legal_actions(game_state)
            if not legal_actions_list:
                break

            try:
                action, _ = agent.select_action(game_state, legal_actions_list)
                game_state = apply_action(game_state, action)
            except:
                # Invalid action, skip turn
                break

            turn += 1

        # Record results
        if game_state.winner is not None:
            wins[game_state.winner] += 1

        for player_id, player in game_state.players.items():
            total_scores[player_id] += player.victory_points

    print()  # New line after progress
    return wins, total_scores

def main():
    print("Creating tournament agents...")

    # Create different agent types
    agents = [
        DQNAgentFactory.create_baseline_agent(0),
        DQNAgentFactory.create_spatial_agent(1),
        DQNAgentFactory.create_rainbow_agent(2),
        DQNAgentFactory.create_baseline_agent(3)
    ]

    agent_names = ["Baseline DQN", "Spatial CNN", "Rainbow DQN", "Baseline DQN #2"]

    print("\\n🎮 TOURNAMENT SETUP")
    print("=" * 40)
    for i, name in enumerate(agent_names):
        print(f"  Player {i}: {name}")

    # Run tournament
    wins, scores = run_tournament_match(agents, num_games=20)

    print("\\n🏆 TOURNAMENT RESULTS")
    print("=" * 40)

    # Sort by wins
    results = [(i, agent_names[i], wins[i], scores[i]/20) for i in range(len(agents))]
    results.sort(key=lambda x: x[2], reverse=True)

    for rank, (player_id, name, win_count, avg_score) in enumerate(results, 1):
        win_rate = win_count / 20 * 100
        print(f"{rank}. {name}")
        print(f"   Wins: {win_count}/20 ({win_rate:.1f}%)")
        print(f"   Avg Score: {avg_score:.1f} VP")
        print()

if __name__ == "__main__":
    main()
"""

    # Write and run tournament script
    script_path = Path("temp_tournament.py")
    try:
        script_path.write_text(tournament_script)
        print(f"{Colors.OKCYAN}Starting tournament between different DQN agents...{Colors.ENDC}")
        subprocess.run([sys.executable, str(script_path)])
    except Exception as e:
        print(f"{Colors.FAIL}Tournament failed: {e}{Colors.ENDC}")
    finally:
        if script_path.exists():
            script_path.unlink()


def run_multi_agent_training():
    """Run the new multi-agent RL training system."""
    print(f"{Colors.OKBLUE}🚀 Multi-Agent RL Training System{Colors.ENDC}")
    print("=" * 50)
    print(f"{Colors.OKCYAN}Train AI agents using cutting-edge RL frameworks:{Colors.ENDC}")
    print("  • Ray RLlib (PPO) - Distributed policy optimization")
    print("  • TorchRL (SAC) - Soft actor-critic with entropy")
    print("  • LangGraph (LLM) - Strategic reasoning with LLMs")
    print()

    try:
        subprocess.run([sys.executable, 'train_agents.py'])
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Training interrupted{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Error running training: {e}{Colors.ENDC}")


def main_menu():
    """Show the main interactive menu."""
    print_banner()

    if not check_dependencies():
        return

    while True:
        print(f"\n{Colors.BOLD}🎮 MAIN MENU{Colors.ENDC}")
        print("=" * 30)
        print(f"{Colors.OKCYAN}1.{Colors.ENDC} 🌐 Start Web Interface (Interactive Game)")
        print(f"{Colors.OKCYAN}2.{Colors.ENDC} 🤖 Multi-Agent RL Training (PPO/SAC/LLM) [NEW!]")
        print(f"{Colors.OKCYAN}3.{Colors.ENDC} 🎲 Play Simple CLI Game")
        print(f"{Colors.OKCYAN}4.{Colors.ENDC} 🚀 Advanced Training System (Legacy DQN)")
        print(f"{Colors.OKCYAN}5.{Colors.ENDC} 🏆 Training Tournament")
        print(f"{Colors.OKCYAN}6.{Colors.ENDC} 📊 Show System Info")
        print(f"{Colors.OKCYAN}7.{Colors.ENDC} ❌ Exit")

        try:
            choice = input(f"\n{Colors.BOLD}Choose option (1-7): {Colors.ENDC}").strip()

            if choice == '1':
                run_web_server()
            elif choice == '2':
                run_multi_agent_training()
            elif choice == '3':
                run_simple_game()
            elif choice == '4':
                run_advanced_training()
            elif choice == '5':
                run_training_tournament()
            elif choice == '6':
                show_system_info()
            elif choice == '7':
                print(f"{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please select 1-7.{Colors.ENDC}")

        except KeyboardInterrupt:
            print(f"\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
            break
        except EOFError:
            print(f"\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}")
            break


if __name__ == "__main__":
    main_menu()