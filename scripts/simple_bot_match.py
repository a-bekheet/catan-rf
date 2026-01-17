#!/usr/bin/env python3
"""Simple bot match - 4 bots playing against each other."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path for local development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.core.game.agents.random_agent import RandomAgent
from catan_rl.core.game.agents.rl_agent import RLAgent
from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state


def run_bot_match():
    """Run a single match between 4 bots."""
    print("🤖 Starting Bot Match - 4 AI players!")
    print("=" * 50)

    # Create agents (RL agents will automatically load saved models if available)
    agents = [
        RLAgent(player_id=0, epsilon=0.1),  # Lower epsilon for evaluation
        RLAgent(player_id=1, epsilon=0.1),  # Lower epsilon for evaluation
        RandomAgent(player_id=2),
        RandomAgent(player_id=3),
    ]

    agent_names = ["RL Bot 1", "RL Bot 2", "Random Bot 1", "Random Bot 2"]

    # Initialize game
    board = standard_board()
    state = initial_game_state(board, num_players=4)

    # Reset agents
    for agent in agents:
        agent.reset()

    print(f"🎲 Board initialized with {len(state.board.tiles)} tiles")
    print(f"🎯 Starting game with {len(agents)} agents")
    print()

    step_count = 0
    max_steps = 500
    last_progress_time = time.time()

    while step_count < max_steps and state.winner is None:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            print(f"⚠️  No legal actions for {agent_names[current_player]} (Player {current_player})")
            break

        # Get action from current agent
        agent = agents[current_player]

        try:
            action = agent.select_action(state, legal_actions)
            next_state = state.apply(action)

            # Compute rewards for RL agents
            for i, agent in enumerate(agents):
                if hasattr(agent, 'compute_reward'):
                    reward = agent.compute_reward(state, next_state)
                    if hasattr(agent, 'update'):
                        agent.update(state, action, reward, next_state)

            state = next_state
            step_count += 1

            # Progress updates
            if time.time() - last_progress_time > 5.0:  # Every 5 seconds
                print(f"📊 Step {step_count}: {agent_names[current_player]} playing in {state.phase}")
                print(f"   Victory Points: {[state.players[i].victory_points for i in range(4)]}")
                last_progress_time = time.time()

        except Exception as e:
            print(f"❌ Error with {agent_names[current_player]}: {e}")
            print(f"   Action type: {action.action_type if 'action' in locals() else 'Unknown'}")
            print(f"   Phase: {state.phase}")
            print(f"   Legal actions: {len(legal_actions)}")

            # Try to continue with a safe action if possible
            if legal_actions:
                try:
                    # Find a simple action like pass_turn if available
                    safe_actions = [a for a in legal_actions if a.action_type.value == 'pass_turn']
                    if safe_actions:
                        action = safe_actions[0]
                        state = state.apply(action)
                        step_count += 1
                        print(f"   ✅ Recovered with pass_turn action")
                    else:
                        break
                except:
                    break
            else:
                break

    # Game completed
    print("\n" + "=" * 50)
    print("🏁 GAME COMPLETED!")
    print("=" * 50)

    print(f"⏱️  Total steps: {step_count}")
    print(f"🏆 Winner: {agent_names[state.winner] if state.winner is not None else 'No winner (timeout)'}")
    print()

    print("📊 Final Scores:")
    for i in range(4):
        player = state.players[i]
        print(f"   {agent_names[i]} (P{i}): {player.victory_points} VP")
        print(f"      Buildings: {len(player.settlements)} settlements, {len(player.cities)} cities, {len(player.roads)} roads")
        print(f"      Dev cards: {len(player.dev_cards)}, Knights: {player.knights_played}")
        print()

    if state.winner is not None:
        winner_type = "RL Agent" if hasattr(agents[state.winner], 'epsilon') else "Random Agent"
        print(f"🎉 {agent_names[state.winner]} ({winner_type}) wins with {state.players[state.winner].victory_points} victory points!")
    else:
        print("⏰ Game ended due to step limit")

    # Return detailed game statistics
    game_stats = {
        "winner": state.winner,
        "steps": step_count,
        "players": {}
    }

    for i in range(4):
        player = state.players[i]
        game_stats["players"][i] = {
            "name": agent_names[i],
            "agent_type": "RL" if hasattr(agents[i], 'epsilon') else "Random",
            "victory_points": player.victory_points,
            "settlements": len(player.settlements),
            "cities": len(player.cities),
            "roads": len(player.roads),
            "dev_cards": len(player.dev_cards),
            "knights_played": player.knights_played,
            "total_resources": sum(player.resources.values())
        }

    return game_stats


def main():
    """Run multiple bot matches."""
    print("🚀 CATAN RL BOT MATCHES")
    print("🤖 AI vs AI Competition")
    print()

    num_matches = 5
    match_results = []

    for match in range(num_matches):
        print(f"\n🎮 MATCH {match + 1}/{num_matches}")
        print("-" * 30)
        game_stats = run_bot_match()
        match_results.append(game_stats)

        if match < num_matches - 1:
            print("\n⏭️  Starting next match in 3 seconds...")
            time.sleep(3)

    # Final statistics
    print("\n" + "=" * 60)
    print("🏆 TOURNAMENT RESULTS")
    print("=" * 60)

    agent_names = ["RL Bot 1", "RL Bot 2", "Random Bot 1", "Random Bot 2"]
    win_counts = [0, 0, 0, 0]

    # Extract winners from match results
    winners = [result["winner"] for result in match_results]
    for winner in winners:
        if winner is not None:
            win_counts[winner] += 1

    print(f"Total matches: {num_matches}")
    print()

    # Sort by wins
    results = [(i, agent_names[i], win_counts[i]) for i in range(4)]
    results.sort(key=lambda x: x[2], reverse=True)

    for rank, (player_id, name, wins) in enumerate(results, 1):
        win_rate = wins / num_matches * 100
        agent_type = "RL Agent" if player_id < 2 else "Random Agent"
        print(f"{rank}. {name} ({agent_type}): {wins} wins ({win_rate:.1f}%)")

    rl_wins = sum(win_counts[:2])
    random_wins = sum(win_counts[2:])

    print(f"\n📈 RL Agents total: {rl_wins} wins ({rl_wins/num_matches*100:.1f}%)")
    print(f"📊 Random Agents total: {random_wins} wins ({random_wins/num_matches*100:.1f}%)")

    if rl_wins > random_wins:
        print("🧠 RL agents are learning and performing better!")
    elif random_wins > rl_wins:
        print("🎲 Random agents are still competitive!")
    else:
        print("🤝 It's a tie! Both strategies are equally matched.")

    # Detailed strategy analysis
    print_match_analysis(match_results)


def print_match_analysis(match_results):
    """Print detailed analysis of match results for strategy insights."""
    print("\n" + "=" * 60)
    print("📊 DETAILED STRATEGY ANALYSIS")
    print("=" * 60)

    agent_names = ["RL Bot 1", "RL Bot 2", "Random Bot 1", "Random Bot 2"]
    num_matches = len(match_results)

    # Aggregate statistics by agent type
    rl_stats = {"vp": [], "settlements": [], "cities": [], "roads": [], "dev_cards": [], "knights": []}
    random_stats = {"vp": [], "settlements": [], "cities": [], "roads": [], "dev_cards": [], "knights": []}

    # Collect data from all matches
    for match in match_results:
        for player_id, player_data in match["players"].items():
            stats_dict = rl_stats if player_data["agent_type"] == "RL" else random_stats
            stats_dict["vp"].append(player_data["victory_points"])
            stats_dict["settlements"].append(player_data["settlements"])
            stats_dict["cities"].append(player_data["cities"])
            stats_dict["roads"].append(player_data["roads"])
            stats_dict["dev_cards"].append(player_data["dev_cards"])
            stats_dict["knights"].append(player_data["knights_played"])

    def print_stats(name, stats):
        print(f"\n🤖 {name} Average Performance:")
        if stats["vp"]:
            print(f"   Victory Points: {sum(stats['vp'])/len(stats['vp']):.1f}")
            print(f"   Settlements: {sum(stats['settlements'])/len(stats['settlements']):.1f}")
            print(f"   Cities: {sum(stats['cities'])/len(stats['cities']):.1f}")
            print(f"   Roads: {sum(stats['roads'])/len(stats['roads']):.1f}")
            print(f"   Dev Cards: {sum(stats['dev_cards'])/len(stats['dev_cards']):.1f}")
            print(f"   Knights Played: {sum(stats['knights'])/len(stats['knights']):.1f}")
        else:
            print("   No data available")

    print_stats("RL Agents", rl_stats)
    print_stats("Random Agents", random_stats)

    # Strategy insights
    print(f"\n🧠 STRATEGY INSIGHTS:")

    if rl_stats["vp"] and random_stats["vp"]:
        rl_avg_vp = sum(rl_stats["vp"]) / len(rl_stats["vp"])
        random_avg_vp = sum(random_stats["vp"]) / len(random_stats["vp"])

        print(f"   • VP Efficiency: RL {rl_avg_vp:.1f} vs Random {random_avg_vp:.1f}")

        if rl_avg_vp > random_avg_vp:
            print("     ✅ RL agents are building more efficiently!")
        else:
            print("     ⚠️  RL agents need better building strategy")

        # Building efficiency analysis
        rl_buildings = sum(rl_stats["settlements"]) + sum(rl_stats["cities"])
        random_buildings = sum(random_stats["settlements"]) + sum(random_stats["cities"])

        if rl_buildings > 0 and random_buildings > 0:
            rl_building_rate = rl_buildings / len(rl_stats["vp"])
            random_building_rate = random_buildings / len(random_stats["vp"])
            print(f"   • Building Rate: RL {rl_building_rate:.1f} vs Random {random_building_rate:.1f}")

        # Development strategy
        rl_dev_rate = sum(rl_stats["dev_cards"]) / len(rl_stats["vp"]) if rl_stats["vp"] else 0
        random_dev_rate = sum(random_stats["dev_cards"]) / len(random_stats["vp"]) if random_stats["vp"] else 0

        print(f"   • Development Focus: RL {rl_dev_rate:.1f} vs Random {random_dev_rate:.1f}")

        if rl_dev_rate > random_dev_rate * 1.2:
            print("     📈 RL agents prefer development cards")
        elif random_dev_rate > rl_dev_rate * 1.2:
            print("     🏗️  RL agents prefer building strategy")
        else:
            print("     ⚖️  Balanced building vs development approach")

    # Game length analysis
    avg_steps = sum(match["steps"] for match in match_results) / num_matches
    print(f"\n⏱️  Average game length: {avg_steps:.0f} steps")

    if avg_steps > 200:
        print("     🐌 Games are running long - agents may be too conservative")
    elif avg_steps < 100:
        print("     🚀 Games are quick - aggressive building strategies")
    else:
        print("     ⚖️  Normal game pace")

    print()


if __name__ == "__main__":
    main()