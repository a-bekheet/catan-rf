#!/usr/bin/env python3
"""
Fast training script using Q-table agent for quick iteration.
Goal: Train an agent that consistently achieves 9-10 VP.
"""

import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from catan_rl.core.game.engine.board import standard_board
from catan_rl.core.game.engine.game_state import initial_game_state, GameState
from catan_rl.core.game.engine.rules import apply_action
from catan_rl.core.game.engine.types import Action, ActionType, ResourceType
from catan_rl.core.game.agents.random_agent import RandomAgent


class SmartAgent:
    """
    A smarter agent that uses heuristics + learning to play Catan well.

    Strategy:
    1. Prioritize building (settlements > cities > roads)
    2. Use resources efficiently
    3. Learn from experience which actions lead to VP gains
    """

    def __init__(self, player_id: int, seed: int = None):
        self.player_id = player_id
        self.rng = random.Random(seed)

        # Q-table for action type preferences based on game state
        self.action_preferences: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Learning parameters
        self.learning_rate = 0.1
        self.epsilon = 0.3  # Exploration rate

        # Track state for learning
        self.last_state_key = None
        self.last_action_type = None

    def _state_key(self, state: GameState) -> str:
        """Create a simple state representation."""
        player = state.players[self.player_id]

        # Key features that matter for decision making
        vp = player.victory_points
        settlements = len(player.settlements)
        cities = len(player.cities)
        roads = len(player.roads)
        total_resources = sum(player.resources.values())

        # Resource readiness
        can_build_settlement = (
            player.resources.get(ResourceType.BRICK, 0) >= 1 and
            player.resources.get(ResourceType.LUMBER, 0) >= 1 and
            player.resources.get(ResourceType.GRAIN, 0) >= 1 and
            player.resources.get(ResourceType.WOOL, 0) >= 1
        )
        can_build_city = (
            player.resources.get(ResourceType.ORE, 0) >= 3 and
            player.resources.get(ResourceType.GRAIN, 0) >= 2 and
            settlements > 0
        )
        can_build_road = (
            player.resources.get(ResourceType.BRICK, 0) >= 1 and
            player.resources.get(ResourceType.LUMBER, 0) >= 1
        )

        return f"vp{vp}_s{settlements}_c{cities}_r{roads}_res{min(total_resources, 10)}_cs{can_build_settlement}_cc{can_build_city}_cr{can_build_road}"

    def _score_action(self, action: Action, state: GameState) -> float:
        """Score an action based on heuristics - aggressive building strategy."""
        action_type = action.action_type
        player = state.players[self.player_id]
        vp = player.victory_points

        # Rolling dice is mandatory - highest priority
        if action_type == ActionType.ROLL_DICE:
            return 1000.0

        # Moving robber is mandatory
        if action_type == ActionType.MOVE_ROBBER:
            return 900.0

        # Discarding is mandatory
        if action_type == ActionType.DISCARD:
            return 900.0

        # Cities are very important - they give VP and double production
        # Prioritize cities when we have settlements to upgrade
        num_settlements = len(player.settlements)
        num_cities = len(player.cities)

        if action_type == ActionType.BUILD_CITY:
            if num_settlements > 0:  # Must have settlements to upgrade
                if vp >= 7:
                    return 750.0  # Very high priority when close to winning
                elif num_cities == 0 and num_settlements >= 3:
                    return 700.0  # Should get first city ASAP
                else:
                    return 650.0
            return 100.0  # Can't build city without settlements

        if action_type == ActionType.BUILD_SETTLEMENT:
            if vp >= 7:
                return 650.0  # Still good but cities better
            elif num_settlements < 3:
                return 700.0  # Early expansion is crucial
            else:
                return 550.0  # Good but might want cities

        # Development cards for VP push
        if action_type == ActionType.BUY_DEV_CARD:
            if vp >= 7:
                return 500.0  # Could be VP cards!
            elif vp >= 5:
                return 200.0
            return 80.0

        # Check if we need roads to expand
        from catan_rl.core.game.engine.rules import _settlement_distance_ok, _player_has_road_touching
        settlement_spots = 0
        for vertex_id in state.board.graph.vertices:
            if (_settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id) and
                _player_has_road_touching(state.board, player, vertex_id)):
                settlement_spots += 1

        # Roads - be careful not to over-build
        if action_type == ActionType.BUILD_ROAD:
            # DON'T build roads if we already have expansion room
            if settlement_spots >= 2:
                return 20.0  # Low priority - we have room to build settlements

            # DON'T build roads if we should focus on cities
            if len(player.settlements) >= 3 and len(player.cities) == 0:
                return 30.0  # Low priority - should build cities instead

            # Build roads only if we really need expansion
            if settlement_spots == 0 and len(player.settlements) < 5:
                return 350.0  # Need roads to expand

            # Otherwise moderate priority
            return 80.0

        # Trading - score based on what we need
        if action_type == ActionType.TRADE_BANK:
            base_score = self._score_trade(action, state, player)
            # Bonus for trading when close to winning
            if vp >= 7:
                base_score += 100.0
            return base_score

        # Development cards
        if action_type == ActionType.BUY_DEV_CARD:
            if vp >= 7:
                return 400.0  # VP cards could win!
            elif vp >= 4:
                return 150.0
            return 50.0

        if action_type == ActionType.PLAY_DEV_CARD:
            # Play knights for largest army when have enough
            if player.knights_played >= 2:
                return 300.0  # Close to largest army
            return 150.0

        # Pass turn - LOWEST priority
        if action_type == ActionType.PASS_TURN:
            return 0.1

        return 10.0

    def _score_trade(self, action: Action, state: GameState, player) -> float:
        """Score a bank trade based on what we need."""
        give = action.payload.get("give")
        receive = action.payload.get("receive")

        # What resources do we need?
        brick = player.resources.get(ResourceType.BRICK, 0)
        lumber = player.resources.get(ResourceType.LUMBER, 0)
        grain = player.resources.get(ResourceType.GRAIN, 0)
        wool = player.resources.get(ResourceType.WOOL, 0)
        ore = player.resources.get(ResourceType.ORE, 0)

        # Settlement needs: brick, lumber, grain, wool
        settlement_needs = {
            'brick': max(0, 1 - brick),
            'lumber': max(0, 1 - lumber),
            'grain': max(0, 1 - grain),
            'wool': max(0, 1 - wool),
        }

        # City needs: 3 ore, 2 grain
        city_needs = {
            'ore': max(0, 3 - ore),
            'grain': max(0, 2 - grain),
        }

        # Road needs: brick, lumber
        road_needs = {
            'brick': max(0, 1 - brick),
            'lumber': max(0, 1 - lumber),
        }

        # Score based on whether trade helps us build
        score = 50.0  # Base trade score

        # If receiving something we need for settlement
        if receive in settlement_needs and settlement_needs[receive] > 0:
            # Check if this completes settlement requirements
            remaining = sum(settlement_needs.values()) - 1
            if remaining == 0:
                score = 350.0  # This trade enables a settlement!
            else:
                score = 150.0 + (4 - remaining) * 30  # Closer = better

        # If receiving something we need for city
        if receive in city_needs and city_needs[receive] > 0:
            if len(player.settlements) > 0:  # Need settlements to upgrade
                score = max(score, 200.0)

        return score

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """Select action using heuristics + learned preferences."""
        if not legal_actions:
            raise ValueError("No legal actions")

        # Setup phase: use smart placement
        if state.phase.value == 'setup':
            return self._select_setup_action(legal_actions, state)

        # Handle mandatory actions (roll, move_robber, discard)
        mandatory_types = {ActionType.ROLL_DICE, ActionType.MOVE_ROBBER, ActionType.DISCARD}
        mandatory_actions = [a for a in legal_actions if a.action_type in mandatory_types]
        if mandatory_actions:
            action = self._select_mandatory(mandatory_actions, state)
            return self._finalize_action(action, state)

        # Exploration vs exploitation
        if self.rng.random() < self.epsilon:
            # Explore: weighted random based on heuristic scores
            scores = [self._score_action(a, state) for a in legal_actions]
            total = sum(scores)
            if total > 0:
                probs = [s / total for s in scores]
                action = self.rng.choices(legal_actions, weights=probs, k=1)[0]
            else:
                action = self.rng.choice(legal_actions)
        else:
            # Exploit: best action based on heuristics + learned preferences
            state_key = self._state_key(state)
            best_score = float('-inf')
            best_action = legal_actions[0]

            for action in legal_actions:
                score = self._score_action(action, state)
                # Add learned preference
                action_key = action.action_type.value
                score += self.action_preferences[state_key][action_key]

                if score > best_score:
                    best_score = score
                    best_action = action

            action = best_action

        return self._finalize_action(action, state)

    def _select_setup_action(self, legal_actions: List[Action], state: GameState) -> Action:
        """Select best action during setup phase - prioritize high-production vertices."""
        settlement_actions = [a for a in legal_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        road_actions = [a for a in legal_actions if a.action_type == ActionType.BUILD_ROAD]

        if settlement_actions:
            # Score each settlement location
            best_action = None
            best_score = -1

            for action in settlement_actions:
                vertex_id = action.payload.get("vertex_id")
                score = self._score_vertex(vertex_id, state)
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action or self.rng.choice(settlement_actions)

        if road_actions:
            # For roads, pick one that leads toward good settlement spots
            best_action = None
            best_score = -1

            for action in road_actions:
                edge_id = action.payload.get("edge_id")
                score = self._score_road_for_expansion(edge_id, state)
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action or self.rng.choice(road_actions)

        return self.rng.choice(legal_actions)

    def _score_vertex(self, vertex_id: int, state: GameState) -> float:
        """Score a vertex based on production value and resource diversity."""
        score = 0.0
        resources_available = set()

        # Get adjacent tiles
        for tile_id, vertices in state.board.graph.hex_to_vertices.items():
            if vertex_id in vertices:
                tile = state.board.tiles.get(tile_id)
                if tile and tile.resource.value != 'desert' and tile.number_token:
                    # Score based on dice probability
                    prob = self._get_dice_probability(tile.number_token)
                    score += prob

                    # Track resource diversity
                    resources_available.add(tile.resource.value)

        # Bonus for resource diversity
        score += len(resources_available) * 0.5

        # Bonus for having access to key resources
        key_resources = {'brick', 'lumber', 'grain', 'wool', 'ore'}
        score += len(resources_available & key_resources) * 0.3

        return score

    def _get_dice_probability(self, number: int) -> float:
        """Get probability of rolling a specific number."""
        probs = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return probs.get(number, 0) / 36.0

    def _score_road_for_expansion(self, edge_id: int, state: GameState) -> float:
        """Score a road based on expansion potential."""
        # Get the edge endpoints
        edge = state.board.graph.edges.get(edge_id)
        if not edge:
            return 0.0

        # Score based on what vertices this road leads to
        score = 0.0
        for vertex_id in [edge.vertex_a, edge.vertex_b]:
            # Check if this vertex could be a future settlement
            from catan_rl.core.game.engine.rules import _settlement_distance_ok
            if _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id):
                vertex_score = self._score_vertex(vertex_id, state)
                score = max(score, vertex_score)

        return score

    def _select_mandatory(self, actions: List[Action], state: GameState) -> Action:
        """Select from mandatory actions."""
        # For robber, try to place on tile that hurts leading opponent
        robber_actions = [a for a in actions if a.action_type == ActionType.MOVE_ROBBER]
        if robber_actions:
            return self._select_robber_placement(robber_actions, state)

        # For discard, discard excess resources we don't need
        discard_actions = [a for a in actions if a.action_type == ActionType.DISCARD]
        if discard_actions:
            return self._generate_smart_discard(state)

        return actions[0]

    def _select_robber_placement(self, actions: List[Action], state: GameState) -> Action:
        """Select best robber placement."""
        # Find the leading opponent
        my_vp = state.players[self.player_id].victory_points
        best_target = None
        best_vp = my_vp

        for pid, player in state.players.items():
            if pid != self.player_id and player.victory_points > best_vp:
                best_vp = player.victory_points
                best_target = pid

        if best_target is None:
            # No one ahead, just pick randomly
            return self.rng.choice(actions)

        # Try to place robber on tile adjacent to best_target's buildings
        target_player = state.players[best_target]
        target_vertices = target_player.settlements | target_player.cities

        best_action = None
        best_score = -1

        for action in actions:
            tile_id = action.payload.get("tile_id")
            if tile_id is None:
                continue

            # Check if this tile is adjacent to target's buildings
            vertices_on_tile = state.board.graph.hex_to_vertices.get(tile_id, [])
            overlap = len(set(vertices_on_tile) & target_vertices)

            if overlap > best_score:
                best_score = overlap
                best_action = action

        return best_action or self.rng.choice(actions)

    def _generate_smart_discard(self, state: GameState) -> Action:
        """Generate a smart discard action."""
        player = state.players[self.player_id]
        required = state.pending_discards.get(self.player_id, 0)

        if required <= 0:
            # Shouldn't happen, but handle it
            return Action(ActionType.DISCARD, {"player_id": self.player_id, "resources": {}})

        # Prioritize keeping resources needed for buildings
        # Settlement: brick, lumber, grain, wool
        # City: 3 ore, 2 grain
        # Road: brick, lumber

        available = []
        for res_type in ResourceType:
            count = player.resources.get(res_type, 0)
            for _ in range(count):
                available.append(res_type)

        if len(available) <= required:
            # Discard everything
            discard_counts = {}
            for res in available:
                discard_counts[res.value] = discard_counts.get(res.value, 0) + 1
            return Action(ActionType.DISCARD, {
                "player_id": self.player_id,
                "resources": discard_counts
            })

        # Keep the most valuable resources, discard the rest
        # Priority: grain > ore > brick = lumber > wool (roughly)
        priority = {
            ResourceType.GRAIN: 5,
            ResourceType.ORE: 4,
            ResourceType.BRICK: 3,
            ResourceType.LUMBER: 3,
            ResourceType.WOOL: 2,
        }

        # Sort by priority (ascending) so we discard low priority first
        available.sort(key=lambda r: priority.get(r, 0))

        to_discard = available[:required]
        discard_counts = {}
        for res in to_discard:
            discard_counts[res.value] = discard_counts.get(res.value, 0) + 1

        return Action(ActionType.DISCARD, {
            "player_id": self.player_id,
            "resources": discard_counts
        })

    def _finalize_action(self, action: Action, state: GameState) -> Action:
        """Finalize action and track for learning."""
        self.last_state_key = self._state_key(state)
        self.last_action_type = action.action_type.value
        return action

    def update(self, prev_state: GameState, next_state: GameState):
        """Update preferences based on VP change."""
        if self.last_state_key is None or self.last_action_type is None:
            return

        prev_vp = prev_state.players[self.player_id].victory_points
        next_vp = next_state.players[self.player_id].victory_points

        reward = (next_vp - prev_vp) * 10.0  # Reward VP gains

        # Check for win/loss
        if next_state.winner == self.player_id:
            reward += 100.0
        elif next_state.winner is not None:
            reward -= 50.0

        # Update preference
        if reward != 0:
            self.action_preferences[self.last_state_key][self.last_action_type] += (
                self.learning_rate * reward
            )

    def decay_epsilon(self, rate: float = 0.995):
        """Decay exploration rate."""
        self.epsilon = max(0.05, self.epsilon * rate)


def run_game(agents: List, max_turns: int = 2000, seed: int = None) -> Tuple[Optional[int], Dict[int, int], int]:
    """Run a single game and return (winner, final_vps, turns)."""
    board = standard_board(seed=seed)
    state = initial_game_state(board, num_players=len(agents))

    turns = 0
    while state.winner is None and turns < max_turns:
        current_player = state.current_player
        legal_actions = state.legal_actions()

        if not legal_actions:
            break

        agent = agents[current_player]
        action = agent.select_action(state, legal_actions)

        prev_state = state
        state = apply_action(state, action)

        # Update learning agents (only SmartAgent has this signature)
        if isinstance(agent, SmartAgent):
            agent.update(prev_state, state)

        turns += 1

    final_vps = {pid: state.players[pid].victory_points for pid in state.players}
    return state.winner, final_vps, turns


def train(num_episodes: int = 1000, log_interval: int = 100):
    """Train the smart agent."""
    print(f"Training SmartAgent for {num_episodes} episodes...")
    print("=" * 60)

    # Create agents - one smart agent (player 0), rest are random
    smart_agent = SmartAgent(player_id=0, seed=42)

    # Statistics
    wins = 0
    total_vp = 0
    vp_history = []
    win_history = []

    start_time = time.time()

    for episode in range(num_episodes):
        # Create opponents (random agents)
        agents = [smart_agent]
        for i in range(1, 4):
            agents.append(RandomAgent(player_id=i, seed=episode * 4 + i))

        # Run game
        winner, final_vps, turns = run_game(agents, max_turns=2000, seed=episode)

        # Track statistics
        my_vp = final_vps[0]
        total_vp += my_vp
        vp_history.append(my_vp)

        if winner == 0:
            wins += 1
            win_history.append(1)
        else:
            win_history.append(0)

        # Decay exploration
        smart_agent.decay_epsilon()

        # Log progress
        if (episode + 1) % log_interval == 0:
            recent_wins = sum(win_history[-log_interval:])
            recent_vp = sum(vp_history[-log_interval:]) / log_interval
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            print(f"Episode {episode + 1:5d} | "
                  f"Win rate: {recent_wins}/{log_interval} ({100*recent_wins/log_interval:.1f}%) | "
                  f"Avg VP: {recent_vp:.1f} | "
                  f"Epsilon: {smart_agent.epsilon:.3f} | "
                  f"{eps_per_sec:.1f} eps/s")

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Overall win rate: {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
    print(f"Overall avg VP: {total_vp/num_episodes:.2f}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (100 games)")
    print("=" * 60)

    smart_agent.epsilon = 0.0  # No exploration during evaluation
    eval_wins = 0
    eval_vps = []

    for i in range(100):
        agents = [smart_agent]
        for j in range(1, 4):
            agents.append(RandomAgent(player_id=j, seed=10000 + i * 4 + j))

        winner, final_vps, turns = run_game(agents, max_turns=2000, seed=10000 + i)
        eval_vps.append(final_vps[0])
        if winner == 0:
            eval_wins += 1

    print(f"Evaluation win rate: {eval_wins}/100 ({eval_wins}%)")
    print(f"Evaluation avg VP: {sum(eval_vps)/100:.2f}")
    print(f"VP distribution: min={min(eval_vps)}, max={max(eval_vps)}, "
          f">=9 VP: {sum(1 for v in eval_vps if v >= 9)}/100")

    return smart_agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=100)
    args = parser.parse_args()
    agent = train(num_episodes=args.episodes, log_interval=args.log_interval)
