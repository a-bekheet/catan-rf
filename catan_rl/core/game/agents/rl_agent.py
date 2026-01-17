"""Reinforcement Learning agent using Q-learning."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from catan_rl.core.game.engine.game_state import GameState, TurnPhase
from catan_rl.core.game.engine.types import Action, ActionType, ResourceType


class RLAgent:
    """A Q-learning agent for Catan."""

    def __init__(
        self,
        player_id: int,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        discount: float = 0.95,
        seed: int | None = None,
        model_path: str | None = None
    ):
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        self.rng = random.Random(seed)
        self.model_path = model_path or f"models/rl_agent_p{player_id}.json"

        # Q-table: state_key -> {action_key: q_value}
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Experience tracking
        self.last_state: GameState | None = None
        self.last_action: Action | None = None

        # Load existing model if available
        self.load_model()

    def _state_key(self, state: GameState) -> str:
        """Create a comprehensive state representation for strategic learning."""
        player = state.players[self.player_id]

        # Basic player state
        my_vp = player.victory_points
        my_settlements = len(player.settlements)
        my_cities = len(player.cities)
        my_roads = len(player.roads)
        my_knights = player.knights_played
        my_dev_cards = len(player.dev_cards)
        my_resources = sum(player.resources.values())

        # Resource diversity and specific resources for building
        resource_types = sum(1 for count in player.resources.values() if count > 0)

        # Settlement building readiness (key strategic insight)
        from catan_rl.core.game.engine.types import ResourceType
        has_brick = player.resources.get(ResourceType.BRICK, 0) > 0
        has_lumber = player.resources.get(ResourceType.LUMBER, 0) > 0
        has_grain = player.resources.get(ResourceType.GRAIN, 0) > 0
        has_wool = player.resources.get(ResourceType.WOOL, 0) > 0
        settlement_resources = sum([has_brick, has_lumber, has_grain, has_wool])

        # City building readiness
        has_ore = player.resources.get(ResourceType.ORE, 0) >= 3
        has_grain_for_city = player.resources.get(ResourceType.GRAIN, 0) >= 2
        city_ready = has_ore and has_grain_for_city and my_settlements > 0

        # Road building capability
        road_resources = has_brick and has_lumber

        # Settlement expansion potential (can build where roads lead)
        settlement_spots = self._count_valid_settlement_spots(state, player)

        # Relative position vs opponents
        max_opponent_vp = 0
        total_opponent_knights = 0
        for pid, other_player in state.players.items():
            if pid != self.player_id:
                max_opponent_vp = max(max_opponent_vp, other_player.victory_points)
                total_opponent_knights += other_player.knights_played

        vp_position = "leading" if my_vp > max_opponent_vp else "tied" if my_vp == max_opponent_vp else "behind"

        # Strategic buckets for key metrics
        vp_bucket = min(my_vp // 2, 5)  # 0-1, 2-3, 4-5, 6-7, 8-9, 10+
        resource_bucket = min(my_resources // 3, 5)  # 0-2, 3-5, 6-8, 9-11, 12-14, 15+
        building_bucket = min((my_settlements + my_cities * 2) // 2, 5)
        development_bucket = min((my_knights + my_dev_cards) // 2, 3)

        # Game phase and turn context
        phase = state.phase.value
        is_my_turn = state.current_player == self.player_id

        return (
            f"vp_pos:{vp_position}_vp_b:{vp_bucket}_"
            f"res_types:{resource_types}_res_b:{resource_bucket}_"
            f"build_b:{building_bucket}_dev_b:{development_bucket}_"
            f"sett_res:{settlement_resources}_city_ready:{city_ready}_road_res:{road_resources}_"
            f"sett_spots:{min(settlement_spots, 5)}_"
            f"knights:{min(my_knights, 5)}_phase:{phase}_"
            f"my_turn:{is_my_turn}_max_opp_vp:{min(max_opponent_vp, 10)}"
        )

    def _count_valid_settlement_spots(self, state: GameState, player: PlayerState) -> int:
        """Count valid settlement locations connected to player's roads."""
        from catan_rl.core.game.engine.rules import _settlement_distance_ok, _player_has_road_touching

        valid_spots = 0
        for vertex_id in state.board.graph.vertices:
            if (_settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id) and
                _player_has_road_touching(state.board, player, vertex_id)):
                valid_spots += 1
        return valid_spots

    def _action_key(self, action: Action) -> str:
        """Create action representation for Q-table."""
        return f"{action.action_type.value}:{hash(str(action.payload)) % 10000}"

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """Select action using epsilon-greedy policy."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        state_key = self._state_key(state)

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.choice(legal_actions)
        else:
            # Exploit: best Q-value action
            best_action = None
            best_q_value = float('-inf')

            for action in legal_actions:
                action_key = self._action_key(action)
                q_value = self.q_table[state_key][action_key]

                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            action = best_action or self.rng.choice(legal_actions)

        # Handle discard actions by generating valid resource combinations
        if action.action_type == ActionType.DISCARD and state.phase == TurnPhase.DISCARD:
            action = self._generate_discard_action(state, action)

        # Store for learning
        self.last_state = state
        self.last_action = action

        return action

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState):
        """Update Q-values using Q-learning."""
        if self.last_state is None or self.last_action is None:
            return

        # Q-learning update
        state_key = self._state_key(self.last_state)
        action_key = self._action_key(self.last_action)
        next_state_key = self._state_key(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action_key]

        # Max Q-value for next state
        next_legal_actions = next_state.legal_actions()
        if next_legal_actions:
            max_next_q = max(
                self.q_table[next_state_key][self._action_key(a)]
                for a in next_legal_actions
            )
        else:
            max_next_q = 0.0

        # Q-learning update rule
        target = reward + self.discount * max_next_q
        self.q_table[state_key][action_key] += self.learning_rate * (target - current_q)

    def compute_reward(self, state: GameState, next_state: GameState, truncated: bool = False) -> float:
        """
        Compute reward for RL training - simplified and focused on key signals.

        Key principles:
        1. Strong win/loss signal (sparse but clear)
        2. VP progress (dense, interpretable)
        3. Building progress (shaped reward for structure)
        4. Endgame acceleration (bonus when close to winning)
        5. Truncation penalty (discourage timeout)
        """
        reward = 0.0
        player = next_state.players[self.player_id]
        prev_player = state.players[self.player_id]

        # === TERMINAL REWARDS (Most Important) ===
        if next_state.winner == self.player_id:
            reward += 100.0  # WIN!
        elif next_state.winner is not None:
            reward -= 50.0   # LOSS
        elif truncated:
            # Truncation penalty - encourages faster play
            # Scale by how close we were to winning
            my_vp = player.victory_points
            opponent_vps = [
                p.victory_points for pid, p in next_state.players.items()
                if pid != self.player_id
            ]
            max_opponent_vp = max(opponent_vps) if opponent_vps else 0
            # If we're ahead, smaller penalty. If behind, larger penalty.
            if my_vp >= max_opponent_vp:
                reward -= 10.0  # We were winning but timed out
            else:
                reward -= 25.0  # We were losing and timed out

        # === VICTORY POINT PROGRESS (Key Dense Reward) ===
        vp_gain = player.victory_points - prev_player.victory_points
        reward += vp_gain * 20.0  # Strong VP incentive

        # === ENDGAME ACCELERATION ===
        # Bonus for getting close to winning - encourages finishing games
        if player.victory_points >= 8 and prev_player.victory_points < 8:
            reward += 15.0  # First time reaching 8+ VP
        elif player.victory_points >= 9 and prev_player.victory_points < 9:
            reward += 20.0  # First time reaching 9+ VP

        # === BUILDING PROGRESS (Shaped Reward) ===
        new_settlements = len(player.settlements) - len(prev_player.settlements)
        new_cities = len(player.cities) - len(prev_player.cities)
        new_roads = len(player.roads) - len(prev_player.roads)

        reward += new_settlements * 5.0  # Settlements = VP + production
        reward += new_cities * 8.0       # Cities = VP + 2x production
        reward += new_roads * 1.0        # Roads enable expansion

        # Bonus for road expansion that opens settlement spots
        if new_roads > 0:
            prev_spots = self._count_valid_settlement_spots(state, prev_player)
            curr_spots = self._count_valid_settlement_spots(next_state, player)
            if curr_spots > prev_spots:
                reward += (curr_spots - prev_spots) * 3.0

        # === DEVELOPMENT CARD PROGRESS ===
        new_knights = player.knights_played - prev_player.knights_played
        reward += new_knights * 2.0  # Knights help with robber + largest army

        return reward

    def _building_rewards(self, state: GameState, next_state: GameState) -> float:
        """Reward building with strategic value consideration."""
        reward = 0.0
        player = next_state.players[self.player_id]
        prev_player = state.players[self.player_id]

        new_settlements = len(player.settlements) - len(prev_player.settlements)
        new_cities = len(player.cities) - len(prev_player.cities)
        new_roads = len(player.roads) - len(prev_player.roads)

        # Base building rewards
        reward += new_settlements * 8.0  # High reward for settlements (key to winning)
        reward += new_cities * 12.0     # Even higher for cities
        reward += new_roads * 2.0       # Increased road reward

        # Strategic road expansion rewards
        if new_roads > 0:
            # Check if new roads opened up settlement opportunities
            prev_spots = self._count_valid_settlement_spots(state, prev_player)
            curr_spots = self._count_valid_settlement_spots(next_state, player)

            if curr_spots > prev_spots:
                # Big reward for opening new settlement locations
                reward += (curr_spots - prev_spots) * 8.0
            else:
                # Still reward road building even if it doesn't immediately open spots
                # (might be building toward a strategic position)
                reward += 1.0

        # Penalty for having no expansion options
        curr_spots = self._count_valid_settlement_spots(next_state, player)
        if curr_spots == 0 and len(player.settlements) < 5:
            # Critical penalty for being unable to expand
            reward -= 3.0

        # Settlement expansion bonus
        if new_settlements > 0:
            # Big bonus if we're progressing toward victory
            if len(player.settlements) >= 3:  # 3rd settlement = 4+ VP
                reward += 5.0
            if len(player.settlements) >= 4:  # 4th settlement = 5+ VP
                reward += 8.0

        # City upgrade bonus
        if new_cities > 0:
            # Cities are critical for winning (2 VP each)
            reward += 10.0

        # Bonus for building on high-yield spots
        for settlement_id in player.settlements:
            if settlement_id not in prev_player.settlements:
                reward += self._calculate_position_value(settlement_id, next_state) * 2.0

        return reward

    def _resource_rewards(self, state: GameState, next_state: GameState) -> float:
        """Reward effective resource management for building."""
        reward = 0.0
        player = next_state.players[self.player_id]
        prev_player = state.players[self.player_id]

        from catan_rl.core.game.engine.types import ResourceType

        # Strategic resource collection rewards
        prev_settlement_res = sum([
            prev_player.resources.get(ResourceType.BRICK, 0) > 0,
            prev_player.resources.get(ResourceType.LUMBER, 0) > 0,
            prev_player.resources.get(ResourceType.GRAIN, 0) > 0,
            prev_player.resources.get(ResourceType.WOOL, 0) > 0
        ])

        curr_settlement_res = sum([
            player.resources.get(ResourceType.BRICK, 0) > 0,
            player.resources.get(ResourceType.LUMBER, 0) > 0,
            player.resources.get(ResourceType.GRAIN, 0) > 0,
            player.resources.get(ResourceType.WOOL, 0) > 0
        ])

        # Bonus for completing settlement resource set
        if curr_settlement_res > prev_settlement_res:
            reward += 2.0 * (curr_settlement_res - prev_settlement_res)

        # Special bonus for getting all 4 settlement resources
        if curr_settlement_res == 4 and prev_settlement_res < 4:
            reward += 5.0  # Ready to build settlement!

        # City resource rewards (ore + grain)
        prev_city_ready = (prev_player.resources.get(ResourceType.ORE, 0) >= 3 and
                          prev_player.resources.get(ResourceType.GRAIN, 0) >= 2)
        curr_city_ready = (player.resources.get(ResourceType.ORE, 0) >= 3 and
                          player.resources.get(ResourceType.GRAIN, 0) >= 2)

        if curr_city_ready and not prev_city_ready and len(player.settlements) > 0:
            reward += 3.0  # Ready to build city!

        # Road building resource rewards
        prev_road_ready = (prev_player.resources.get(ResourceType.BRICK, 0) > 0 and
                          prev_player.resources.get(ResourceType.LUMBER, 0) > 0)
        curr_road_ready = (player.resources.get(ResourceType.BRICK, 0) > 0 and
                          player.resources.get(ResourceType.LUMBER, 0) > 0)

        if curr_road_ready and not prev_road_ready:
            reward += 1.5  # Can build roads to expand

        # Extra incentive for road resources when stuck
        curr_spots = self._count_valid_settlement_spots(next_state, player)
        if curr_spots == 0 and len(player.settlements) < 5:
            # If we have no settlement spots, strongly incentivize road building resources
            if curr_road_ready:
                reward += 3.0  # Critical to get unstuck!
            else:
                # Reward getting closer to road resources
                brick_progress = player.resources.get(ResourceType.BRICK, 0) > 0
                lumber_progress = player.resources.get(ResourceType.LUMBER, 0) > 0
                if brick_progress and not prev_player.resources.get(ResourceType.BRICK, 0) > 0:
                    reward += 1.0
                if lumber_progress and not prev_player.resources.get(ResourceType.LUMBER, 0) > 0:
                    reward += 1.0

        # Penalty for passing turn when we could build roads to expand
        if curr_spots == 0 and curr_road_ready and len(player.settlements) < 5:
            # We have road resources but no settlement spots - should build roads!
            reward += 2.0  # Encourage road building when expansion is needed

        # Resource gain reward
        prev_total = sum(prev_player.resources.values())
        curr_total = sum(player.resources.values())
        resource_gain = curr_total - prev_total
        reward += resource_gain * 0.5

        # Reward resource diversity (having all types)
        unique_resources = sum(1 for count in player.resources.values() if count > 0)
        reward += unique_resources * 0.3

        # Penalty for hoarding too many cards (vulnerable to robber)
        if curr_total > 7:
            reward -= (curr_total - 7) * 0.5

        # Reward efficient resource conversion (spending resources)
        if resource_gain < 0:  # Spent resources
            reward += abs(resource_gain) * 0.2  # Small reward for spending

        return reward

    def _development_rewards(self, state: GameState, next_state: GameState) -> float:
        """Reward development card strategy."""
        reward = 0.0
        player = next_state.players[self.player_id]
        prev_player = state.players[self.player_id]

        # Development card acquisition
        new_dev_cards = len(player.dev_cards) - len(prev_player.dev_cards)
        reward += new_dev_cards * 1.5

        # Knight usage (good for largest army and robber control)
        new_knights = player.knights_played - prev_player.knights_played
        reward += new_knights * 2.0

        # Progress toward largest army
        if player.knights_played >= 3:
            army_leader = self._get_largest_army_leader(next_state)
            if army_leader == self.player_id:
                reward += 3.0  # Bonus for having largest army
            elif army_leader is not None:
                # Bonus for being close to taking largest army
                leader_knights = next_state.players[army_leader].knights_played
                if player.knights_played >= leader_knights - 1:
                    reward += 1.0

        return reward

    def _strategic_interaction_rewards(self, state: GameState, next_state: GameState) -> float:
        """Reward strategic interaction with opponents."""
        reward = 0.0

        # Reward effective robber usage
        if next_state.robber_tile != state.robber_tile:
            # Robber was moved - reward if it targets the leader or productive tiles
            leader_id = self._get_vp_leader(next_state)
            if leader_id is not None and leader_id != self.player_id:
                # Check if robber affects the leader
                if self._robber_affects_player(next_state.robber_tile, leader_id, next_state):
                    reward += 2.0

        # Reward blocking opponent expansion
        reward += self._calculate_blocking_value(state, next_state)

        return reward

    def _position_control_rewards(self, state: GameState, next_state: GameState) -> float:
        """Reward controlling strategic positions."""
        reward = 0.0
        player = next_state.players[self.player_id]

        # Reward for controlling ports
        port_bonus = self._calculate_port_control_value(player, next_state)
        reward += port_bonus

        # Reward for progress toward longest road
        road_length = self._calculate_longest_road_progress(player, next_state)
        if road_length >= 5:  # Minimum for longest road
            longest_road_holder = self._get_longest_road_holder(next_state)
            if longest_road_holder == self.player_id:
                reward += 5.0  # Bonus for having longest road
            else:
                reward += road_length * 0.3  # Progress bonus

        return reward

    def _calculate_position_value(self, vertex_id: int, state: GameState) -> float:
        """Calculate the strategic value of a building position."""
        # Count adjacent tiles that produce resources
        resource_value = 0.0
        adjacent_tiles = 0

        # Check which tiles are adjacent to this vertex
        for tile_id, vertex_list in state.board.graph.hex_to_vertices.items():
            if vertex_id in vertex_list:
                adjacent_tiles += 1
                tile = state.board.tiles.get(tile_id)
                if tile and tile.resource != ResourceType.DESERT and tile.number_token:
                    # Give bonus for productive tiles, weighted by probability
                    probability = {6: 5, 8: 5, 5: 4, 9: 4, 4: 3, 10: 3, 3: 2, 11: 2, 2: 1, 12: 1}.get(tile.number_token, 0)
                    resource_value += probability * 0.1

        # Bonus for being adjacent to multiple tiles (more diverse resource access)
        position_bonus = min(adjacent_tiles * 0.3, 1.0)

        return resource_value + position_bonus

    def _get_vp_leader(self, state: GameState) -> int | None:
        """Get the player with the most victory points."""
        max_vp = 0
        leader = None
        for pid, player in state.players.items():
            if player.victory_points > max_vp:
                max_vp = player.victory_points
                leader = pid
        return leader

    def _get_largest_army_leader(self, state: GameState) -> int | None:
        """Get the player with the largest army."""
        max_knights = 2  # Need at least 3 for largest army
        leader = None
        for pid, player in state.players.items():
            if player.knights_played > max_knights:
                max_knights = player.knights_played
                leader = pid
        return leader

    def _robber_affects_player(self, robber_tile: int, player_id: int, state: GameState) -> bool:
        """Check if robber placement affects a specific player."""
        player = state.players[player_id]

        # Get vertices adjacent to the robber tile
        vertices_on_tile = state.board.graph.hex_to_vertices.get(robber_tile, [])

        # Check if player has buildings adjacent to this tile
        for vertex_id in vertices_on_tile:
            if vertex_id in player.settlements or vertex_id in player.cities:
                return True
        return False

    def _calculate_blocking_value(self, state: GameState, next_state: GameState) -> float:
        """Calculate reward for blocking opponent expansion."""
        # Simplified blocking detection - reward for building near opponents
        player = next_state.players[self.player_id]
        prev_player = state.players[self.player_id]

        new_buildings = (set(player.settlements) | set(player.cities)) - \
                       (set(prev_player.settlements) | set(prev_player.cities))

        blocking_value = 0.0
        for building_vertex in new_buildings:
            # Check if this building blocks opponent expansion
            # (This is a simplified check - real implementation would be more complex)
            for pid, other_player in next_state.players.items():
                if pid != self.player_id:
                    for other_vertex in other_player.settlements | other_player.cities:
                        # If we built within 2 edges of an opponent, give small blocking bonus
                        if abs(building_vertex - other_vertex) <= 2:  # Simplified distance
                            blocking_value += 0.5

        return min(blocking_value, 2.0)  # Cap blocking bonus

    def _calculate_port_control_value(self, player, state: GameState) -> float:
        """Calculate value from controlling ports."""
        # Simplified port value - would need actual port data structure
        # For now, just give small bonus for coastal settlements
        port_value = 0.0
        for vertex_id in player.settlements | player.cities:
            # Simplified: assume some vertices are "coastal" (this would need real port data)
            if vertex_id % 7 == 0:  # Arbitrary "port" detection
                port_value += 1.0
        return min(port_value, 3.0)

    def _calculate_longest_road_progress(self, player, state: GameState) -> int:
        """Calculate the longest road length for a player."""
        # Simplified road length calculation
        # Real implementation would need graph traversal
        return len(player.roads)  # Simplified - just count roads

    def _get_longest_road_holder(self, state: GameState) -> int | None:
        """Get the player with the longest road."""
        max_roads = 4  # Need at least 5 for longest road
        leader = None
        for pid, player in state.players.items():
            road_length = self._calculate_longest_road_progress(player, state)
            if road_length > max_roads:
                max_roads = road_length
                leader = pid
        return leader

    def _generate_discard_action(self, state: GameState, template_action: Action) -> Action:
        """Generate a valid discard action with proper resource allocation."""
        # Use state.current_player instead of self.player_id since agents
        # may be placed at different positions in different games
        current_player_id = state.current_player
        player = state.players[current_player_id]
        required = state.pending_discards.get(current_player_id, 0)

        if required <= 0:
            return template_action

        # Get available resources
        available = []
        for resource_type in ResourceType:
            count = player.resources.get(resource_type, 0)
            for _ in range(count):
                available.append(resource_type)

        if len(available) < required:
            # Not enough resources, discard what we have
            required = len(available)

        # Randomly select resources to discard
        to_discard = self.rng.sample(available, required)

        # Count resources to discard
        discard_counts = {}
        for resource in to_discard:
            discard_counts[resource.value] = discard_counts.get(resource.value, 0) + 1

        # Create new action with proper resources
        return Action(
            action_type=ActionType.DISCARD,
            payload={
                "player_id": current_player_id,
                "resources": discard_counts
            }
        )

    def reset(self):
        """Reset agent state for new episode."""
        self.last_state = None
        self.last_action = None

    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate."""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

    def save_model(self):
        """Save Q-table and parameters to file."""
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Convert defaultdict to regular dict for JSON serialization
        q_table_serializable = {}
        for state_key, actions in self.q_table.items():
            q_table_serializable[state_key] = dict(actions)

        model_data = {
            "player_id": self.player_id,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "discount": self.discount,
            "q_table": q_table_serializable
        }

        with open(self.model_path, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self):
        """Load Q-table and parameters from file."""
        if not Path(self.model_path).exists():
            return

        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)

            # Restore Q-table
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state_key, actions in model_data.get("q_table", {}).items():
                for action_key, q_value in actions.items():
                    self.q_table[state_key][action_key] = q_value

            # Restore parameters (optional - can override with constructor values)
            if "epsilon" in model_data:
                self.epsilon = model_data["epsilon"]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load model from {self.model_path}: {e}")
            print("Starting with fresh Q-table")