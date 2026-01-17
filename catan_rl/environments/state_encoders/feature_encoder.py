"""Feature-based state encoder - hand-crafted strategic features."""

from typing import Tuple, Dict, Any, List
import torch
import numpy as np

from .base import BaseStateEncoder, StateEncoderFactory
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import ResourceType, DevCardType


class FeatureStateEncoder(BaseStateEncoder):
    """
    Encodes game state as hand-crafted strategic features.

    Creates a flat feature vector with interpretable strategic information
    that captures the essence of Catan gameplay.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.feature_dim = 90  # Total number of features

    def encode(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode game state as feature vector."""
        features = []

        player = game_state.players[player_id]

        # === PLAYER RESOURCES (5 features) ===
        for resource in [ResourceType.BRICK, ResourceType.LUMBER, ResourceType.ORE,
                        ResourceType.GRAIN, ResourceType.WOOL]:
            features.append(float(player.resources.get(resource, 0)))

        # === PLAYER BUILDINGS (8 features) ===
        features.extend([
            float(len(player.settlements)),
            float(len(player.cities)),
            float(len(player.roads)),
            float(player.victory_points),
            float(player.knights_played),
            float(len(player.dev_cards)),
            float(sum(player.resources.values())),  # Total resources
            float(len([r for r in player.resources.values() if r > 0]))  # Resource diversity
        ])

        # === STRATEGIC POSITION (10 features) ===
        features.extend([
            float(self._calculate_expansion_potential(game_state, player_id)),
            float(self._calculate_resource_production(game_state, player_id)),
            float(self._calculate_port_access(game_state, player_id)),
            float(self._calculate_robber_vulnerability(game_state, player_id)),
            float(self._has_longest_road(game_state, player_id)),
            float(self._has_largest_army(game_state, player_id)),
            float(self._calculate_building_efficiency(game_state, player_id)),
            float(self._calculate_development_focus(game_state, player_id)),
            float(self._calculate_endgame_position(game_state, player_id)),
            float(self._calculate_blocking_potential(game_state, player_id))
        ])

        # === RESOURCE RATIOS & READINESS (12 features) ===
        # Settlement building readiness
        settlement_resources = [
            player.resources.get(ResourceType.BRICK, 0) >= 1,
            player.resources.get(ResourceType.LUMBER, 0) >= 1,
            player.resources.get(ResourceType.GRAIN, 0) >= 1,
            player.resources.get(ResourceType.WOOL, 0) >= 1
        ]
        features.extend([float(x) for x in settlement_resources])

        # City building readiness
        city_ready = (player.resources.get(ResourceType.ORE, 0) >= 3 and
                     player.resources.get(ResourceType.GRAIN, 0) >= 2)
        features.append(float(city_ready))

        # Road building readiness
        road_ready = (player.resources.get(ResourceType.BRICK, 0) >= 1 and
                     player.resources.get(ResourceType.LUMBER, 0) >= 1)
        features.append(float(road_ready))

        # Dev card readiness
        dev_ready = (player.resources.get(ResourceType.ORE, 0) >= 1 and
                    player.resources.get(ResourceType.GRAIN, 0) >= 1 and
                    player.resources.get(ResourceType.WOOL, 0) >= 1)
        features.append(float(dev_ready))

        # Resource excess (for trading)
        for resource in [ResourceType.BRICK, ResourceType.LUMBER, ResourceType.ORE,
                        ResourceType.GRAIN, ResourceType.WOOL]:
            count = player.resources.get(resource, 0)
            features.append(float(min(count // 4, 2)))  # How many 4:1 trades possible

        # === OPPONENT ANALYSIS (20 features) ===
        opponent_features = self._encode_opponent_analysis(game_state, player_id)
        features.extend(opponent_features)

        # === BOARD STATE (15 features) ===
        board_features = self._encode_board_state(game_state, player_id)
        features.extend(board_features)

        # === GAME PHASE & CONTEXT (10 features) ===
        context_features = self._encode_game_context(game_state, player_id)
        features.extend(context_features)

        # === DEVELOPMENT CARDS (10 features) ===
        dev_features = self._encode_development_cards(game_state, player_id)
        features.extend(dev_features)

        # Convert to tensor and normalize
        feature_tensor = torch.FloatTensor(features)
        return self.to_device(self.normalize_state(feature_tensor))

    def _calculate_expansion_potential(self, state: GameState, player_id: int) -> int:
        """Calculate number of valid settlement locations."""
        # Simplified - would need proper implementation
        player = state.players[player_id]
        return max(0, 5 - len(player.settlements))  # Max settlements - current

    def _calculate_resource_production(self, state: GameState, player_id: int) -> float:
        """Calculate expected resource production per turn."""
        player = state.players[player_id]
        total_production = 0.0

        for vertex_id in player.settlements:
            # Find adjacent tiles and their productivity
            adjacent_tiles = self._get_adjacent_tiles(state, vertex_id)
            for tile_id in adjacent_tiles:
                if tile_id in state.board.tiles:
                    tile = state.board.tiles[tile_id]
                    if tile.number_token and tile_id != state.robber_tile:
                        # Probability of rolling this number
                        prob = self._get_roll_probability(tile.number_token)
                        total_production += prob

        for vertex_id in player.cities:
            adjacent_tiles = self._get_adjacent_tiles(state, vertex_id)
            for tile_id in adjacent_tiles:
                if tile_id in state.board.tiles:
                    tile = state.board.tiles[tile_id]
                    if tile.number_token and tile_id != state.robber_tile:
                        prob = self._get_roll_probability(tile.number_token)
                        total_production += prob * 2  # Cities produce 2x

        return total_production

    def _get_roll_probability(self, number: int) -> float:
        """Get probability of rolling a specific number with 2 dice."""
        probabilities = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
                        7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}
        return probabilities.get(number, 0.0)

    def _get_adjacent_tiles(self, state: GameState, vertex_id: int) -> List[int]:
        """Get tiles adjacent to a vertex using hex_to_vertices mapping."""
        tiles = []
        for tile_id, vertices in state.board.graph.hex_to_vertices.items():
            if vertex_id in vertices:
                tiles.append(tile_id)
        return tiles

    def _calculate_port_access(self, state: GameState, player_id: int) -> int:
        """Calculate number of accessible ports based on settlements/cities."""
        player = state.players[player_id]
        port_count = 0

        # Check if player has buildings at port vertices
        # Ports are typically at specific vertices - check for port markers in board
        if hasattr(state.board, 'ports'):
            for port in state.board.ports:
                if hasattr(port, 'vertices'):
                    for vertex_id in port.vertices:
                        if vertex_id in player.settlements or vertex_id in player.cities:
                            port_count += 1
                            break  # Count each port only once

        # Fallback: estimate based on coastal positions
        # Coastal vertices tend to be at the edge of the hex grid
        if port_count == 0:
            coastal_vertices = self._get_coastal_vertices(state)
            for vertex_id in player.settlements | player.cities:
                if vertex_id in coastal_vertices:
                    port_count += 1

        return min(port_count, 5)  # Cap at 5 ports

    def _get_coastal_vertices(self, state: GameState) -> set:
        """Get vertices at the edge of the board (potential port locations)."""
        # Vertices that are adjacent to fewer than 3 tiles are coastal
        coastal = set()
        for vertex_id in state.board.graph.vertices:
            adjacent_tiles = self._get_adjacent_tiles(state, vertex_id)
            if len(adjacent_tiles) < 3:  # Edge vertices touch fewer tiles
                coastal.add(vertex_id)
        return coastal

    def _calculate_robber_vulnerability(self, state: GameState, player_id: int) -> float:
        """Calculate vulnerability to robber."""
        player = state.players[player_id]
        total_cards = sum(player.resources.values())
        return float(total_cards > 7)  # Vulnerable to discard

    def _has_longest_road(self, state: GameState, player_id: int) -> bool:
        """Check if player has longest road."""
        # Simplified - would need proper road calculation
        return False

    def _has_largest_army(self, state: GameState, player_id: int) -> bool:
        """Check if player has largest army."""
        player = state.players[player_id]
        if player.knights_played < 3:
            return False

        # Check if this player has most knights
        max_knights = max(p.knights_played for p in state.players.values())
        return player.knights_played == max_knights and player.knights_played >= 3

    def _calculate_building_efficiency(self, state: GameState, player_id: int) -> float:
        """Calculate VP per building ratio."""
        player = state.players[player_id]
        total_buildings = len(player.settlements) + len(player.cities) + len(player.roads)
        if total_buildings == 0:
            return 0.0
        return player.victory_points / total_buildings

    def _calculate_development_focus(self, state: GameState, player_id: int) -> float:
        """Calculate focus on development vs buildings."""
        player = state.players[player_id]
        dev_points = len(player.dev_cards) + player.knights_played
        building_points = len(player.settlements) + len(player.cities)
        total = dev_points + building_points
        if total == 0:
            return 0.5
        return dev_points / total

    def _calculate_endgame_position(self, state: GameState, player_id: int) -> float:
        """Calculate how close to victory."""
        player = state.players[player_id]
        return min(1.0, player.victory_points / 10.0)

    def _calculate_blocking_potential(self, state: GameState, player_id: int) -> float:
        """Calculate ability to block opponents."""
        # Simplified implementation
        return 0.0

    def _encode_opponent_analysis(self, state: GameState, player_id: int) -> List[float]:
        """Encode information about opponents."""
        features = []

        # Find leading opponent
        max_vp = 0
        for pid, player in state.players.items():
            if pid != player_id and player.victory_points > max_vp:
                max_vp = player.victory_points

        features.append(float(max_vp))  # Leading opponent VP

        # Average opponent metrics
        opponent_settlements = []
        opponent_cities = []
        opponent_resources = []
        opponent_dev_cards = []

        for pid, player in state.players.items():
            if pid != player_id:
                opponent_settlements.append(len(player.settlements))
                opponent_cities.append(len(player.cities))
                opponent_resources.append(sum(player.resources.values()))
                opponent_dev_cards.append(len(player.dev_cards))

        if opponent_settlements:
            features.extend([
                float(np.mean(opponent_settlements)),
                float(np.mean(opponent_cities)),
                float(np.mean(opponent_resources)),
                float(np.mean(opponent_dev_cards)),
                float(max(opponent_settlements)),  # Most threatening opponent buildings
                float(max(opponent_cities)),
                float(max(opponent_resources)),
                float(max(opponent_dev_cards))
            ])
        else:
            features.extend([0.0] * 8)

        # Relative position
        my_vp = state.players[player_id].victory_points
        features.extend([
            float(my_vp >= max_vp),  # Am I leading?
            float(max_vp - my_vp),   # VP gap
            float(len([p for pid, p in state.players.items() if pid != player_id and p.victory_points >= 8]))  # Opponents close to win
        ])

        # Fill to 20 features
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _encode_board_state(self, state: GameState, player_id: int) -> List[float]:
        """Encode board-level information."""
        features = []

        # Turn information
        features.append(float(state.turn_index / 100.0))  # Normalized turn count
        features.append(float(state.current_player == player_id))  # Is my turn?

        # Phase information
        phase_encoding = {
            'setup': 0.0, 'roll': 0.25, 'main': 0.5, 'discard': 0.75, 'move_robber': 1.0
        }
        features.append(phase_encoding.get(state.phase.value, 0.0))

        # Resource scarcity (bank status)
        total_bank_resources = sum(state.bank.values())
        features.append(float(total_bank_resources / 95.0))  # Normalized by starting resources

        # Development cards remaining
        features.append(float(len(state.dev_deck) / 25.0))  # Normalized

        # Robber effects
        features.append(float(state.robber_tile is not None))

        # Fill to 15 features
        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _encode_game_context(self, state: GameState, player_id: int) -> List[float]:
        """Encode game context and phase information."""
        features = []

        # Game progress indicators
        total_vp = sum(p.victory_points for p in state.players.values())
        features.append(float(total_vp / 40.0))  # Progress toward game end

        total_buildings = sum(len(p.settlements) + len(p.cities) + len(p.roads)
                            for p in state.players.values())
        features.append(float(total_buildings / 60.0))  # Building density

        # Endgame proximity
        highest_vp = max(p.victory_points for p in state.players.values())
        features.append(float(highest_vp / 10.0))
        features.append(float(highest_vp >= 8))  # Someone close to winning

        # Resource economy
        total_player_resources = sum(sum(p.resources.values()) for p in state.players.values())
        features.append(float(total_player_resources / 100.0))

        # Development focus of game
        total_dev_cards = sum(len(p.dev_cards) + p.knights_played for p in state.players.values())
        features.append(float(total_dev_cards / 25.0))

        # Fill to 10 features
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def _encode_development_cards(self, state: GameState, player_id: int) -> List[float]:
        """Encode development card information."""
        player = state.players[player_id]
        features = []

        # Current dev cards by type
        dev_counts = {DevCardType.KNIGHT: 0, DevCardType.MONOPOLY: 0,
                     DevCardType.YEAR_OF_PLENTY: 0, DevCardType.ROAD_BUILDING: 0,
                     DevCardType.VICTORY_POINT: 0}

        for card in player.dev_cards:
            if card in dev_counts:
                dev_counts[card] += 1

        features.extend([float(count) for count in dev_counts.values()])

        # Playable this turn (not bought this turn)
        new_cards = state.new_dev_cards.get(player_id, [])
        playable_cards = [card for card in player.dev_cards if card not in new_cards]
        features.append(float(len(playable_cards)))

        # Knights played
        features.append(float(player.knights_played))

        # Can play knight this turn
        can_play_knight = (DevCardType.KNIGHT in playable_cards and
                          not state.played_dev_card_this_turn)
        features.append(float(can_play_knight))

        # Development strategy indicator
        dev_investment = len(player.dev_cards) + player.knights_played
        building_investment = len(player.settlements) + len(player.cities)
        if dev_investment + building_investment > 0:
            dev_ratio = dev_investment / (dev_investment + building_investment)
        else:
            dev_ratio = 0.0
        features.append(dev_ratio)

        # Victory points from hidden dev cards
        vp_cards = [card for card in player.dev_cards if card == DevCardType.VICTORY_POINT]
        features.append(float(len(vp_cards)))

        return features

    def get_state_shape(self) -> Tuple[int]:
        """Get shape of encoded state."""
        return (self.feature_dim,)

    def get_feature_names(self) -> List[str]:
        """Get feature descriptions for analysis."""
        names = []

        # Resources (5)
        names.extend([f"Resource_{res.value}" for res in
                     [ResourceType.BRICK, ResourceType.LUMBER, ResourceType.ORE,
                      ResourceType.GRAIN, ResourceType.WOOL]])

        # Buildings & basic stats (8)
        names.extend(["Settlements", "Cities", "Roads", "Victory_Points",
                     "Knights_Played", "Dev_Cards", "Total_Resources", "Resource_Diversity"])

        # Strategic position (10)
        names.extend(["Expansion_Potential", "Resource_Production", "Port_Access",
                     "Robber_Vulnerability", "Has_Longest_Road", "Has_Largest_Army",
                     "Building_Efficiency", "Development_Focus", "Endgame_Position",
                     "Blocking_Potential"])

        # Continue with other feature categories...
        # This would be expanded to cover all 85 features

        return names

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize feature values for stable training."""
        # Most features are already normalized to [0, 1] or small integer ranges
        # Could add standardization if needed
        return torch.clamp(state, 0.0, 10.0)  # Clamp to reasonable range


# Register with factory
StateEncoderFactory.register("feature", FeatureStateEncoder)