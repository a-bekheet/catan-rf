"""Spatial state encoder - represents board as image-like tensor."""

from typing import Tuple, Dict, Any
import torch
import numpy as np

from .base import BaseStateEncoder, StateEncoderFactory
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import ResourceType


class SpatialStateEncoder(BaseStateEncoder):
    """
    Encodes Catan board as spatial tensor suitable for CNNs.

    Creates multi-channel representation where each channel represents
    different aspects of the game state (resources, buildings, etc.).
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}
        self.board_size = config.get('board_size', (7, 7))  # Hex board as square grid
        self.num_channels = 12  # See _get_channel_info for details

    def encode(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode game state as spatial tensor."""
        batch_size = 1  # Single state
        channels, height, width = self.get_state_shape()

        # Initialize empty tensor
        state_tensor = torch.zeros((batch_size, channels, height, width))

        # Channel 0-4: Resource types (brick=1, lumber=2, ore=3, grain=4, wool=5)
        self._encode_resource_types(state_tensor, game_state, 0)

        # Channel 5: Number tokens (normalized 2-12 -> 0-1)
        self._encode_number_tokens(state_tensor, game_state, 5)

        # Channel 6: Robber position
        self._encode_robber(state_tensor, game_state, 6)

        # Channel 7: Current player settlements
        self._encode_player_settlements(state_tensor, game_state, player_id, 7)

        # Channel 8: Current player cities
        self._encode_player_cities(state_tensor, game_state, player_id, 8)

        # Channel 9: Current player roads
        self._encode_player_roads(state_tensor, game_state, player_id, 9)

        # Channel 10: All opponent buildings
        self._encode_opponent_buildings(state_tensor, game_state, player_id, 10)

        # Channel 11: Valid placements for current player
        self._encode_valid_placements(state_tensor, game_state, player_id, 11)

        return self.to_device(state_tensor.squeeze(0))  # Remove batch dimension for proper stacking

    def _encode_resource_types(self, tensor: torch.Tensor, state: GameState, channel: int):
        """Encode resource types on tiles."""
        for tile_id, tile in state.board.tiles.items():
            pos = self._tile_to_position(tile_id)
            if pos is not None:
                row, col = pos
                if tile.resource == ResourceType.BRICK:
                    tensor[0, channel, row, col] = 1.0
                elif tile.resource == ResourceType.LUMBER:
                    tensor[0, channel, row, col] = 2.0
                elif tile.resource == ResourceType.ORE:
                    tensor[0, channel, row, col] = 3.0
                elif tile.resource == ResourceType.GRAIN:
                    tensor[0, channel, row, col] = 4.0
                elif tile.resource == ResourceType.WOOL:
                    tensor[0, channel, row, col] = 5.0
                # Desert = 0 (default)

    def _encode_number_tokens(self, tensor: torch.Tensor, state: GameState, channel: int):
        """Encode number tokens (2-12) normalized to [0, 1]."""
        for tile_id, tile in state.board.tiles.items():
            pos = self._tile_to_position(tile_id)
            if pos is not None and tile.number_token:
                row, col = pos
                # Normalize 2-12 to 0-1 range
                normalized_number = (tile.number_token - 2) / 10.0
                tensor[0, channel, row, col] = normalized_number

    def _encode_robber(self, tensor: torch.Tensor, state: GameState, channel: int):
        """Encode robber position."""
        robber_tile = state.robber_tile
        if robber_tile is not None:
            pos = self._tile_to_position(robber_tile)
            if pos is not None:
                row, col = pos
                tensor[0, channel, row, col] = 1.0

    def _encode_player_settlements(self, tensor: torch.Tensor, state: GameState, player_id: int, channel: int):
        """Encode current player's settlements."""
        player = state.players[player_id]
        for settlement_vertex in player.settlements:
            pos = self._vertex_to_position(settlement_vertex)
            if pos is not None:
                row, col = pos
                tensor[0, channel, row, col] = 1.0

    def _encode_player_cities(self, tensor: torch.Tensor, state: GameState, player_id: int, channel: int):
        """Encode current player's cities."""
        player = state.players[player_id]
        for city_vertex in player.cities:
            pos = self._vertex_to_position(city_vertex)
            if pos is not None:
                row, col = pos
                tensor[0, channel, row, col] = 1.0

    def _encode_player_roads(self, tensor: torch.Tensor, state: GameState, player_id: int, channel: int):
        """Encode current player's roads."""
        player = state.players[player_id]
        for road_edge in player.roads:
            pos = self._edge_to_position(road_edge)
            if pos is not None:
                row, col = pos
                tensor[0, channel, row, col] = 1.0

    def _encode_opponent_buildings(self, tensor: torch.Tensor, state: GameState, player_id: int, channel: int):
        """Encode all opponent buildings."""
        for pid, player in state.players.items():
            if pid != player_id:
                # Encode settlements as 1.0, cities as 2.0
                for settlement in player.settlements:
                    pos = self._vertex_to_position(settlement)
                    if pos is not None:
                        row, col = pos
                        tensor[0, channel, row, col] = 1.0

                for city in player.cities:
                    pos = self._vertex_to_position(city)
                    if pos is not None:
                        row, col = pos
                        tensor[0, channel, row, col] = 2.0

    def _encode_valid_placements(self, tensor: torch.Tensor, state: GameState, player_id: int, channel: int):
        """Encode valid placement locations for current player."""
        # This would require implementing logic to find valid settlement/road positions
        # For now, leave as zeros - can be enhanced later
        pass

    def _tile_to_position(self, tile_id: int) -> Tuple[int, int]:
        """Convert tile ID to (row, col) position in spatial grid."""
        # Simple mapping for standard Catan board
        # This is a simplified version - real implementation would need proper hex->grid mapping
        tile_positions = {
            0: (1, 3), 1: (2, 2), 2: (2, 4), 3: (3, 1), 4: (3, 3), 5: (3, 5),
            6: (4, 2), 7: (4, 4), 8: (5, 1), 9: (5, 3), 10: (5, 5),
            11: (6, 2), 12: (6, 4), 13: (1, 1), 14: (1, 5), 15: (2, 0),
            16: (2, 6), 17: (4, 0), 18: (4, 6)
        }
        return tile_positions.get(tile_id)

    def _vertex_to_position(self, vertex_id: int) -> Tuple[int, int]:
        """Convert vertex ID to approximate (row, col) position."""
        # Simplified mapping - would need proper implementation
        row = min(vertex_id // 8, self.board_size[0] - 1)
        col = min(vertex_id % 8, self.board_size[1] - 1)
        return (row, col)

    def _edge_to_position(self, edge_id: int) -> Tuple[int, int]:
        """Convert edge ID to approximate (row, col) position."""
        # Simplified mapping - would need proper implementation
        row = min(edge_id // 10, self.board_size[0] - 1)
        col = min(edge_id % 10, self.board_size[1] - 1)
        return (row, col)

    def get_state_shape(self) -> Tuple[int, int, int]:
        """Get shape of encoded state: (channels, height, width)."""
        return (self.num_channels, self.board_size[0], self.board_size[1])

    def get_feature_names(self) -> list:
        """Get channel descriptions."""
        return [
            "Resource Types",
            "Number Tokens",
            "Robber Position",
            "My Settlements",
            "My Cities",
            "My Roads",
            "Opponent Buildings",
            "Valid Placements"
        ]

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize spatial state for stable training."""
        # Resource types and positions are already normalized
        # Could add per-channel normalization here if needed
        return state


# Register with factory
StateEncoderFactory.register("spatial", SpatialStateEncoder)