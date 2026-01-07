"""Base state encoder interface for different state representations."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import numpy as np

from catan_rl.core.game.engine.game_state import GameState


class BaseStateEncoder(ABC):
    """Abstract base class for state encoders."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def encode(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """
        Encode game state for the given player.

        Args:
            game_state: Current game state
            player_id: ID of the player whose perspective to encode

        Returns:
            Encoded state as torch tensor
        """
        pass

    @abstractmethod
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get the shape of encoded state tensor."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list:
        """Get human-readable feature names for analysis."""
        pass

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device."""
        return tensor.to(self.device)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state values for stable training."""
        return state

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration for reproducibility."""
        return {
            'encoder_type': self.__class__.__name__,
            'config': self.config,
            'state_shape': self.get_state_shape()
        }


class StateEncoderFactory:
    """Factory for creating different state encoders."""

    _encoders = {}

    @classmethod
    def register(cls, name: str, encoder_class: type):
        """Register a new encoder type."""
        cls._encoders[name] = encoder_class

    @classmethod
    def create(cls, encoder_type: str, **kwargs) -> BaseStateEncoder:
        """Create encoder of specified type."""
        if encoder_type not in cls._encoders:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        return cls._encoders[encoder_type](**kwargs)

    @classmethod
    def list_encoders(cls) -> list:
        """List all registered encoder types."""
        return list(cls._encoders.keys())