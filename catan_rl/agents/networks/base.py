"""Base Q-network interface for different architectures."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class BaseQNetwork(nn.Module, ABC):
    """Abstract base class for Q-networks."""

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: Encoded state tensor
            action_mask: Binary mask for valid actions (1 = valid, 0 = invalid)

        Returns:
            Q-values for all actions
        """
        pass

    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor = None, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: State tensor [batch_size, ...] or single state [...]
            action_mask: Valid action mask
            epsilon: Exploration rate

        Returns:
            Selected action index
        """
        if len(state.shape) == len(self.state_shape):
            # Single state, add batch dimension
            state = state.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0) if action_mask is not None else None

        with torch.no_grad():
            # Get Q-values
            q_values = self.forward(state, action_mask)

            # Apply action masking
            if action_mask is not None:
                q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

            # Epsilon-greedy action selection
            if torch.rand(1).item() < epsilon:
                # Random action among valid actions
                if action_mask is not None:
                    valid_actions = torch.nonzero(action_mask[0]).squeeze(-1)
                    if len(valid_actions) > 0:
                        action_idx = torch.randint(0, len(valid_actions), (1,)).item()
                        return valid_actions[action_idx].item()
                    else:
                        return 0  # Fallback if no valid actions
                else:
                    return torch.randint(0, self.action_dim, (1,)).item()
            else:
                # Greedy action
                return torch.argmax(q_values[0]).item()

    def get_q_values(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Get Q-values for state(s)."""
        return self.forward(state, action_mask)

    def save_checkpoint(self, path: str):
        """Save network parameters."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'state_shape': self.state_shape,
            'action_dim': self.action_dim
        }, path)

    def load_checkpoint(self, path: str):
        """Load network parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def get_network_info(self) -> Dict[str, Any]:
        """Get network architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'network_type': self.__class__.__name__,
            'state_shape': self.state_shape,
            'action_dim': self.action_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }

    def to_device(self):
        """Move network to appropriate device."""
        return self.to(self.device)


class NetworkFactory:
    """Factory for creating different network architectures."""

    _networks = {}

    @classmethod
    def register(cls, name: str, network_class: type):
        """Register a new network type."""
        cls._networks[name] = network_class

    @classmethod
    def create(cls, network_type: str, state_shape: Tuple, action_dim: int, **kwargs) -> BaseQNetwork:
        """Create network of specified type."""
        if network_type not in cls._networks:
            raise ValueError(f"Unknown network type: {network_type}")
        return cls._networks[network_type](state_shape, action_dim, **kwargs)

    @classmethod
    def list_networks(cls) -> list:
        """List all registered network types."""
        return list(cls._networks.keys())