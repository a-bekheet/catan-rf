"""Convolutional Q-network for spatial state representations."""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseQNetwork, NetworkFactory


class ConvQNetwork(BaseQNetwork):
    """
    Convolutional neural network for processing spatial board representations.

    Designed to work with SpatialStateEncoder - treats the Catan board as
    an image-like input with multiple channels representing different aspects
    of the game state.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        # Validate input shape for spatial data
        if len(state_shape) != 3:  # (channels, height, width)
            raise ValueError(f"ConvQNetwork expects 3D input (C,H,W), got {state_shape}")

        channels, height, width = state_shape

        # Network hyperparameters
        self.conv_channels = config.get('conv_channels', [32, 64, 128])
        self.kernel_size = config.get('kernel_size', 3)
        self.dropout_rate = config.get('dropout', 0.3)
        self.hidden_dim = config.get('hidden_dim', 512)

        # Convolutional layers for spatial feature extraction
        conv_layers = []
        in_channels = channels

        for out_channels in self.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout_rate)
            ])
            in_channels = out_channels

        self.conv_backbone = nn.Sequential(*conv_layers)

        # Calculate size after convolution
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            conv_output = self.conv_backbone(dummy_input)
            self.conv_output_size = conv_output.numel()

        # Fully connected layers for decision making
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through convolutional network."""
        # Handle both single samples (3D) and batches (4D)
        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False

        batch_size = state.size(0)

        # Spatial feature extraction
        spatial_features = self.conv_backbone(state)

        # Flatten for fully connected layers
        flattened = spatial_features.view(batch_size, -1)

        # Get Q-values
        q_values = self.fc_layers(flattened)

        # Remove batch dimension for single samples
        if single_sample:
            q_values = q_values.squeeze(0)

        return q_values

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def get_spatial_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract spatial features for analysis."""
        with torch.no_grad():
            spatial_features = self.conv_backbone(state)
            return spatial_features

    def get_attention_maps(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate attention maps for visualization."""
        with torch.no_grad():
            attention_maps = {}
            x = state

            # Track activations through conv layers
            layer_idx = 0
            for module in self.conv_backbone:
                if isinstance(module, nn.Conv2d):
                    x = module(x)
                    # Average across channels for visualization
                    attention = torch.mean(x, dim=1, keepdim=True)
                    attention_maps[f'conv_layer_{layer_idx}'] = attention
                    layer_idx += 1
                elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.Dropout2d)):
                    x = module(x)

            return attention_maps


class DuelingConvQNetwork(BaseQNetwork):
    """
    Dueling architecture with convolutional backbone.

    Separates state value V(s) and action advantages A(s,a) for better
    learning of state values independent of action selection.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        if len(state_shape) != 3:
            raise ValueError(f"DuelingConvQNetwork expects 3D input (C,H,W), got {state_shape}")

        channels, height, width = state_shape

        # Shared convolutional backbone
        conv_channels = config.get('conv_channels', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)
        dropout_rate = config.get('dropout', 0.3)
        hidden_dim = config.get('hidden_dim', 512)

        # Build shared conv layers
        conv_layers = []
        in_channels = channels

        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        self.shared_conv = nn.Sequential(*conv_layers)

        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            conv_output = self.shared_conv(dummy_input)
            conv_output_size = conv_output.numel()

        # Value stream V(s) - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream A(s,a) - estimates action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through dueling network."""
        batch_size = state.size(0)

        # Shared spatial features
        shared_features = self.shared_conv(state)
        flattened = shared_features.view(batch_size, -1)

        # Value and advantage streams
        value = self.value_stream(flattened)  # [batch_size, 1]
        advantage = self.advantage_stream(flattened)  # [batch_size, action_dim]

        # Combine using dueling architecture formula:
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean

        return q_values

    def get_value_and_advantage(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate value and advantage estimates."""
        with torch.no_grad():
            batch_size = state.size(0)
            shared_features = self.shared_conv(state)
            flattened = shared_features.view(batch_size, -1)

            value = self.value_stream(flattened)
            advantage = self.advantage_stream(flattened)

            return value, advantage

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


# Register networks with factory
NetworkFactory.register("conv", ConvQNetwork)
NetworkFactory.register("dueling_conv", DuelingConvQNetwork)