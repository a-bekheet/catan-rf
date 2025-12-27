"""Multi-layer perceptron networks for feature-based state representations."""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseQNetwork, NetworkFactory


class MLPQNetwork(BaseQNetwork):
    """
    Multi-layer perceptron for processing feature-based state representations.

    Designed to work with FeatureStateEncoder - processes hand-crafted strategic
    features through fully connected layers for decision making.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        # Validate input shape for feature data
        if len(state_shape) != 1:  # (features,)
            raise ValueError(f"MLPQNetwork expects 1D input (features,), got {state_shape}")

        input_dim = state_shape[0]

        # Network hyperparameters
        self.hidden_dims = config.get('hidden_dims', [512, 256, 128])
        self.dropout_rate = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        self.batch_norm = config.get('batch_norm', True)

        # Build fully connected layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if self.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif self.activation == 'elu':
                layers.append(nn.ELU(inplace=True))

            # Dropout (except for last hidden layer)
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation for Q-values)
        layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through MLP network."""
        batch_size = state.size(0)

        # Ensure proper shape for batch norm
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Get Q-values
        q_values = self.network(state)

        return q_values

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def get_layer_activations(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate layer activations for analysis."""
        activations = {}
        x = state

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        layer_idx = 0
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                x = module(x)
                activations[f'linear_{layer_idx}'] = x.clone()
                layer_idx += 1
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                x = module(x)
                activations[f'activation_{layer_idx-1}'] = x.clone()
            elif isinstance(module, (nn.BatchNorm1d, nn.Dropout)):
                x = module(x)

        return activations


class DuelingMLPQNetwork(BaseQNetwork):
    """
    Dueling architecture with MLP backbone.

    Separates state value V(s) and action advantages A(s,a) for better
    learning of state values independent of action selection.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        if len(state_shape) != 1:
            raise ValueError(f"DuelingMLPQNetwork expects 1D input (features,), got {state_shape}")

        input_dim = state_shape[0]

        # Shared network hyperparameters
        hidden_dims = config.get('hidden_dims', [512, 256])
        dropout_rate = config.get('dropout', 0.3)
        activation = config.get('activation', 'relu')
        batch_norm = config.get('batch_norm', True)

        # Build shared feature extractor
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                shared_layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                shared_layers.append(nn.LeakyReLU(0.1, inplace=True))
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Value stream V(s) - estimates state value
        value_dim = config.get('value_dim', 128)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, value_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(value_dim, 1)
        )

        # Advantage stream A(s,a) - estimates action advantages
        advantage_dim = config.get('advantage_dim', 256)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, advantage_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(advantage_dim, action_dim)
        )

        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through dueling MLP network."""
        batch_size = state.size(0)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Shared feature extraction
        shared_features = self.shared_network(state)

        # Value and advantage streams
        value = self.value_stream(shared_features)  # [batch_size, 1]
        advantage = self.advantage_stream(shared_features)  # [batch_size, action_dim]

        # Combine using dueling architecture formula:
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean

        return q_values

    def get_value_and_advantage(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate value and advantage estimates."""
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.unsqueeze(0)

            shared_features = self.shared_network(state)
            value = self.value_stream(shared_features)
            advantage = self.advantage_stream(shared_features)

            return value, advantage

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class ResidualMLPQNetwork(BaseQNetwork):
    """
    MLP with residual connections for deeper networks.

    Uses skip connections to enable training of deeper networks
    while maintaining gradient flow.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        if len(state_shape) != 1:
            raise ValueError(f"ResidualMLPQNetwork expects 1D input (features,), got {state_shape}")

        input_dim = state_shape[0]

        # Network hyperparameters
        hidden_dim = config.get('hidden_dim', 512)
        num_blocks = config.get('num_blocks', 3)
        dropout_rate = config.get('dropout', 0.3)

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.apply(self._init_weights)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through residual MLP network."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Project to hidden dimension
        x = self.input_proj(state)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output Q-values
        q_values = self.output_layers(x)

        return q_values

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ResidualBlock(nn.Module):
    """Residual block for deeper MLP networks."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.layers(x)
        x = x + residual  # Skip connection
        x = F.relu(x)
        x = self.dropout(x)
        return x


# Register networks with factory
NetworkFactory.register("mlp", MLPQNetwork)
NetworkFactory.register("dueling_mlp", DuelingMLPQNetwork)
NetworkFactory.register("residual_mlp", ResidualMLPQNetwork)