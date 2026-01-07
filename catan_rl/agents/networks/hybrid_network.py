"""Hybrid networks that combine multiple input modalities."""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseQNetwork, NetworkFactory


class HybridQNetwork(BaseQNetwork):
    """
    Hybrid network that processes both spatial and feature inputs.

    Combines convolutional processing of spatial board state with
    MLP processing of strategic features for comprehensive understanding.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        # Expected input: dict with 'spatial' and 'features' keys
        self.spatial_shape = config.get('spatial_shape', (12, 7, 7))  # (C, H, W)
        self.feature_dim = config.get('feature_dim', 85)

        # Spatial processing branch (CNN)
        spatial_channels, height, width = self.spatial_shape
        conv_channels = config.get('conv_channels', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)

        # Build convolutional layers
        conv_layers = []
        in_channels = spatial_channels

        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2) if out_channels != conv_channels[-1] else nn.Identity()
            ])
            in_channels = out_channels

        self.spatial_backbone = nn.Sequential(*conv_layers)

        # Calculate spatial output size
        with torch.no_grad():
            dummy_spatial = torch.zeros(1, spatial_channels, height, width)
            spatial_output = self.spatial_backbone(dummy_spatial)
            self.spatial_output_size = spatial_output.numel()

        # Feature processing branch (MLP)
        feature_hidden_dims = config.get('feature_hidden_dims', [256, 128])
        feature_layers = []
        prev_dim = self.feature_dim

        for hidden_dim in feature_hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.feature_backbone = nn.Sequential(*feature_layers)
        self.feature_output_size = prev_dim

        # Fusion layer
        fusion_input_size = self.spatial_output_size + self.feature_output_size
        fusion_hidden = config.get('fusion_hidden', 512)
        dropout_rate = config.get('dropout', 0.3)

        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden // 2, action_dim)
        )

        self.apply(self._init_weights)

    def forward(self, state: Dict[str, torch.Tensor], action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through hybrid network."""
        if isinstance(state, torch.Tensor):
            raise ValueError("HybridQNetwork expects dict input with 'spatial' and 'features' keys")

        batch_size = state['spatial'].size(0)

        # Process spatial input
        spatial_features = self.spatial_backbone(state['spatial'])
        spatial_flat = spatial_features.view(batch_size, -1)

        # Process feature input
        feature_features = self.feature_backbone(state['features'])

        # Fuse modalities
        fused = torch.cat([spatial_flat, feature_features], dim=1)

        # Get Q-values
        q_values = self.fusion_layers(fused)

        return q_values

    def get_modality_features(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get intermediate features from each modality."""
        with torch.no_grad():
            batch_size = state['spatial'].size(0)

            # Spatial features
            spatial_features = self.spatial_backbone(state['spatial'])
            spatial_flat = spatial_features.view(batch_size, -1)

            # Feature features
            feature_features = self.feature_backbone(state['features'])

            return {
                'spatial_features': spatial_flat,
                'feature_features': feature_features,
                'spatial_raw': spatial_features
            }

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class AttentionFusionQNetwork(BaseQNetwork):
    """
    Hybrid network with attention-based fusion of modalities.

    Uses attention mechanisms to dynamically weight the importance
    of spatial vs feature information for each decision.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        self.spatial_shape = config.get('spatial_shape', (12, 7, 7))
        self.feature_dim = config.get('feature_dim', 85)

        # Spatial branch
        spatial_channels, height, width = self.spatial_shape
        self.spatial_backbone = self._build_spatial_backbone(spatial_channels, config)

        # Calculate spatial output size
        with torch.no_grad():
            dummy_spatial = torch.zeros(1, spatial_channels, height, width)
            spatial_output = self.spatial_backbone(dummy_spatial)
            self.spatial_output_size = spatial_output.numel()

        # Feature branch
        self.feature_backbone = self._build_feature_backbone(self.feature_dim, config)
        self.feature_output_size = config.get('feature_hidden_dims', [256, 128])[-1]

        # Attention mechanism
        attention_dim = config.get('attention_dim', 128)
        self.spatial_attention = nn.Linear(self.spatial_output_size, attention_dim)
        self.feature_attention = nn.Linear(self.feature_output_size, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 2)  # 2 modalities

        # Final layers
        fusion_hidden = config.get('fusion_hidden', 512)
        dropout_rate = config.get('dropout', 0.3)

        self.final_layers = nn.Sequential(
            nn.Linear(self.spatial_output_size + self.feature_output_size, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden, action_dim)
        )

        self.apply(self._init_weights)

    def _build_spatial_backbone(self, spatial_channels: int, config: Dict[str, Any]) -> nn.Module:
        """Build spatial processing backbone."""
        conv_channels = config.get('conv_channels', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)

        layers = []
        in_channels = spatial_channels

        for out_channels in conv_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_feature_backbone(self, feature_dim: int, config: Dict[str, Any]) -> nn.Module:
        """Build feature processing backbone."""
        hidden_dims = config.get('feature_hidden_dims', [256, 128])

        layers = []
        prev_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, state: Dict[str, torch.Tensor], action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with attention fusion."""
        if isinstance(state, torch.Tensor):
            raise ValueError("AttentionFusionQNetwork expects dict input")

        batch_size = state['spatial'].size(0)

        # Process modalities
        spatial_features = self.spatial_backbone(state['spatial'])
        spatial_flat = spatial_features.view(batch_size, -1)

        feature_features = self.feature_backbone(state['features'])

        # Compute attention weights
        spatial_attention = torch.tanh(self.spatial_attention(spatial_flat))
        feature_attention = torch.tanh(self.feature_attention(feature_features))

        # Combine attention contexts
        combined_attention = (spatial_attention + feature_attention) / 2
        attention_weights = F.softmax(self.attention_weights(combined_attention), dim=1)

        # Apply attention weighting
        spatial_weighted = spatial_flat * attention_weights[:, 0:1]
        feature_weighted = feature_features * attention_weights[:, 1:2]

        # Fuse and predict
        fused = torch.cat([spatial_weighted, feature_weighted], dim=1)
        q_values = self.final_layers(fused)

        return q_values

    def get_attention_weights(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get attention weights for analysis."""
        with torch.no_grad():
            batch_size = state['spatial'].size(0)

            spatial_features = self.spatial_backbone(state['spatial'])
            spatial_flat = spatial_features.view(batch_size, -1)
            feature_features = self.feature_backbone(state['features'])

            spatial_attention = torch.tanh(self.spatial_attention(spatial_flat))
            feature_attention = torch.tanh(self.feature_attention(feature_features))

            combined_attention = (spatial_attention + feature_attention) / 2
            attention_weights = F.softmax(self.attention_weights(combined_attention), dim=1)

            return attention_weights  # [batch_size, 2]

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class EnsembleQNetwork(BaseQNetwork):
    """
    Ensemble of multiple networks for robust predictions.

    Combines predictions from multiple architectures to improve
    robustness and reduce overfitting.
    """

    def __init__(self, state_shape: Tuple, action_dim: int, config: Dict[str, Any] = None):
        super().__init__(state_shape, action_dim, config)

        self.ensemble_size = config.get('ensemble_size', 3)
        self.ensemble_method = config.get('ensemble_method', 'average')  # 'average', 'max', 'voting'

        # Create multiple networks
        network_configs = config.get('network_configs', [])
        if not network_configs:
            # Default configurations for diversity
            network_configs = [
                {'hidden_dims': [512, 256, 128], 'dropout': 0.2},
                {'hidden_dims': [768, 384], 'dropout': 0.3},
                {'hidden_dims': [256, 512, 256], 'dropout': 0.4}
            ]

        self.networks = nn.ModuleList()
        for i in range(self.ensemble_size):
            config_idx = i % len(network_configs)
            net_config = network_configs[config_idx].copy()
            net_config.update(config.get('base_config', {}))

            # Import here to avoid circular imports
            from .mlp_network import MLPQNetwork
            network = MLPQNetwork(state_shape, action_dim, net_config)
            self.networks.append(network)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all networks
        predictions = []
        for network in self.networks:
            pred = network(state, action_mask)
            predictions.append(pred)

        # Combine predictions
        predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, action_dim]

        if self.ensemble_method == 'average':
            q_values = torch.mean(predictions, dim=0)
        elif self.ensemble_method == 'max':
            q_values = torch.max(predictions, dim=0)[0]
        elif self.ensemble_method == 'voting':
            # Majority voting on argmax
            actions = torch.argmax(predictions, dim=2)  # [ensemble_size, batch_size]
            q_values = torch.zeros_like(predictions[0])
            for i in range(q_values.size(0)):  # For each batch
                action_votes = actions[:, i]
                winner = torch.mode(action_votes)[0]
                q_values[i, winner] = 1.0
        else:
            q_values = torch.mean(predictions, dim=0)

        return q_values

    def get_ensemble_predictions(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> List[torch.Tensor]:
        """Get individual predictions from each network."""
        with torch.no_grad():
            predictions = []
            for network in self.networks:
                pred = network(state, action_mask)
                predictions.append(pred)
            return predictions

    def get_prediction_variance(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Get variance in predictions across ensemble."""
        with torch.no_grad():
            predictions = self.get_ensemble_predictions(state, action_mask)
            predictions = torch.stack(predictions, dim=0)
            variance = torch.var(predictions, dim=0)
            return variance


# Register networks with factory
NetworkFactory.register("hybrid", HybridQNetwork)
NetworkFactory.register("attention_fusion", AttentionFusionQNetwork)
NetworkFactory.register("ensemble", EnsembleQNetwork)