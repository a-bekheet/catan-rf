# Modular PyTorch DQN Architecture for Catan RL

## 🎯 Design Philosophy

Create a **highly modular** DQN system where we can easily swap:
- **State Representations**: Spatial vs Feature-based vs Hybrid
- **Network Architectures**: CNN vs MLP vs Transformer
- **DQN Variants**: Vanilla vs Double vs Dueling vs Rainbow
- **Training Strategies**: Experience Replay vs Prioritized vs Multi-step

## 🧩 Modular Components

### 1. State Encoders (`catan_rl/environments/state_encoders/`)
```python
class BaseStateEncoder(ABC):
    @abstractmethod
    def encode(self, game_state: GameState, player_id: int) -> torch.Tensor
    @abstractmethod
    def get_state_shape(self) -> Tuple[int, ...]

# Multiple implementations:
- SpatialEncoder      # Board as image-like tensor
- FeatureEncoder      # Hand-crafted features
- GraphEncoder        # Graph neural network
- HybridEncoder       # Combination approach
```

### 2. Network Architectures (`catan_rl/agents/networks/`)
```python
class BaseQNetwork(nn.Module):
    @abstractmethod
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor

# Multiple architectures:
- ConvQNetwork        # CNN for spatial understanding
- MLPQNetwork         # Fully connected
- DuelingQNetwork     # Dueling architecture (V + A streams)
- TransformerQNetwork # Attention-based
- HybridQNetwork      # CNN + MLP combination
```

### 3. DQN Algorithms (`catan_rl/agents/dqn_variants/`)
```python
class BaseDQNAgent(BaseAgent):
    # Swappable components
    def __init__(self, state_encoder, q_network, experience_replay, ...):

# Multiple DQN variants:
- VanillaDQN          # Standard DQN
- DoubleDQN           # Reduces overestimation
- DuelingDQN          # Value/Advantage decomposition
- RainbowDQN          # All improvements combined
- NoisyDQN            # Noisy networks for exploration
```

### 4. Experience Replay (`catan_rl/agents/replay/`)
```python
# Multiple replay strategies:
- UniformReplay       # Standard random sampling
- PrioritizedReplay   # Sample important transitions
- HindsightReplay     # Modified goals for sparse rewards
- EpisodicReplay      # Episode-aware sampling
```

## 🔬 Experimental Matrix

We can systematically test combinations:

| State Encoder | Network | DQN Variant | Replay | Experiment ID |
|---------------|---------|-------------|--------|---------------|
| Spatial       | CNN     | Vanilla     | Uniform| DQN_001       |
| Spatial       | CNN     | Double      | Priority| DQN_002       |
| Feature       | MLP     | Dueling     | Uniform| DQN_003       |
| Hybrid        | Hybrid  | Rainbow     | Priority| DQN_004       |
| Graph         | GNN     | Vanilla     | Episodic| DQN_005       |

## 🎨 State Representation Approaches

### Spatial Encoding (Image-like)
```python
# Board as multi-channel tensor [channels, height, width]
channels = [
    'resource_types',    # Brick=1, Lumber=2, etc.
    'number_tokens',     # 2-12 values
    'robber_position',   # Binary mask
    'my_settlements',    # Binary mask
    'my_cities',         # Binary mask
    'my_roads',          # Binary mask
    'opponent_buildings',# Separate channels per opponent
    'valid_placements'   # Action mask overlay
]
```

### Feature Encoding (Hand-crafted)
```python
# Flat feature vector
features = [
    # Player state (20 features)
    'resources[5]', 'buildings[3]', 'dev_cards[5]', 'victory_points',
    'knights_played', 'longest_road', 'largest_army',

    # Board state (30 features)
    'tile_productivity[19]', 'port_access[5]', 'robber_effects[4]',

    # Strategic features (15 features)
    'expansion_potential', 'resource_diversity', 'building_efficiency',
    'opponent_threat_level', 'endgame_position'
]
```

### Hybrid Encoding
```python
# Combine spatial + features
spatial_features = cnn_encoder(board_tensor)     # [batch, 128]
numeric_features = feature_encoder(game_vector) # [batch, 64]
combined = torch.cat([spatial_features, numeric_features], dim=1)
```

## 🧠 Network Architecture Variants

### 1. Convolutional Network
```python
class ConvQNetwork(nn.Module):
    def __init__(self):
        # Spatial understanding layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),  # Board patterns
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), # Complex patterns
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))     # Global features
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size)
        )
```

### 2. Dueling Architecture
```python
class DuelingQNetwork(nn.Module):
    def __init__(self):
        self.feature_layer = FeatureExtractor()
        self.value_stream = nn.Linear(256, 1)        # State value V(s)
        self.advantage_stream = nn.Linear(256, actions)  # Action advantages A(s,a)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

## 🎮 Action Space Design

```python
# Structured action representation
class CatanActionSpace:
    BUILD_SETTLEMENT = list(range(0, 54))      # 54 vertices
    BUILD_CITY = list(range(54, 108))          # 54 vertices
    BUILD_ROAD = list(range(108, 180))         # 72 edges
    TRADE_BANK = list(range(180, 200))         # 20 trade combinations
    BUY_DEV_CARD = [200]
    PLAY_KNIGHT = list(range(201, 220))        # 19 tiles
    PASS_TURN = [220]

    total_actions = 221

# Action masking for invalid actions
def get_action_mask(state: GameState, player_id: int) -> torch.Tensor:
    mask = torch.zeros(221)
    legal_actions = state.legal_actions()
    for action in legal_actions:
        action_id = action_to_id(action)
        mask[action_id] = 1
    return mask
```

## 🔄 Training Pipeline

```python
class ModularDQNTrainer:
    def __init__(self,
                 state_encoder: BaseStateEncoder,
                 q_network: BaseQNetwork,
                 replay_buffer: BaseReplayBuffer,
                 config: DQNConfig):
        self.components = {
            'encoder': state_encoder,
            'network': q_network,
            'replay': replay_buffer
        }

    def train_step(self):
        # Modular training allows easy experimentation
        batch = self.replay.sample()
        states = self.encoder.encode(batch.states)
        q_values = self.network(states, batch.action_masks)
        loss = self.compute_loss(q_values, batch.targets)
        return loss
```

## 📊 Evaluation Framework

```python
class DQNEvaluator:
    def evaluate_agent(self, agent, num_games=100):
        return {
            'win_rate': float,
            'avg_score': float,
            'strategic_metrics': dict,
            'sample_efficiency': dict,
            'component_analysis': {
                'encoder_effectiveness': float,
                'network_utilization': dict,
                'replay_quality': dict
            }
        }
```

This modular design lets us systematically answer:
1. **Which state representation works best?**
2. **What network architecture is most effective?**
3. **How do different DQN improvements contribute?**
4. **What's the optimal component combination?**

Ready to start implementation? 🚀