# Deep Q-Network (DQN) Framework for Catan RL

## Overview

This document describes the comprehensive PyTorch-based Deep Q-Network (DQN) framework that has been integrated with the existing Catan game engine. This modular framework allows for systematic experimentation with different RL approaches and provides a scalable foundation for Catan AI research.

## 🏗️ Architecture

### Modular Design Philosophy

The framework is built around **modularity** and **composability**, allowing researchers to:
- Mix and match different components
- Easily compare approaches
- Add new architectures without breaking existing code
- Conduct systematic ablation studies

### Core Components

#### 1. State Encoders (`catan_rl/environments/state_encoders/`)

**Purpose**: Convert game states into neural network input formats

- **`FeatureStateEncoder`**: Hand-crafted 85-dimensional strategic features
  - Resources, buildings, strategic positions
  - Expansion potential, development focus
  - Opponent analysis and relative positioning

- **`SpatialStateEncoder`**: 12-channel spatial representation for CNNs
  - Board layout as image-like tensor
  - Resource types, number tokens, robber position
  - Player buildings, opponent structures, valid placements

#### 2. Neural Networks (`catan_rl/agents/networks/`)

**Purpose**: Q-value approximation with different architectures

- **`MLPQNetwork`**: Multi-layer perceptron for feature inputs
- **`ConvQNetwork`**: Convolutional networks for spatial inputs
- **`DuelingMLPQNetwork`** & **`DuelingConvQNetwork`**: Dueling architectures
- **`HybridQNetwork`**: Combines spatial and feature inputs
- **`AttentionFusionQNetwork`**: Attention-based multi-modal fusion
- **`EnsembleQNetwork`**: Ensemble of multiple networks

#### 3. DQN Algorithms (`catan_rl/agents/algorithms/`)

**Purpose**: Training algorithms with different improvements

- **`VanillaDQN`**: Standard Deep Q-Network
- **`DoubleDQN`**: Reduces Q-value overestimation bias
- **`DuelingDQN`**: Separates state values and action advantages
- **`RainbowDQN`**: Combines multiple DQN improvements

#### 4. Experience Replay (`catan_rl/agents/memory/`)

**Purpose**: Store and sample experiences for training

- **`UniformReplayBuffer`**: Standard uniform random sampling
- **`PrioritizedReplayBuffer`**: Priority-based sampling with importance weights
- **`EpisodicReplayBuffer`**: Stores complete episodes
- **`NStepReplayBuffer`**: Multi-step return computation

#### 5. Action Space Management (`catan_rl/environments/action_space.py`)

**Purpose**: Handle complex Catan action space

- **`ActionEncoder`**: Maps complex actions to discrete indices
- **`ActionMasker`**: Creates binary masks for legal actions
- **`CatanActionSpace`**: Complete action space management

## 🚀 Getting Started

### Installation

1. **Install Dependencies**
   ```bash
   pip install torch torchvision numpy matplotlib seaborn pandas scipy scikit-learn
   ```

2. **Verify Installation**
   ```bash
   python scripts/demo_integration.py
   ```

### Quick Start Examples

#### 1. Basic Training

```bash
# Train baseline DQN agent
python scripts/train_dqn_agents.py --config baseline --episodes 1000

# Train spatial CNN agent
python scripts/train_dqn_agents.py --config comparative --episodes 500

# Advanced Rainbow DQN training
python scripts/train_dqn_agents.py --config analysis --episodes 2000
```

#### 2. Agent Evaluation

```bash
# Evaluate single agent
python scripts/evaluate_dqn_agents.py --model checkpoints/baseline/final --agent-type baseline --games 100

# Compare multiple agents
python scripts/evaluate_dqn_agents.py --compare baseline spatial rainbow --games 100

# Run tournament
python scripts/evaluate_dqn_agents.py --tournament --rounds 5 --games 50
```

#### 3. Custom Experiments

```bash
# Run modular experiments
python examples/run_dqn_experiment.py --config baseline
python examples/run_dqn_experiment.py --config compare_all
```

## 🔬 Research Capabilities

### Pre-built Agent Configurations

The framework includes three ready-to-use configurations optimized for different research questions:

1. **Baseline Agent** (`DQNAgentFactory.create_baseline_agent()`)
   - Feature-based state encoding (85 dimensions)
   - MLP network (512→256→128)
   - Vanilla DQN with uniform replay
   - Good starting point and baseline comparison

2. **Spatial Agent** (`DQNAgentFactory.create_spatial_agent()`)
   - Spatial state encoding (12 channels, 7×7 grid)
   - Convolutional network (32→64→128 channels)
   - Double DQN with prioritized experience replay
   - Tests spatial understanding of board layout

3. **Rainbow Agent** (`DQNAgentFactory.create_rainbow_agent()`)
   - Feature-based encoding with dueling architecture
   - N-step returns, prioritized replay, target networks
   - State-of-the-art DQN improvements
   - Maximum performance configuration

### Systematic Experimentation

#### Component Ablation Studies

```python
# Test different state encoders
configs = {
    'feature_mlp': {'state_encoder': 'feature', 'network': 'mlp'},
    'spatial_cnn': {'state_encoder': 'spatial', 'network': 'conv'},
    'hybrid_fusion': {'state_encoder': 'hybrid', 'network': 'attention_fusion'}
}

# Test different algorithms
algorithms = ['vanilla', 'double', 'dueling', 'rainbow']

# Test different replay buffers
buffers = ['uniform', 'prioritized', 'episodic', 'n_step']
```

#### Hyperparameter Optimization

```python
# Example custom configuration
config = {
    'state_encoder': {
        'type': 'feature',
        'params': {}
    },
    'network': {
        'type': 'mlp',
        'params': {
            'hidden_dims': [512, 256, 128],
            'dropout': 0.3,
            'activation': 'relu'
        }
    },
    'algorithm': {
        'type': 'double',
        'params': {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'target_update_frequency': 1000
        }
    }
}
```

## 📊 Analysis and Evaluation

### Comprehensive Metrics

The framework tracks detailed metrics for analysis:

- **Performance**: Win rates, scores, game lengths
- **Learning**: Q-values, TD errors, loss curves
- **Exploration**: Epsilon decay, action distributions
- **Efficiency**: Training time, network parameters
- **Robustness**: Performance across different opponents

### Statistical Analysis

- Significance testing between configurations
- Confidence intervals and error bars
- Consistency analysis across multiple runs
- Performance breakdown by game phase

### Visualization Tools

- Training curves and learning progress
- Performance comparisons and rankings
- Head-to-head matchup analysis
- Action distribution heatmaps

## 🎯 Research Applications

### Immediate Research Questions

1. **State Representation**: How do different state encodings affect learning?
2. **Architecture Design**: CNN vs MLP vs hybrid approaches?
3. **Algorithm Improvements**: Which DQN variants work best for Catan?
4. **Sample Efficiency**: How much training data is needed?
5. **Generalization**: Do agents learn transferable strategies?

### Advanced Research Directions

1. **Multi-Agent Learning**: Self-play and population-based training
2. **Curriculum Learning**: Structured opponent progression
3. **Transfer Learning**: Knowledge transfer between game variants
4. **Interpretability**: Understanding learned strategies
5. **Human-AI Interaction**: Collaborative play capabilities

## 🔧 Integration with Existing System

### Backward Compatibility

The new DQN framework is fully integrated with the existing game engine:

- **`DQNAgentAdapter`**: Makes DQN agents compatible with existing `RLAgent` interface
- **`DQNGameBridge`**: Handles communication between DQN agents and game state
- **Reward System**: Uses the enhanced reward system from the original RL agent

### Migration Path

Existing code can be upgraded gradually:

```python
# Old approach
from catan_rl.core.game.agents.rl_agent import RLAgent
agent = RLAgent(player_id=0)

# New approach
from catan_rl.core.integration import DQNAgentAdapter
agent = DQNAgentAdapter(player_id=0, dqn_config_name='baseline')
```

### Performance Improvements

The new framework addresses key limitations of the Q-learning approach:

- **State Space**: Neural networks handle continuous state spaces
- **Generalization**: Learned representations generalize across similar states
- **Sample Efficiency**: Experience replay and target networks improve learning
- **Scalability**: GPU acceleration for faster training

## 📈 Results and Benchmarks

### Expected Performance Improvements

Based on the framework design and DQN literature, we expect:

- **Win Rate**: 40-60% against random opponents (vs 15-25% for Q-learning)
- **Sample Efficiency**: Convergence in 1000-5000 episodes (vs 20,000+ for Q-learning)
- **Strategy Quality**: More sophisticated building and trading strategies
- **Consistency**: More stable performance across different game conditions

### Benchmark Metrics

Standard evaluation includes:
- Win rate vs random, baseline, and other DQN agents
- Average victory points and game length
- Training time and computational requirements
- Robustness across different board layouts

## 🚧 Future Development

### Planned Enhancements

1. **Additional Algorithms**: PPO, A3C, multi-agent RL
2. **Advanced Architectures**: Transformers, graph neural networks
3. **Curriculum Learning**: Structured training progressions
4. **Interactive Tools**: Real-time visualization and analysis
5. **Distributed Training**: Multi-GPU and distributed setups

### Research Integration

The framework is designed to support ongoing research:

- Easy addition of new state encoders and network architectures
- Pluggable algorithm implementations
- Comprehensive logging and experiment tracking
- Integration with popular ML tools (TensorBoard, Weights & Biases)

## 📚 Code Structure

```
catan_rl/
├── agents/
│   ├── algorithms/          # DQN algorithm variants
│   ├── networks/            # Neural network architectures
│   ├── memory/              # Experience replay buffers
│   └── dqn_agent.py        # Main DQN agent implementation
├── environments/
│   ├── state_encoders/      # State representation methods
│   └── action_space.py     # Action space management
├── evaluation/              # Evaluation and comparison tools
├── experiments/             # Experiment management
└── core/
    └── integration/         # Bridge to existing game engine
```

## 🤝 Contributing

The modular design makes it easy to contribute new components:

1. **New State Encoders**: Implement `BaseStateEncoder` interface
2. **New Networks**: Extend `BaseQNetwork` class
3. **New Algorithms**: Implement `BaseDQN` interface
4. **New Replay Buffers**: Extend `BaseReplayBuffer` class

Each component includes comprehensive documentation and examples for easy extension and modification.

---

This framework provides a solid foundation for advanced Catan AI research while maintaining compatibility with existing systems. The modular design and comprehensive evaluation tools enable systematic investigation of different RL approaches and facilitate reproducible research results.