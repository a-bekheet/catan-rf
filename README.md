# Catan Reinforcement Learning

Deep reinforcement learning framework for Settlers of Catan using PyTorch DQN agents.

## Features

- **Complete Game Engine**: Full Catan rules implementation with setup, trading, building, and victory conditions
- **DQN Agents**: Baseline MLP, Spatial CNN, and Rainbow DQN architectures
- **State Encoders**: 90-feature vector encoder and 12×7×7 spatial tensor encoder
- **Action Space**: 1017 possible actions with legal action masking
- **Training System**: Progress tracking, checkpointing, and real-time visualization
- **Web Interface**: Interactive browser-based gameplay
- **Tournament Mode**: Multi-agent evaluation and comparison

## Quick Start

```bash
python run.py
```

Choose from:
1. **Web Interface** - Interactive game at `http://localhost:8001`
2. **DQN Training** - Train agents with various configurations
3. **Tournament** - Compare agent performance

## Training Results

Recent demo (20 episodes):
- **Agent 2**: 4.8% win rate (Rainbow DQN)
- **Agent 3**: 2.4% win rate (Baseline DQN)
- **Agent 0**: 1.2% win rate (Baseline DQN)

## Architecture

### State Representation
- **Feature Encoder**: Hand-crafted strategic features (resources, buildings, positions)
- **Spatial Encoder**: CNN-compatible board representation with resource/building channels

### Action Space
- Building actions (settlements, cities, roads)
- Trading (player-to-player, bank)
- Development cards (buy, play)
- Game flow (dice roll, robber movement, turn management)

### Reward System
- Victory bonus: +100 points
- Victory point gain: +10 per VP
- Building progress: +1 per building
- Time penalty: -0.01 per turn
- Loss penalty: -10 points

## Requirements

- Python 3.11+
- PyTorch 2.0+
- FastAPI (for web interface)
- NumPy, matplotlib

## Training Modes

- **Demo**: 20 episodes (quick test)
- **Quick**: 100 episodes
- **Standard**: 1,000 episodes
- **Full**: 10,000 episodes
- **Custom**: User-defined parameters

Agents learn through epsilon-greedy exploration with experience replay and target networks.