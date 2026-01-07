# Catan RL Training Guide

This guide explains how to use and continue training the reinforcement learning (RL) bots for your Catan game.

## Quick Start

### 1. Train New RL Agents
```bash
# Train for 1000 episodes (default)
PYTHONPATH=src python scripts/train_rl_bots.py

# Train for a specific number of episodes
PYTHONPATH=src python scripts/train_rl_bots.py --episodes 500

# Train with verbose output (includes progress bar and detailed stats)
PYTHONPATH=src python scripts/train_rl_bots.py --episodes 200 --verbose
```

### 2. Watch Trained Agents Compete
```bash
# Run 5-match tournament between RL and random agents
PYTHONPATH=src python scripts/simple_bot_match.py
```

## How It Works

### Agent Types

**RL Agent** (`RLAgent`)
- Uses Q-learning with epsilon-greedy exploration
- Learns from game outcomes and rewards
- Automatically saves/loads trained models from `models/` directory
- **Model Files**: `models/rl_agent_p0.json`, `models/rl_agent_p1.json`

**Random Agent** (`RandomAgent`)
- Selects random legal actions (baseline for comparison)
- No learning or persistence

### Training Process

1. **Multi-Agent Setup**: 2 RL agents compete against 2 random agents
2. **Learning**: RL agents receive rewards for:
   - Victory points gained (+10 per VP)
   - Winning the game (+100)
   - Building progress (settlements: +2, cities: +3, roads: +0.5)
   - Losing penalty (-50 if someone else wins)
3. **Exploration**: Epsilon-greedy strategy balances exploration vs exploitation
4. **Persistence**: Q-tables automatically save after training

### Model Persistence

- **Automatic Loading**: RL agents automatically load existing models when created
- **Automatic Saving**: Models save after each training session
- **File Location**: `models/rl_agent_p{player_id}.json`
- **Contents**: Q-table, learning parameters, current epsilon value

## Training Strategies

### Incremental Training
```bash
# Start with short sessions
PYTHONPATH=src python scripts/train_rl_bots.py --episodes 100

# Continue with longer sessions
PYTHONPATH=src python scripts/train_rl_bots.py --episodes 500

# Keep extending training
PYTHONPATH=src python scripts/train_rl_bots.py --episodes 1000
```

### Monitoring Progress
- **Progress Bar**: Real-time training progress with episode completion percentage
- **Turn Statistics**: Total turns played and average turns per game
- **Performance Metrics**: Games per minute, current epsilon values
- **Rewards**: Watch average rewards increase over time
- **Exploration**: Monitor epsilon decay (less exploration over time)
- **Evaluation**: Test performance with `simple_bot_match.py` periodically

### Verbose Output Features
When using `--verbose`, you get:
- **Live Progress Bar**: `[████████████░░░░░░] 40/50 (80.0%)`
- **Timing Info**: ⏱️ 6.4s elapsed, 🎮 374.7 games/min
- **Turn Tracking**: 🎲 40,000 total turns (1000.0 average per game)
- **Game Results**: 📊 Winner info or timeout status
- **Learning Progress**: ε: 0.067/0.067 (current epsilon values)
- **Detailed Final Stats**: Total turns played and comprehensive results

## Advanced Usage

### Custom Model Paths
```python
# In your own scripts
from catan.agents.rl_agent import RLAgent

agent = RLAgent(
    player_id=0,
    model_path="custom_models/my_agent.json"
)
```

### Hyperparameter Tuning
```python
agent = RLAgent(
    player_id=0,
    learning_rate=0.05,    # How fast to learn (default: 0.1)
    epsilon=0.3,           # Exploration rate (default: 0.1)
    discount=0.99          # Future reward importance (default: 0.95)
)
```

### Manual Model Management
```python
# Save manually
agent.save_model()

# Check if model exists
from pathlib import Path
if Path("models/rl_agent_p0.json").exists():
    print("Model exists!")
```

## Troubleshooting

### No Models Loading
- Check that `models/` directory exists
- Verify file permissions
- Look for JSON syntax errors in model files

### Poor Performance
- Increase training episodes (try 2000+ episodes)
- Adjust learning rate (try 0.05 or 0.2)
- Monitor that rewards are increasing during training

### Training Too Slow
- Reduce episodes for testing
- Remove `--verbose` flag
- Consider simpler reward functions

## File Structure

```
catan-rf/
├── models/                          # Saved RL models
│   ├── rl_agent_p0.json            # Player 0 Q-table
│   └── rl_agent_p1.json            # Player 1 Q-table
├── scripts/
│   ├── train_rl_bots.py            # Training script
│   └── simple_bot_match.py         # Tournament script
└── src/catan/agents/
    ├── rl_agent.py                 # Q-learning implementation
    └── random_agent.py             # Random baseline agent
```

## Next Steps

1. **Longer Training**: Try 2000+ episodes for better performance
2. **Hyperparameter Experiments**: Adjust learning rates and exploration
3. **Better State Representation**: Enhance the `_state_key()` method in `RLAgent`
4. **Reward Function Tuning**: Modify `compute_reward()` for better learning signals
5. **Deep RL**: Consider neural networks instead of Q-tables for complex strategies

## Performance Expectations

- **Early Training**: Agents play randomly, low rewards
- **100-500 Episodes**: Basic patterns emerge, positive average rewards
- **1000+ Episodes**: More strategic play, better building decisions
- **5000+ Episodes**: Advanced strategies, competitive with rule-based bots

The current implementation provides a solid foundation for RL experimentation in Catan!