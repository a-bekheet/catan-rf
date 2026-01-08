# Multi-Agent RL Training System

## 🎯 Overview

This system enables training AI agents using three cutting-edge RL frameworks to play Settlers of Catan:

1. **Ray RLlib (PPO)** - Proximal Policy Optimization
2. **TorchRL (SAC)** - Soft Actor-Critic
3. **LangGraph (LLM)** - LLM-powered strategic reasoning

## 🚀 Quick Start

### Option 1: Interactive Menu
```bash
python run.py
# Select option 2: Multi-Agent RL Training
```

### Option 2: Direct Training
```bash
# Train a specific agent
python train_agents.py --agent ppo --episodes 100
python train_agents.py --agent sac --episodes 100
python train_agents.py --agent llm --episodes 100

# Train all agents + tournament
python train_agents.py --agent all --episodes 100 --tournament 50
```

### Option 3: Interactive Training CLI
```bash
python train_agents.py
# Follow the interactive prompts
```

## 📦 Installation

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional: LLM API Keys (for LangGraph agent)

Set environment variables for LLM providers:

```bash
# OpenAI (default)
export OPENAI_API_KEY="your-key-here"

# or Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"
```

**Note**: The LLM agent will work without API keys using heuristic fallback mode.

## 🤖 Agent Types

### 1. Ray RLlib PPO Agent

**Algorithm**: Proximal Policy Optimization
**Best For**: Fast distributed training, stable policy updates
**Key Features**:
- Clipped surrogate objective for stability
- Distributed training across multiple CPUs/GPUs
- Excellent for multi-agent environments

**Config Example**:
```python
config = {
    'state_encoder': {'type': 'feature'},
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'num_workers': 2,  # Parallel rollout workers
    'clip_param': 0.2,
}
```

### 2. TorchRL SAC Agent

**Algorithm**: Soft Actor-Critic
**Best For**: Sample efficiency, exploration
**Key Features**:
- Off-policy learning (efficient data usage)
- Maximum entropy principle for exploration
- Stable training with soft target updates

**Config Example**:
```python
config = {
    'state_encoder': {'type': 'feature'},
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'alpha': 0.2,  # Entropy temperature
    'batch_size': 256,
}
```

### 3. LangGraph LLM Agent

**Algorithm**: LLM Reasoning + RL Adaptation
**Best For**: Strategic decision-making, human-like play
**Key Features**:
- Uses GPT-4/Claude for strategic reasoning
- LangGraph workflow management
- Learns strategy preferences from outcomes

**Config Example**:
```python
config = {
    'llm_provider': 'openai',  # or 'anthropic'
    'model_name': 'gpt-4-turbo-preview',
    'temperature': 0.7,
}
```

## 📊 Training Progress

The training CLI provides real-time progress tracking:

```
Training PPO Agent...
━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45% 45/100 ETA: 02:35

╭─ PPO Agent - Episode 40/100 ─╮
│ Metric            Value       │
│ Win Rate          23.5%       │
│ Avg Reward        12.34       │
│ Avg Episode Time  2.45s       │
│ ETA               14:32:15    │
╰─────────────────────────────╯
```

## 💾 Checkpoints & Model Management

### Automatic Checkpointing

Models are automatically saved during training:

```
checkpoints/
├── multi_rl_20260108_123456/
│   ├── ppo_ep50/
│   │   ├── checkpoint_data
│   │   └── metadata.json
│   ├── ppo_final/
│   ├── sac_ep50/
│   ├── sac_final/
│   ├── llm_ep50/
│   └── llm_final/
```

### Loading Trained Models

```python
from catan_rl.agents.rllib_ppo_agent import RLlibPPOAgent
from pathlib import Path

# Create agent
agent = RLlibPPOAgent(agent_id=0, config={})

# Load checkpoint
checkpoint_path = Path("checkpoints/multi_rl_XXX/ppo_final")
agent.load_checkpoint(checkpoint_path)

# Set to evaluation mode
agent.set_training_mode(False)
```

## 🎮 Training Modes

### 1. Single Agent Training

Train one agent against random opponents:

```bash
python train_agents.py --agent ppo --episodes 100
```

### 2. All Agents Training

Train all three agents sequentially:

```bash
python train_agents.py --agent all --episodes 100
```

### 3. Tournament Mode

After training, agents compete against each other:

```bash
python train_agents.py --agent all --episodes 100 --tournament 50
```

Tournament results:

```
╔═ Tournament Results ═╗
║ Player │ Agent │ Wins │ Win Rate ║
║ 0      │ PPO   │ 15   │ 30.0%    ║
║ 1      │ SAC   │ 18   │ 36.0%    ║
║ 2      │ LLM   │ 12   │ 24.0%    ║
║ 3      │ Rand  │ 5    │ 10.0%    ║
╚════════════════════════════════╝
```

## ⚙️ Configuration

### State Encoders

Choose how the game state is represented:

```python
# Feature-based encoding (90 features)
config = {'state_encoder': {'type': 'feature'}}

# Spatial CNN encoding (12×7×7 tensor)
config = {'state_encoder': {'type': 'spatial'}}
```

### Training Hyperparameters

Customize training behavior:

```python
config = {
    'learning_rate': 3e-4,  # Learning rate
    'gamma': 0.99,          # Discount factor
    'epsilon_start': 1.0,   # Initial exploration
    'epsilon_min': 0.1,     # Min exploration
    'epsilon_decay': 0.995, # Exploration decay
}
```

## 🎯 Best Practices

### For Fast Training
- Use PPO with multiple workers
- Smaller network architectures
- Fewer episodes (100-500)

### For Best Performance
- SAC with large replay buffer
- More training episodes (1000+)
- Tune hyperparameters

### For Human-like Play
- LLM agent with GPT-4
- Higher temperature (0.7-0.9)
- Learn from human game transcripts

## 🔧 Troubleshooting

### Ray RLlib not working
```bash
# Reinstall Ray with RLlib
pip install --upgrade ray[rllib]
```

### TorchRL errors
```bash
# Install TorchRL and TensorDict
pip install torchrl tensordict
```

### LLM agent using fallback mode
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Verify key is loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Out of memory
- Reduce batch size
- Decrease replay buffer size
- Use fewer parallel workers

## 📈 Performance Tips

### Training Speed
1. **Use GPU**: Set `use_gpu: true` in config
2. **Parallel Workers**: Increase `num_workers` for PPO
3. **Smaller Batches**: Reduce `batch_size` for faster updates

### Sample Efficiency
1. **Use SAC**: Best sample efficiency
2. **Larger Replay Buffer**: More data to learn from
3. **Lower Learning Rate**: More stable learning

### Exploration
1. **Tune Entropy**: Higher alpha for SAC = more exploration
2. **Epsilon Schedule**: Decay epsilon slowly
3. **Diverse Training**: Mix opponent types

## 🎓 Next Steps

### After Training

1. **Evaluate Agents**: Run tournament mode
2. **Play Against AI**: Use web interface with trained agents
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Extend**: Add new agent architectures

### Advanced Usage

1. **Custom Agents**: Implement `BaseRLAgent` interface
2. **Curriculum Learning**: Train against progressively harder opponents
3. **Transfer Learning**: Use pre-trained models
4. **Multi-agent Self-Play**: Agents train against each other

## 📚 Architecture Overview

```
train_agents.py                    # Main training CLI
├── TrainingSession               # Training orchestration
└── BaseRLAgent implementations
    ├── RLlibPPOAgent            # Ray RLlib wrapper
    ├── TorchRLSACAgent          # TorchRL wrapper
    └── LangGraphLLMAgent        # LangGraph wrapper

catan_rl/agents/
├── base_rl_agent.py             # Common interface
├── rllib_ppo_agent.py           # PPO implementation
├── torchrl_sac_agent.py         # SAC implementation
└── langgraph_llm_agent.py       # LLM implementation
```

## 🌟 Features

✅ Beautiful CLI with progress bars
✅ Real-time training metrics
✅ Automatic checkpointing
✅ Tournament mode
✅ Multiple RL frameworks
✅ Configurable player positions
✅ Easy model loading
✅ Comprehensive logging

## 📖 References

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)

---

**Built with ❤️ for strategic AI research**
