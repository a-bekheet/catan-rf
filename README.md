# Catan Multi-Agent RL Training System

Train AI agents using cutting-edge RL frameworks to play Settlers of Catan!

## 🚀 Quick Start (UV - Recommended)

### Automatic Setup (Easiest!)
```bash
./setup.sh
source .venv/bin/activate
python train_agents.py
```

### Manual Setup
```bash
# 1. Install UV (blazingly fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Activate environment
source .venv/bin/activate

# 4. Start training!
python train_agents.py
```

## 📦 Alternative: Using pip

```bash
pip install -e .
python train_agents.py
```

---

## 🎯 What's Included

## 🤖 Three RL Frameworks

1. **Ray RLlib (PPO)** - Proximal Policy Optimization for fast distributed training
2. **TorchRL (SAC)** - Soft Actor-Critic for sample-efficient learning
3. **LangGraph (LLM)** - Strategic reasoning with GPT-4/Claude

## ✨ Features

- **Complete Game Engine**: Full Catan rules implementation
- **Beautiful CLI**: Real-time progress bars, ETA, and metrics
- **Auto Checkpointing**: Never lose training progress
- **Tournament Mode**: Compare all agents head-to-head
- **Web Interface**: Play against trained AI agents
- **Easy Setup**: One command with UV
- **Comprehensive Docs**: Step-by-step guides included

## 📖 Documentation

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Get started in 2 minutes
- **[UV_SETUP.md](UV_SETUP.md)** - Complete UV setup guide
- **[MULTI_AGENT_TRAINING.md](MULTI_AGENT_TRAINING.md)** - Full training documentation

## 🎮 Usage

### Interactive Menu
```bash
python run.py
# Select option 2: Multi-Agent RL Training
```

### Direct Training
```bash
# Train specific agent
python train_agents.py --agent ppo --episodes 100

# Train all agents + tournament
python train_agents.py --agent all --episodes 100 --tournament 50
```

## 📊 Example Output

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

## 🏗️ Architecture

```
train_agents.py              # Beautiful training CLI
├── BaseRLAgent             # Common interface
├── RLlibPPOAgent          # Ray RLlib (PPO)
├── TorchRLSACAgent        # TorchRL (SAC)
└── LangGraphLLMAgent      # LangGraph (LLM)

Features:
- 90-feature state encoder
- 1017-action space with masking
- Automatic checkpointing
- Tournament comparison mode
```

## ⚙️ Requirements

- Python 3.11+
- UV (recommended) or pip
- Optional: OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM agent

## 🎯 Training Tips

### Fast Training
```bash
python train_agents.py --agent ppo --episodes 50
```

### Best Performance
```bash
python train_agents.py --agent sac --episodes 1000
```

### Strategic Play
```bash
export OPENAI_API_KEY="your-key"
python train_agents.py --agent llm --episodes 100
```