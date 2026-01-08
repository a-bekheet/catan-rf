# 🚀 Quick Start Guide - Multi-Agent RL Training

## Installation (First Time Only)

### Using UV (Recommended - 10x Faster!)
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies with UV
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Using pip (Alternative)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Set API Keys (Optional - for LLM agent)
```bash
# OpenAI (default)
export OPENAI_API_KEY="your-key-here"

# or Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage - Three Easy Ways!

### 1. Interactive Menu (Easiest)
```bash
python run.py
# Select option 2: Multi-Agent RL Training
```

### 2. Direct Training
```bash
# Train PPO agent (100 episodes)
python train_agents.py --agent ppo --episodes 100

# Train SAC agent (100 episodes)
python train_agents.py --agent sac --episodes 100

# Train LLM agent (100 episodes)
python train_agents.py --agent llm --episodes 100

# Train ALL agents + tournament
python train_agents.py --agent all --episodes 100 --tournament 50
```

### 3. Interactive Training CLI
```bash
python train_agents.py
# Follow the prompts!
```

## What You'll See

During training, you'll see beautiful progress bars and real-time stats:

```
🚀 Multi-Agent RL Training System
══════════════════════════════════════════════════════

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

## Results

After training, your models will be saved in:
```
checkpoints/multi_rl_YYYYMMDD_HHMMSS/
├── ppo_final/     # Trained PPO agent
├── sac_final/     # Trained SAC agent
└── llm_final/     # Trained LLM agent
```

## Next Steps

1. **Train your first agent**:
   ```bash
   python train_agents.py --agent ppo --episodes 50
   ```

2. **Compare all agents**:
   ```bash
   python train_agents.py --agent all --episodes 100 --tournament 50
   ```

3. **Read the full docs**:
   - `MULTI_AGENT_TRAINING.md` - Complete documentation
   - `README.md` - Project overview

## Troubleshooting

### "ModuleNotFoundError: No module named 'ray'"
```bash
# With UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### LLM agent not working?
It's okay! The LLM agent works in fallback mode without API keys.
To use GPT-4/Claude:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Out of memory?
Reduce batch size in training config or train one agent at a time.

---

**That's it! You're ready to train AI agents!** 🎉
