# 🎯 Catan RL System - Quick Start

## 🚀 How to Run Everything

### **Simple One-Command Launch**
```bash
python run.py
```

This launches an interactive menu with all available options:

### 🎮 **Available Options**

1. **🌐 Web Interface** - Interactive browser game at http://localhost:8001
2. **🎲 CLI Game** - Simple command-line game with random agents
3. **🤖 DQN Integration Test** - Verify PyTorch DQN framework works
4. **🎯 Basic Training Demo** - Legacy simple training demo
5. **🚀 Advanced Training System** - Full-featured training with visualization
6. **🏆 Training Tournament** - Compare different agent types
7. **📊 System Info** - Check dependencies and integration status

### 🔧 **Quick Commands**

#### Start Web Server Directly:
```bash
# Option 1: Via unified CLI
python run.py
# Choose option 1

# Option 2: Direct command
PYTHONPATH=src uvicorn catan.web.server:app --host 127.0.0.1 --port 8001 --reload
```

#### Test DQN Integration:
```bash
# Option 1: Via unified CLI
python run.py
# Choose option 3

# Option 2: Direct command
python scripts/demo_integration.py
```

#### Check System Status:
```bash
python run.py
# Choose option 5
```

### ✅ **What's Working**

- ✅ **Web Interface** - Full interactive Catan game in browser
- ✅ **DQN Framework** - PyTorch deep learning agents integrated
- ✅ **State Encoders** - 90-feature vectors + 12×7×7 spatial tensors
- ✅ **Action Space** - 1017 possible actions properly encoded
- ✅ **Game Engine** - Complete Catan rules and mechanics

### 🧠 **AI Agents Available**

- **Random Agent** - Baseline random player
- **DQN Agent** - Deep Q-Network (untrained initially)
- **Spatial DQN** - CNN-based agent for board patterns
- **Rainbow DQN** - Advanced DQN with all improvements

### 🎓 **For Development**

The system is modular and ready for:
- Training new agents
- Experimenting with different neural networks
- Adding new state representations
- Implementing other RL algorithms

### 🌐 **Web Interface**

Once running at http://localhost:8001:
- Click buildings/roads to place them
- Use resource/dev card panels
- Full game rules implemented
- Real-time game state updates

### 🚀 **Advanced Training System**

The new training system includes:
- **Multiple Training Modes**: Demo (20 episodes), Quick (100), Standard (1K), Full (10K)
- **Real-time Visualization**: Live plots of win rates, rewards, learning curves
- **Progress Tracking**: Detailed statistics and performance metrics
- **Model Management**: Automatic checkpointing and model saving
- **Tournament Mode**: Compare different agent types head-to-head

#### Training Modes:
```bash
python run.py
# Choose option 5 -> Advanced Training System

# Available modes:
# 1. Demo (20 episodes) - Quick test
# 2. Quick (100 episodes) - Short training session
# 3. Standard (1K episodes) - Full training
# 4. Full (10K episodes) - Extensive training
# 5. Custom - Set your own parameters
```

#### Tournament System:
```bash
python run.py
# Choose option 6 -> Training Tournament

# Compares:
# - Baseline DQN
# - Spatial CNN DQN
# - Rainbow DQN
# - Multiple agent configurations
```

#### Training Features:
- **Live Progress Bars**: Visual feedback during training
- **Performance Metrics**: Win rates, average rewards, invalid action rates
- **Checkpoint System**: Save/load trained models
- **Evaluation Periods**: Regular testing during training
- **Training Interruption**: Pause/resume with Ctrl+C

### 📊 **Training Output**
The training system provides:
- Real-time progress visualization
- Episode statistics and summaries
- Agent performance comparisons
- Model checkpoints in `checkpoints/` directory
- Training plots and analysis

### 🎯 **Next Steps**

1. **Play the game**: Use the web interface
2. **Quick training**: Try the demo mode (20 episodes)
3. **Full training**: Run standard mode for real learning
4. **Tournament**: Compare different agent types
5. **Experiment**: Modify agents in `catan_rl/agents/`
6. **Analyze**: Check training results and plots

---
*🤖 PyTorch DQN Framework Successfully Integrated!*