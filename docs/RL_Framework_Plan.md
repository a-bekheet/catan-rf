# Catan RL Research Framework

## Architecture Overview

```
catan-rf/
├── src/
│   ├── rl_framework/
│   │   ├── base/
│   │   │   ├── agent.py              # Abstract agent interface
│   │   │   ├── environment.py        # Standardized Catan environment
│   │   │   ├── experiment.py         # Experiment runner/manager
│   │   │   └── metrics.py            # Performance tracking
│   │   ├── agents/
│   │   │   ├── random_agent.py       # Baseline: random play
│   │   │   ├── rule_based_agent.py   # Baseline: heuristic rules
│   │   │   ├── qtable_agent.py       # Current Q-learning approach
│   │   │   ├── dqn_agent.py          # Deep Q-Network
│   │   │   ├── ppo_agent.py          # Proximal Policy Optimization
│   │   │   ├── a3c_agent.py          # Asynchronous Actor-Critic
│   │   │   └── transformer_agent.py  # Transformer-based agent
│   │   ├── environments/
│   │   │   ├── catan_env.py          # OpenAI Gym interface
│   │   │   ├── state_encoder.py      # Multiple state representations
│   │   │   └── action_space.py       # Action encoding/masking
│   │   ├── training/
│   │   │   ├── curriculum.py         # Curriculum learning
│   │   │   ├── self_play.py          # Self-play infrastructure
│   │   │   └── population.py         # Population-based training
│   │   └── analysis/
│   │       ├── visualize.py          # Game replay/visualization
│   │       ├── statistics.py         # Statistical analysis
│   │       └── compare.py            # Agent comparison tools
├── experiments/
│   ├── configs/                      # Hyperparameter configurations
│   ├── results/                      # All experimental data
│   └── notebooks/                    # Jupyter analysis notebooks
└── data/
    ├── game_logs/                    # Detailed game recordings
    ├── training_metrics/             # Training progress data
    ├── agent_models/                 # Saved model checkpoints
    └── analysis_cache/               # Preprocessed data for analysis
```

## Data Collection Strategy

### 1. Game-Level Data
```json
{
  "game_id": "uuid",
  "timestamp": "2024-01-01T00:00:00Z",
  "experiment_id": "dqn_vs_random_001",
  "agents": ["DQN_v1.2", "Random", "Random", "Random"],
  "game_length": 156,
  "winner": 0,
  "victory_condition": "10_victory_points",
  "final_scores": [10, 3, 4, 2],
  "game_phases": {
    "setup_duration": 16,
    "main_game_duration": 140
  }
}
```

### 2. Turn-Level Data
```json
{
  "turn_id": 42,
  "player_id": 0,
  "agent_type": "DQN_v1.2",
  "game_state_before": "encoded_state",
  "legal_actions": ["action_ids"],
  "action_taken": "build_settlement_vertex_23",
  "action_confidence": 0.87,
  "thinking_time": 0.023,
  "reward": 8.5,
  "game_state_after": "encoded_state",
  "strategic_metrics": {
    "victory_points": 4,
    "settlements": 3,
    "cities": 0,
    "roads": 5,
    "development_cards": 1,
    "resources": {"brick": 1, "lumber": 0, "ore": 2, "grain": 1, "wool": 0},
    "expansion_potential": 2
  }
}
```

### 3. Training-Level Data
```json
{
  "episode": 1500,
  "agent_id": "DQN_v1.2",
  "training_metrics": {
    "loss": 0.034,
    "q_value_mean": 15.7,
    "epsilon": 0.12,
    "learning_rate": 0.0003,
    "replay_buffer_size": 50000
  },
  "evaluation_results": {
    "win_rate_vs_random": 0.73,
    "avg_game_length": 145,
    "avg_victory_points": 7.2,
    "strategic_distribution": {
      "building_focused": 0.6,
      "development_focused": 0.3,
      "mixed_strategy": 0.1
    }
  }
}
```

## Experiment Types

### 1. Baseline Comparison
- All agents vs. Random agents
- Head-to-head tournaments
- Performance over time tracking

### 2. Ablation Studies
- Different state representations
- Various reward functions
- Network architecture variations
- Hyperparameter sweeps

### 3. Curriculum Learning
- Setup-only training → Full game
- Simple scenarios → Complex strategies
- Single-opponent → Multi-opponent

### 4. Self-Play Evolution
- Population diversity tracking
- Strategy emergence patterns
- Skill progression curves

## Key Research Questions

1. **Which RL algorithm learns Catan most efficiently?**
   - Sample efficiency comparison
   - Convergence speed analysis
   - Final performance ceiling

2. **What state representation works best?**
   - Raw board state vs. engineered features
   - Spatial convolutions vs. fully connected
   - Multi-modal inputs (board + player state)

3. **How do different reward functions affect strategy?**
   - Sparse vs. dense rewards
   - Strategic milestones vs. outcome-only
   - Multi-objective optimization

4. **What emerges from self-play?**
   - Counter-strategies and meta-games
   - Diversity of successful approaches
   - Human-like vs. novel strategies

5. **How does curriculum learning help?**
   - Optimal task progression
   - Transfer learning between game phases
   - Robustness to different scenarios

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. Base classes and interfaces
2. Standardized environment wrapper
3. Data collection infrastructure
4. Simple baseline agents

### Phase 2: Core Algorithms (Week 2)
1. DQN implementation with multiple variants
2. Policy gradient methods (PPO)
3. Training pipelines and evaluation

### Phase 3: Advanced Features (Week 3)
1. Self-play infrastructure
2. Population-based training
3. Analysis and visualization tools

### Phase 4: Research & Analysis (Week 4)
1. Comprehensive experiments
2. Statistical analysis
3. Research paper preparation
4. Open-source release

## Success Metrics

### Technical Performance
- **Sample Efficiency**: Games to reach 70% win rate vs. random
- **Peak Performance**: Max win rate achieved
- **Robustness**: Performance vs. different opponent types
- **Generalization**: Transfer to board variants/rule changes

### Strategic Insights
- **Strategy Diversity**: Number of distinct winning approaches discovered
- **Human Similarity**: Comparison with human expert play patterns
- **Emergent Behaviors**: Novel strategies not known to humans
- **Counter-Play**: Ability to adapt to opponent strategies

This framework will give us comprehensive data to write a definitive analysis of "Which RL approach works best for Catan and why."