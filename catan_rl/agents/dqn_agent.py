"""Complete DQN agent implementation for Catan."""

from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np
import random
from pathlib import Path
import json

from .algorithms.dqn import BaseDQN, DQNFactory
from .networks.base import BaseQNetwork, NetworkFactory
from .memory.replay_buffer import BaseReplayBuffer, ReplayBufferFactory, Experience
from ..environments.state_encoders.base import BaseStateEncoder, StateEncoderFactory
from ..environments.action_space import CatanActionSpace
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action


class SimpleAgentMetrics:
    """Simple metrics class for DQN agent."""
    def __init__(self, decision_time: float = 0.0, confidence: float = 0.0,
                 exploration_rate: float = 0.0, additional_data: Dict[str, Any] = None):
        self.decision_time = decision_time
        self.confidence = confidence
        self.exploration_rate = exploration_rate
        self.additional_data = additional_data or {}


class DQNAgent:
    """
    Deep Q-Network agent for Catan.

    Modular DQN implementation with swappable components:
    - State encoders (spatial, feature, hybrid)
    - Network architectures (CNN, MLP, hybrid)
    - DQN algorithms (vanilla, double, dueling, rainbow)
    - Replay buffers (uniform, prioritized, episodic)
    """

    def __init__(self, agent_id: int, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.state_encoder = self._create_state_encoder()
        self.action_space = CatanActionSpace(config.get('action_space', {}))
        self.q_network = self._create_network()
        self.dqn_algorithm = self._create_algorithm()
        self.replay_buffer = self._create_replay_buffer()

        # Training parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.update_frequency = config.get('update_frequency', 4)
        self.warmup_steps = config.get('warmup_steps', 1000)

        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.last_state = None
        self.last_action_idx = None
        self.last_action_mask = None

        # Performance tracking
        self.training_metrics = {
            'total_reward': 0.0,
            'episode_length': 0,
            'wins': 0,
            'losses': [],
            'q_values': [],
            'td_errors': []
        }

    def _create_state_encoder(self) -> BaseStateEncoder:
        """Create state encoder from config."""
        encoder_config = self.config.get('state_encoder', {})
        encoder_type = encoder_config.get('type', 'feature')
        encoder_params = encoder_config.get('params', {})

        return StateEncoderFactory.create(encoder_type, config=encoder_params)

    def _create_network(self) -> BaseQNetwork:
        """Create Q-network from config."""
        network_config = self.config.get('network', {})
        network_type = network_config.get('type', 'mlp')
        network_params = network_config.get('params', {})

        # Get state shape from encoder
        state_shape = self.state_encoder.get_state_shape()
        action_dim = self.action_space.get_action_space_size()

        return NetworkFactory.create(
            network_type,
            state_shape,
            action_dim,
            config=network_params
        )

    def _create_algorithm(self) -> BaseDQN:
        """Create DQN algorithm from config."""
        algorithm_config = self.config.get('algorithm', {})
        algorithm_type = algorithm_config.get('type', 'vanilla')
        algorithm_params = algorithm_config.get('params', {})

        return DQNFactory.create(algorithm_type, self.q_network, algorithm_params)

    def _create_replay_buffer(self) -> BaseReplayBuffer:
        """Create replay buffer from config."""
        buffer_config = self.config.get('replay_buffer', {})
        buffer_type = buffer_config.get('type', 'uniform')
        buffer_capacity = buffer_config.get('capacity', 10000)
        buffer_params = buffer_config.get('params', {})

        return ReplayBufferFactory.create(
            buffer_type,
            buffer_capacity,
            device=self.device,
            **buffer_params
        )

    def select_action(self, game_state: GameState, legal_actions: List[Action]) -> Tuple[Action, SimpleAgentMetrics]:
        """Select action using epsilon-greedy policy."""
        # Encode state
        encoded_state = self.state_encoder.encode(game_state, self.agent_id)

        # Create action mask based on legal actions
        action_mask = self._create_legal_action_mask(legal_actions)

        # Update epsilon
        if self.step_count > self.warmup_steps:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Select action with epsilon-greedy
        if np.random.random() < self.epsilon:
            # Random action from legal actions
            selected_action = np.random.choice(legal_actions)
            action_idx = self.action_space.encode_action(selected_action)
        else:
            # Greedy action selection from legal actions only
            self.q_network.eval()  # Set to eval mode for inference
            action_idx = self.dqn_algorithm.select_action(encoded_state, action_mask, 0.0)  # No exploration
            self.q_network.train()  # Return to training mode
            selected_action = self.action_space.decode_action(action_idx)

            # Double-check that selected action is valid
            if selected_action is None or selected_action not in legal_actions:
                # Force fallback to random legal action
                selected_action = np.random.choice(legal_actions)
                action_idx = self.action_space.encode_action(selected_action)

        # Store state-action for experience creation
        self.last_state = encoded_state
        self.last_action_idx = action_idx
        self.last_action_mask = action_mask

        # Create metrics with network in eval mode
        self.q_network.eval()  # Ensure BatchNorm uses running stats
        with torch.no_grad():
            q_values = self.q_network.get_q_values(encoded_state, action_mask)
            max_q_value = q_values.max().item()
        self.q_network.train()  # Return to training mode

        # Ensure action_idx is an integer for indexing
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()

        # Get Q-value for selected action safely
        try:
            if q_values.dim() == 1:
                selected_q_value = q_values[action_idx].item()
            elif q_values.dim() == 2:
                selected_q_value = q_values[0, action_idx].item()
            else:
                # Flatten and index for higher dimensions
                selected_q_value = q_values.flatten()[action_idx].item()
        except (IndexError, RuntimeError):
            selected_q_value = 0.0

        metrics = SimpleAgentMetrics(
            decision_time=0.0,  # Would measure actual time in practice
            confidence=max_q_value,
            exploration_rate=self.epsilon,
            additional_data={
                'action_idx': action_idx,
                'q_value': selected_q_value,
                'valid_actions': action_mask.sum().item()
            }
        )

        self.step_count += 1
        return selected_action, metrics

    def _create_legal_action_mask(self, legal_actions: List[Action]) -> torch.Tensor:
        """Create action mask for legal actions only."""
        action_space_size = self.action_space.get_action_space_size()
        mask = torch.zeros(action_space_size)

        for action in legal_actions:
            action_idx = self.action_space.encode_action(action)
            if action_idx is not None:
                mask[action_idx] = 1.0

        return mask.to(self.device)

    def observe_reward(self, reward: float, next_state: GameState, done: bool):
        """Observe reward and next state, update replay buffer."""
        if self.last_state is not None and self.last_action_idx is not None:
            # Encode next state
            next_encoded_state = self.state_encoder.encode(next_state, self.agent_id)
            next_action_mask = self.action_space.get_action_mask(next_state, self.agent_id)

            # Create experience
            experience = Experience(
                state=self.last_state,
                action=self.last_action_idx,
                reward=reward,
                next_state=next_encoded_state,
                done=done,
                action_mask=self.last_action_mask,
                next_action_mask=next_action_mask
            )

            # Add to replay buffer
            self.replay_buffer.add(experience)

            # Update training metrics
            self.training_metrics['total_reward'] += reward
            self.training_metrics['episode_length'] += 1

            # Train if ready
            if (self.step_count > self.warmup_steps and
                self.replay_buffer.can_sample(self.dqn_algorithm.config.get('batch_size', 32)) and
                self.step_count % self.update_frequency == 0):
                self._train_step()

        # Reset if episode done
        if done:
            self._end_episode(reward > 0)  # Win if positive reward

    def _train_step(self):
        """Perform one training step."""
        batch_size = self.dqn_algorithm.config.get('batch_size', 32)
        batch = self.replay_buffer.sample(batch_size)

        # Update network
        metrics = self.dqn_algorithm.update(batch)

        # Update replay buffer priorities if using prioritized replay
        if hasattr(self.replay_buffer, 'update_priorities') and 'td_errors' in metrics:
            indices = batch.get('indices')
            if indices is not None:
                self.replay_buffer.update_priorities(indices, metrics['td_errors'])

        # Track training metrics
        if 'loss' in metrics:
            self.training_metrics['losses'].append(metrics['loss'])
        if 'q_value_mean' in metrics:
            self.training_metrics['q_values'].append(metrics['q_value_mean'])
        if 'td_error_mean' in metrics:
            self.training_metrics['td_errors'].append(metrics['td_error_mean'])

    def _end_episode(self, won: bool):
        """End episode and reset state."""
        self.episode_count += 1

        if won:
            self.training_metrics['wins'] += 1

        # Reset episode state
        self.last_state = None
        self.last_action_idx = None
        self.last_action_mask = None
        self.training_metrics['total_reward'] = 0.0
        self.training_metrics['episode_length'] = 0

        # End episode for replay buffer if needed
        if hasattr(self.replay_buffer, 'end_episode'):
            self.replay_buffer.end_episode()

    def get_metrics(self) -> Dict[str, Any]:
        """Get training and performance metrics."""
        metrics = {}

        # Add DQN-specific metrics
        metrics.update({
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'replay_buffer_size': len(self.replay_buffer),
            'wins': self.training_metrics['wins'],
            'win_rate': self.training_metrics['wins'] / max(1, self.episode_count),
            'avg_loss': np.mean(self.training_metrics['losses'][-100:]) if self.training_metrics['losses'] else 0.0,
            'avg_q_value': np.mean(self.training_metrics['q_values'][-100:]) if self.training_metrics['q_values'] else 0.0,
            'avg_td_error': np.mean(self.training_metrics['td_errors'][-100:]) if self.training_metrics['td_errors'] else 0.0
        })

        return metrics

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save DQN algorithm checkpoint
        dqn_checkpoint_path = checkpoint_path.with_suffix('.dqn.pth')
        self.dqn_algorithm.save_checkpoint(str(dqn_checkpoint_path))

        # Save agent state
        agent_state = {
            'agent_id': self.agent_id,
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_metrics': self.training_metrics
        }

        agent_checkpoint_path = checkpoint_path.with_suffix('.agent.json')
        with open(agent_checkpoint_path, 'w') as f:
            json.dump(agent_state, f, indent=2)

        # Save replay buffer if supported
        if hasattr(self.replay_buffer, 'save'):
            buffer_checkpoint_path = checkpoint_path.with_suffix('.buffer.pkl')
            self.replay_buffer.save(str(buffer_checkpoint_path))

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint_path = Path(path)

        # Load DQN algorithm checkpoint
        dqn_checkpoint_path = checkpoint_path.with_suffix('.dqn.pth')
        if dqn_checkpoint_path.exists():
            self.dqn_algorithm.load_checkpoint(str(dqn_checkpoint_path))

        # Load agent state
        agent_checkpoint_path = checkpoint_path.with_suffix('.agent.json')
        if agent_checkpoint_path.exists():
            with open(agent_checkpoint_path, 'r') as f:
                agent_state = json.load(f)

            self.epsilon = agent_state.get('epsilon', self.epsilon)
            self.step_count = agent_state.get('step_count', 0)
            self.episode_count = agent_state.get('episode_count', 0)
            self.training_metrics = agent_state.get('training_metrics', self.training_metrics)

        # Load replay buffer if supported
        if hasattr(self.replay_buffer, 'load'):
            buffer_checkpoint_path = checkpoint_path.with_suffix('.buffer.pkl')
            if buffer_checkpoint_path.exists():
                self.replay_buffer.load(str(buffer_checkpoint_path))

    def reset(self):
        """Reset agent for new episode."""
        self.last_state = None
        self.last_action_idx = None
        self.last_action_mask = None
        self.training_metrics['total_reward'] = 0.0
        self.training_metrics['episode_length'] = 0

    def set_evaluation_mode(self, eval_mode: bool):
        """Set agent to evaluation mode (no exploration, no training)."""
        if eval_mode:
            self.epsilon = 0.0  # No exploration
            self.q_network.eval()
        else:
            self.q_network.train()

    def get_network_info(self) -> Dict[str, Any]:
        """Get information about network architecture."""
        return {
            'state_encoder': self.state_encoder.get_config(),
            'network': self.q_network.get_network_info(),
            'algorithm': {
                'type': self.dqn_algorithm.__class__.__name__,
                'config': self.dqn_algorithm.config
            },
            'replay_buffer': {
                'type': self.replay_buffer.__class__.__name__,
                'capacity': self.replay_buffer.capacity,
                'size': len(self.replay_buffer)
            },
            'action_space_size': self.action_space.get_action_space_size()
        }


class DQNAgentFactory:
    """Factory for creating DQN agents with different configurations."""

    @staticmethod
    def create_baseline_agent(agent_id: int) -> DQNAgent:
        """Create baseline DQN agent with simple configuration."""
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
                    'batch_norm': False
                }
            },
            'algorithm': {
                'type': 'vanilla',
                'params': {
                    'learning_rate': 1e-4,
                    'gamma': 0.99,
                    'target_update_frequency': 1000,
                    'batch_size': 32
                }
            },
            'replay_buffer': {
                'type': 'uniform',
                'capacity': 50000,
                'params': {}
            },
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'update_frequency': 4,
            'warmup_steps': 1000
        }
        return DQNAgent(agent_id, config)

    @staticmethod
    def create_spatial_agent(agent_id: int) -> DQNAgent:
        """Create DQN agent with spatial CNN architecture."""
        config = {
            'state_encoder': {
                'type': 'spatial',
                'params': {
                    'board_size': (7, 7)
                }
            },
            'network': {
                'type': 'conv',
                'params': {
                    'conv_channels': [32, 64, 128],
                    'hidden_dim': 512,
                    'dropout': 0.3
                }
            },
            'algorithm': {
                'type': 'double',
                'params': {
                    'learning_rate': 1e-4,
                    'gamma': 0.99,
                    'target_update_frequency': 1000,
                    'batch_size': 32
                }
            },
            'replay_buffer': {
                'type': 'prioritized',
                'capacity': 50000,
                'params': {
                    'alpha': 0.6,
                    'beta_start': 0.4
                }
            },
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'update_frequency': 4,
            'warmup_steps': 1000
        }
        return DQNAgent(agent_id, config)

    @staticmethod
    def create_rainbow_agent(agent_id: int) -> DQNAgent:
        """Create Rainbow DQN agent with all improvements."""
        config = {
            'state_encoder': {
                'type': 'feature',
                'params': {}
            },
            'network': {
                'type': 'dueling_mlp',
                'params': {
                    'hidden_dims': [512, 256],
                    'value_dim': 128,
                    'advantage_dim': 256,
                    'dropout': 0.3
                }
            },
            'algorithm': {
                'type': 'rainbow',
                'params': {
                    'learning_rate': 1e-4,
                    'gamma': 0.99,
                    'target_update_frequency': 1000,
                    'batch_size': 32,
                    'n_step_returns': 3,
                    'use_prioritized_replay': True
                }
            },
            'replay_buffer': {
                'type': 'prioritized',
                'capacity': 50000,
                'params': {
                    'alpha': 0.6,
                    'beta_start': 0.4
                }
            },
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'update_frequency': 4,
            'warmup_steps': 1000
        }
        return DQNAgent(agent_id, config)