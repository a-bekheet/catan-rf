"""
TorchRL SAC Agent for Catan
============================

Wraps TorchRL's SAC (Soft Actor-Critic) algorithm for Catan gameplay.
SAC is excellent for:
- Sample efficiency (learns from less data)
- Maximum entropy RL (encourages exploration)
- Continuous/discrete action spaces
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
import json
import torch
import numpy as np

try:
    from torchrl.data import ReplayBuffer, LazyTensorStorage
    from torchrl.objectives import SACLoss
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule, TensorDictSequential
    TORCHRL_AVAILABLE = True
except ImportError:
    TORCHRL_AVAILABLE = False
    print("Warning: TorchRL not installed. Install with: pip install torchrl tensordict")

from .base_rl_agent import BaseRLAgent, AgentMetrics
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action
from ..environments.action_space import CatanActionSpace
from ..environments.state_encoders.base import StateEncoderFactory


class TorchRLSACAgent(BaseRLAgent):
    """
    TorchRL SAC agent wrapper for Catan.

    SAC (Soft Actor-Critic) features:
    - Off-policy learning (efficient data usage)
    - Entropy maximization (better exploration)
    - Stable training with soft updates
    """

    def __init__(
        self,
        agent_id: int,
        config: Dict[str, Any]
    ):
        super().__init__(agent_id, "TorchRL SAC Agent", config)

        if not TORCHRL_AVAILABLE:
            raise ImportError(
                "TorchRL is required for SACAgent. "
                "Install with: pip install torchrl tensordict"
            )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.get('use_gpu', False)
            else 'cpu'
        )

        # Setup state encoder and action space
        encoder_config = config.get('state_encoder', {'type': 'feature'})
        self.state_encoder = StateEncoderFactory.create(
            encoder_config.get('type', 'feature'),
            encoder_config.get('params', {})
        )
        self.action_space = CatanActionSpace(config.get('action_space', {}))

        # Get dimensions
        self.state_shape = self.state_encoder.get_state_shape()
        self.action_dim = self.action_space.get_action_space_size()
        self.state_dim = int(np.prod(self.state_shape)) if isinstance(self.state_shape, tuple) else self.state_shape

        # SAC hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)  # Entropy temperature
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 100000)

        # Build SAC components
        self._build_networks(config)
        self._build_replay_buffer()

        # Training state
        self.update_counter = 0

    def _build_networks(self, config: Dict[str, Any]) -> None:
        """Build actor and critic networks for SAC."""
        hidden_sizes = config.get('hidden_sizes', [256, 256])

        # Actor network (policy)
        actor_layers = []
        in_size = self.state_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([
                torch.nn.Linear(in_size, hidden_size),
                torch.nn.ReLU()
            ])
            in_size = hidden_size
        actor_layers.append(torch.nn.Linear(in_size, self.action_dim))
        actor_layers.append(torch.nn.Softmax(dim=-1))

        self.actor = torch.nn.Sequential(*actor_layers).to(self.device)

        # Critic networks (Q-functions) - SAC uses two Q-networks
        self.critic_1 = self._build_critic(hidden_sizes).to(self.device)
        self.critic_2 = self._build_critic(hidden_sizes).to(self.device)

        # Target critics for stable training
        self.target_critic_1 = self._build_critic(hidden_sizes).to(self.device)
        self.target_critic_2 = self._build_critic(hidden_sizes).to(self.device)

        # Initialize target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.learning_rate
        )

        # Automatic entropy tuning
        self.log_alpha = torch.tensor(
            np.log(self.alpha),
            requires_grad=True,
            device=self.device,
            dtype=torch.float32
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.learning_rate
        )
        self.target_entropy = -self.action_dim  # Heuristic target entropy

    def _build_critic(self, hidden_sizes: List[int]) -> torch.nn.Module:
        """Build a critic (Q-function) network."""
        layers = []
        in_size = self.state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(in_size, hidden_size),
                torch.nn.ReLU()
            ])
            in_size = hidden_size
        layers.append(torch.nn.Linear(in_size, self.action_dim))

        return torch.nn.Sequential(*layers)

    def _build_replay_buffer(self) -> None:
        """Build experience replay buffer."""
        self.replay_buffer = []
        self.buffer_capacity = self.buffer_size

    def select_action(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Tuple[Action, AgentMetrics]:
        """Select action using SAC policy."""
        start_time = time.time()

        # Encode state
        state_array = self.state_encoder.encode(game_state, self.agent_id)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

        # Create action mask
        action_mask = self._create_action_mask(legal_actions)

        # Get action probabilities from policy
        with torch.no_grad():
            action_probs = self.actor(state_tensor).cpu().numpy()[0]

        # Mask illegal actions
        masked_probs = action_probs * action_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback to uniform over legal actions
            masked_probs = action_mask / action_mask.sum()

        # Sample action during training, choose max during evaluation
        if self.is_training:
            action_idx = np.random.choice(len(masked_probs), p=masked_probs)
        else:
            action_idx = np.argmax(masked_probs)

        # Decode action
        selected_action = self.action_space.decode_action(action_idx)
        if selected_action is None or selected_action not in legal_actions:
            selected_action = np.random.choice(legal_actions)
            action_idx = self.action_space.encode_action(selected_action)

        # Calculate entropy
        entropy = -np.sum(masked_probs * np.log(masked_probs + 1e-8))

        # Create metrics
        metrics = AgentMetrics(
            decision_time=time.time() - start_time,
            confidence=float(masked_probs[action_idx]),
            exploration_rate=float(self.alpha),
            policy_entropy=float(entropy),
            additional_data={
                'action_idx': action_idx,
                'max_prob': float(masked_probs.max()),
            }
        )

        self.total_steps += 1
        return selected_action, metrics

    def _create_action_mask(self, legal_actions: List[Action]) -> np.ndarray:
        """Create binary mask for legal actions."""
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for action in legal_actions:
            action_idx = self.action_space.encode_action(action)
            if action_idx < self.action_dim:
                mask[action_idx] = 1.0
        return mask

    def update(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """Update SAC networks with new experience."""
        # Encode states
        state_array = self.state_encoder.encode(state, self.agent_id)
        next_state_array = self.state_encoder.encode(next_state, self.agent_id)
        action_idx = self.action_space.encode_action(action)

        # Store experience
        experience = {
            'state': state_array,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state_array,
            'done': done
        }
        self.replay_buffer.append(experience)

        # Keep buffer at capacity
        if len(self.replay_buffer) > self.buffer_capacity:
            self.replay_buffer.pop(0)

        # Train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        return self._train_step()

    def _train_step(self) -> Dict[str, float]:
        """Perform one SAC training step."""
        # Sample batch
        batch_indices = np.random.choice(
            len(self.replay_buffer),
            size=self.batch_size,
            replace=False
        )
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Prepare tensors
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Update alpha (temperature)
        alpha_loss = self._update_alpha(states)

        # Soft update target networks
        self._soft_update_targets()

        self.update_counter += 1

        return {
            'critic_loss': float(critic_loss),
            'actor_loss': float(actor_loss),
            'alpha_loss': float(alpha_loss),
            'alpha': float(self.alpha),
        }

    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Sample actions for next states
            next_action_probs = self.actor(next_states)
            next_action_log_probs = torch.log(next_action_probs + 1e-8)

            # Compute target Q-values
            next_q1 = self.target_critic_1(next_states)
            next_q2 = self.target_critic_2(next_states)
            next_q = torch.min(next_q1, next_q2)

            # Compute soft V-value (expectation over actions)
            next_v = (next_action_probs * (next_q - self.alpha * next_action_log_probs)).sum(dim=1)

            # Compute TD target
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # Current Q-values
        current_q1 = self.critic_1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.critic_2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Critic loss
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + \
                      torch.nn.functional.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        # Get action probabilities
        action_probs = self.actor(states)
        action_log_probs = torch.log(action_probs + 1e-8)

        # Compute Q-values
        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        q = torch.min(q1, q2)

        # Actor loss (maximize expected Q-value + entropy)
        actor_loss = (action_probs * (self.alpha * action_log_probs - q)).sum(dim=1).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states: torch.Tensor) -> float:
        """Update entropy temperature."""
        with torch.no_grad():
            action_probs = self.actor(states)
            action_log_probs = torch.log(action_probs + 1e-8)

        # Compute current entropy
        entropy = -(action_probs * action_log_probs).sum(dim=1).mean()

        # Alpha loss (match current entropy to target entropy)
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy)

        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

        return alpha_loss.item()

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_critic_1.parameters(),
            self.critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic_2.parameters(),
            self.critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save_checkpoint(self, path: Path) -> None:
        """Save SAC agent checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        # Save networks
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, path / 'sac_checkpoint.pt')

        # Save metadata
        metadata = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'framework': 'torchrl_sac',
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'alpha': self.alpha,
            'update_counter': self.update_counter,
        }

        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load SAC agent checkpoint."""
        # Load networks
        checkpoint = torch.load(path / 'sac_checkpoint.pt', map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.total_steps = metadata.get('total_steps', 0)
        self.total_episodes = metadata.get('total_episodes', 0)
        self.update_counter = metadata.get('update_counter', 0)
