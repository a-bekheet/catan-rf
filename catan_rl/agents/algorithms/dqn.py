"""Deep Q-Network (DQN) algorithm implementations."""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
import copy

from ..networks.base import BaseQNetwork


class BaseDQN(ABC):
    """Base class for DQN variants."""

    def __init__(
        self,
        q_network: BaseQNetwork,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Network setup
        self.q_network = q_network.to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()

        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.target_update_frequency = self.config.get('target_update_frequency', 1000)
        self.gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)

        # Optimizer
        optimizer_type = self.config.get('optimizer', 'adam')
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        if self.config.get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 10000),
                gamma=self.config.get('lr_decay', 0.9)
            )
        else:
            self.scheduler = None

        # Training state
        self.training_steps = 0
        self.episode_count = 0

    @abstractmethod
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for the specific DQN variant."""
        pass

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0
    ) -> int:
        """Select action using epsilon-greedy policy."""
        return self.q_network.get_action(state, action_mask, epsilon)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the network with a batch of experiences."""
        self.q_network.train()

        # Extract batch data
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        action_masks = batch.get('action_masks')
        next_action_masks = batch.get('next_action_masks')

        if action_masks is not None:
            action_masks = action_masks.to(self.device)
        if next_action_masks is not None:
            next_action_masks = next_action_masks.to(self.device)

        # Compute loss
        loss, metrics = self.compute_loss(
            states, actions, rewards, next_states, dones,
            action_masks, next_action_masks
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.gradient_clip_norm
            )

        self.optimizer.step()

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()

        # Add training metrics
        metrics.update({
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_steps': self.training_steps
        })

        return metrics

    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episode_count = checkpoint['episode_count']


class VanillaDQN(BaseDQN):
    """Standard Deep Q-Network implementation."""

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute standard DQN loss."""
        batch_size = states.size(0)

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states, action_masks)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values: max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_action_masks)

            # Apply action masking to next states
            if next_action_masks is not None:
                next_q_values = next_q_values.masked_fill(next_action_masks == 0, float('-inf'))

            max_next_q_values = next_q_values.max(1)[0]

            # Target values: r + gamma * max Q(s', a') * (1 - done)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Metrics
        metrics = {
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'target_mean': target_q_values.mean().item(),
            'td_error_mean': (target_q_values - current_q_values).abs().mean().item()
        }

        return loss, metrics


class DoubleDQN(BaseDQN):
    """Double DQN to reduce overestimation bias."""

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Double DQN loss."""
        batch_size = states.size(0)

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states, action_masks)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network for action selection, target network for evaluation
        with torch.no_grad():
            # Online network selects actions
            online_next_q_values = self.q_network(next_states, next_action_masks)

            # Apply action masking
            if next_action_masks is not None:
                online_next_q_values = online_next_q_values.masked_fill(
                    next_action_masks == 0, float('-inf')
                )

            next_actions = online_next_q_values.argmax(1)

            # Target network evaluates selected actions
            target_next_q_values = self.target_network(next_states, next_action_masks)
            max_next_q_values = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Target values
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Metrics
        metrics = {
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'target_mean': target_q_values.mean().item(),
            'td_error_mean': (target_q_values - current_q_values).abs().mean().item(),
            'action_distribution': torch.bincount(next_actions).float()
        }

        return loss, metrics


class DuelingDQN(BaseDQN):
    """DQN with dueling architecture (handled by the network itself)."""

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Dueling DQN loss (same as vanilla, architecture handles the rest)."""
        return VanillaDQN.compute_loss(
            self, states, actions, rewards, next_states, dones,
            action_masks, next_action_masks
        )


class RainbowDQN(BaseDQN):
    """Rainbow DQN combining multiple improvements."""

    def __init__(self, q_network: BaseQNetwork, config: Dict[str, Any] = None):
        super().__init__(q_network, config)

        # Additional Rainbow components
        self.use_noisy_networks = config.get('use_noisy_networks', False)
        self.use_categorical = config.get('use_categorical', False)
        self.n_step_returns = config.get('n_step_returns', 1)

        # Prioritized replay parameters (handled by replay buffer)
        self.use_prioritized_replay = config.get('use_prioritized_replay', False)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        next_action_masks: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        n_step_rewards: Optional[torch.Tensor] = None,
        n_step_next_states: Optional[torch.Tensor] = None,
        n_step_dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Rainbow DQN loss with n-step returns."""
        batch_size = states.size(0)

        # Use n-step returns if available
        if n_step_rewards is not None:
            actual_rewards = n_step_rewards
            actual_next_states = n_step_next_states
            actual_dones = n_step_dones
            gamma_n = self.gamma ** self.n_step_returns
        else:
            actual_rewards = rewards
            actual_next_states = next_states
            actual_dones = dones
            gamma_n = self.gamma

        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states, action_masks)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            # Online network selects actions
            online_next_q_values = self.q_network(actual_next_states, next_action_masks)

            if next_action_masks is not None:
                online_next_q_values = online_next_q_values.masked_fill(
                    next_action_masks == 0, float('-inf')
                )

            next_actions = online_next_q_values.argmax(1)

            # Target network evaluates
            target_next_q_values = self.target_network(actual_next_states, next_action_masks)
            max_next_q_values = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Target values with n-step discount
            target_q_values = actual_rewards + gamma_n * max_next_q_values * (1 - actual_dones)

        # Compute TD errors
        td_errors = target_q_values - current_q_values

        # Weighted loss for prioritized replay
        if weights is not None:
            loss = (weights * td_errors.pow(2)).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)

        # Metrics
        metrics = {
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'target_mean': target_q_values.mean().item(),
            'td_error_mean': td_errors.abs().mean().item(),
            'td_errors': td_errors.detach()  # For prioritized replay updates
        }

        if self.use_noisy_networks:
            # Reset noise in noisy networks
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        return loss, metrics


class DQNFactory:
    """Factory for creating different DQN variants."""

    _algorithms = {
        'vanilla': VanillaDQN,
        'double': DoubleDQN,
        'dueling': DuelingDQN,
        'rainbow': RainbowDQN
    }

    @classmethod
    def create(
        cls,
        algorithm_type: str,
        q_network: BaseQNetwork,
        config: Dict[str, Any] = None
    ) -> BaseDQN:
        """Create DQN algorithm of specified type."""
        if algorithm_type not in cls._algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}. Available: {list(cls._algorithms.keys())}")

        return cls._algorithms[algorithm_type](q_network, config)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all available algorithm types."""
        return list(cls._algorithms.keys())