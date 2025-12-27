"""Experience replay buffer implementations for DQN training."""

from typing import Dict, Any, Tuple, Optional, List, NamedTuple
import torch
import numpy as np
from collections import deque
import random
from abc import ABC, abstractmethod
import pickle


class Experience(NamedTuple):
    """Single experience tuple."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    action_mask: Optional[torch.Tensor] = None
    next_action_mask: Optional[torch.Tensor] = None
    info: Optional[Dict[str, Any]] = None


class BaseReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(self, capacity: int, device: torch.device = None):
        self.capacity = capacity
        self.device = device or torch.device('cpu')
        self.size = 0

    @abstractmethod
    def add(self, experience: Experience):
        """Add experience to buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all experiences."""
        pass

    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size


class UniformReplayBuffer(BaseReplayBuffer):
    """Standard uniform random sampling replay buffer."""

    def __init__(self, capacity: int, device: torch.device = None):
        super().__init__(capacity, device)
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.size = len(self.buffer)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences uniformly."""
        if not self.can_sample(batch_size):
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {self.size}")

        experiences = random.sample(self.buffer, batch_size)
        return self._experiences_to_batch(experiences)

    def _experiences_to_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Convert list of experiences to batch tensors."""
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)

        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        # Add action masks if available
        if experiences[0].action_mask is not None:
            action_masks = torch.stack([exp.action_mask for exp in experiences]).to(self.device)
            batch['action_masks'] = action_masks

        if experiences[0].next_action_mask is not None:
            next_action_masks = torch.stack([exp.next_action_mask for exp in experiences]).to(self.device)
            batch['next_action_masks'] = next_action_masks

        return batch

    def clear(self):
        """Clear all experiences."""
        self.buffer.clear()
        self.size = 0


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 100000,
        device: torch.device = None
    ):
        super().__init__(capacity, device)
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling exponent start value
        self.beta_steps = beta_steps  # Steps to anneal beta to 1.0
        self.current_step = 0

        # Sum tree for efficient priority sampling
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2

        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, float('inf'))
        self.data = [None] * capacity
        self.data_pointer = 0
        self.max_priority = 1.0

    def add(self, experience: Experience):
        """Add experience with maximum priority."""
        tree_index = self.data_pointer + self.tree_capacity - 1

        self.data[self.data_pointer] = experience
        self.update_priority(tree_index, self.max_priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update_priority(self, tree_index: int, priority: float):
        """Update priority in sum tree."""
        priority = max(priority, 1e-8)  # Avoid zero priority
        self.max_priority = max(self.max_priority, priority)

        change = priority - self.sum_tree[tree_index]
        self.sum_tree[tree_index] = priority
        self.min_tree[tree_index] = priority

        # Propagate changes up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.sum_tree[tree_index] += change
            self.min_tree[tree_index] = min(
                self.min_tree[2 * tree_index + 1],
                self.min_tree[2 * tree_index + 2]
            )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with importance sampling weights."""
        if not self.can_sample(batch_size):
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {self.size}")

        indices = []
        priorities = []
        segment = self.sum_tree[0] / batch_size

        # Current beta for importance sampling
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.current_step / self.beta_steps)
        self.current_step += 1

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)
            index, priority = self._retrieve(value)
            indices.append(index)
            priorities.append(priority)

        # Convert to data indices
        data_indices = [idx - self.tree_capacity + 1 for idx in indices]
        experiences = [self.data[idx] for idx in data_indices]

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.sum_tree[0]
        min_prob = self.min_tree[0] / self.sum_tree[0]
        weights = (sampling_probabilities / min_prob) ** (-beta)
        weights = weights / weights.max()  # Normalize

        batch = self._experiences_to_batch(experiences)
        batch['weights'] = torch.tensor(weights, dtype=torch.float).to(self.device)
        batch['indices'] = indices  # For priority updates

        return batch

    def _retrieve(self, value: float) -> Tuple[int, float]:
        """Retrieve sample index from sum tree."""
        index = 0
        while index < self.tree_capacity - 1:
            left = 2 * index + 1
            if value <= self.sum_tree[left]:
                index = left
            else:
                value -= self.sum_tree[left]
                index = left + 1

        return index, self.sum_tree[index]

    def update_priorities(self, indices: List[int], td_errors: torch.Tensor):
        """Update priorities based on TD errors."""
        priorities = (td_errors.abs() + 1e-8) ** self.alpha
        for idx, priority in zip(indices, priorities.cpu().numpy()):
            self.update_priority(idx, priority)

    def _experiences_to_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Convert experiences to batch (same as UniformReplayBuffer)."""
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)

        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        if experiences[0].action_mask is not None:
            action_masks = torch.stack([exp.action_mask for exp in experiences]).to(self.device)
            batch['action_masks'] = action_masks

        if experiences[0].next_action_mask is not None:
            next_action_masks = torch.stack([exp.next_action_mask for exp in experiences]).to(self.device)
            batch['next_action_masks'] = next_action_masks

        return batch

    def clear(self):
        """Clear all experiences."""
        self.data = [None] * self.capacity
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, float('inf'))
        self.data_pointer = 0
        self.size = 0
        self.max_priority = 1.0


class EpisodicReplayBuffer(BaseReplayBuffer):
    """Replay buffer that stores complete episodes."""

    def __init__(self, capacity: int, device: torch.device = None):
        super().__init__(capacity, device)
        self.episodes = deque()
        self.current_episode = []
        self.episode_count = 0

    def add(self, experience: Experience):
        """Add experience to current episode."""
        self.current_episode.append(experience)

        # End episode if done
        if experience.done:
            self.end_episode()

    def end_episode(self):
        """Finish current episode and add to buffer."""
        if self.current_episode:
            self.episodes.append(self.current_episode.copy())
            self.current_episode.clear()
            self.episode_count += 1

            # Remove oldest episodes if over capacity
            while len(self.episodes) > self.capacity:
                removed_episode = self.episodes.popleft()
                self.size -= len(removed_episode)

            # Update size
            self.size = sum(len(episode) for episode in self.episodes)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample experiences from random episodes."""
        if not self.can_sample(batch_size):
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {self.size}")

        experiences = []
        while len(experiences) < batch_size:
            # Sample random episode
            episode = random.choice(self.episodes)
            # Sample random experience from episode
            experience = random.choice(episode)
            experiences.append(experience)

        return self._experiences_to_batch(experiences)

    def sample_episodes(self, num_episodes: int) -> List[List[Experience]]:
        """Sample complete episodes."""
        if num_episodes > len(self.episodes):
            raise ValueError(f"Cannot sample {num_episodes} episodes from {len(self.episodes)} available")

        return random.sample(self.episodes, num_episodes)

    def _experiences_to_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Convert experiences to batch."""
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)

        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        if experiences[0].action_mask is not None:
            action_masks = torch.stack([exp.action_mask for exp in experiences]).to(self.device)
            batch['action_masks'] = action_masks

        if experiences[0].next_action_mask is not None:
            next_action_masks = torch.stack([exp.next_action_mask for exp in experiences]).to(self.device)
            batch['next_action_masks'] = next_action_masks

        return batch

    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()
        self.current_episode.clear()
        self.size = 0
        self.episode_count = 0

    def get_episode_returns(self) -> List[float]:
        """Get returns for all stored episodes."""
        returns = []
        for episode in self.episodes:
            episode_return = sum(exp.reward for exp in episode)
            returns.append(episode_return)
        return returns


class NStepReplayBuffer(UniformReplayBuffer):
    """Replay buffer with n-step returns."""

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        device: torch.device = None
    ):
        super().__init__(capacity, device)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, experience: Experience):
        """Add experience and compute n-step returns."""
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_return = 0.0
            for i, exp in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * exp.reward
                if exp.done:
                    break

            # Create n-step experience
            first_exp = self.n_step_buffer[0]
            last_exp = self.n_step_buffer[-1]

            n_step_experience = Experience(
                state=first_exp.state,
                action=first_exp.action,
                reward=n_step_return,
                next_state=last_exp.next_state,
                done=any(exp.done for exp in self.n_step_buffer),
                action_mask=first_exp.action_mask,
                next_action_mask=last_exp.next_action_mask,
                info={'n_step': self.n_step}
            )

            super().add(n_step_experience)


class ReplayBufferFactory:
    """Factory for creating different replay buffer types."""

    _buffers = {
        'uniform': UniformReplayBuffer,
        'prioritized': PrioritizedReplayBuffer,
        'episodic': EpisodicReplayBuffer,
        'n_step': NStepReplayBuffer
    }

    @classmethod
    def create(
        cls,
        buffer_type: str,
        capacity: int,
        device: torch.device = None,
        **kwargs
    ) -> BaseReplayBuffer:
        """Create replay buffer of specified type."""
        if buffer_type not in cls._buffers:
            raise ValueError(f"Unknown buffer type: {buffer_type}. Available: {list(cls._buffers.keys())}")

        return cls._buffers[buffer_type](capacity, device=device, **kwargs)

    @classmethod
    def list_buffers(cls) -> List[str]:
        """List all available buffer types."""
        return list(cls._buffers.keys())