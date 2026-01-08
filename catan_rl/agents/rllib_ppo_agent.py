"""
Ray RLlib PPO Agent for Catan
==============================

Wraps Ray RLlib's PPO algorithm for Catan gameplay.
PPO (Proximal Policy Optimization) is great for:
- Fast distributed training
- Stable policy updates
- Multi-agent environments
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
import json
import numpy as np

try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import Policy
    import ray
    from ray import tune
    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False
    print("Warning: Ray RLlib not installed. Install with: pip install ray[rllib]")

from .base_rl_agent import BaseRLAgent, AgentMetrics
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action
from ..environments.action_space import CatanActionSpace
from ..environments.state_encoders.base import StateEncoderFactory


class RLlibPPOAgent(BaseRLAgent):
    """
    Ray RLlib PPO agent wrapper for Catan.

    This agent uses Proximal Policy Optimization (PPO), which:
    - Uses clipped surrogate objective for stable updates
    - Supports distributed training across multiple CPUs/GPUs
    - Excellent for complex multi-agent games like Catan
    """

    def __init__(
        self,
        agent_id: int,
        config: Dict[str, Any]
    ):
        super().__init__(agent_id, "RLlib PPO Agent", config)

        if not RLLIB_AVAILABLE:
            raise ImportError(
                "Ray RLlib is required for PPOAgent. "
                "Install with: pip install ray[rllib]"
            )

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=config.get('num_cpus', 4),
                num_gpus=config.get('num_gpus', 0)
            )

        # Setup state encoder and action space
        encoder_config = config.get('state_encoder', {'type': 'feature'})
        self.state_encoder = StateEncoderFactory.create(
            encoder_config.get('type', 'feature'),
            encoder_config.get('params', {})
        )
        self.action_space = CatanActionSpace(config.get('action_space', {}))

        # Create PPO configuration
        self.ppo_config = self._create_ppo_config(config)

        # Build the algorithm
        self.algorithm = None
        self.policy = None
        self._initialize_algorithm()

    def _create_ppo_config(self, config: Dict[str, Any]) -> "PPOConfig":
        """Create PPO configuration from agent config."""
        state_shape = self.state_encoder.get_state_shape()
        action_dim = self.action_space.get_action_space_size()

        # Get state and action space dimensions
        if isinstance(state_shape, tuple):
            obs_space_size = int(np.prod(state_shape))
        else:
            obs_space_size = state_shape

        ppo_config = PPOConfig()

        # Environment config
        ppo_config = ppo_config.environment(
            observation_space=self._get_observation_space(state_shape),
            action_space=self._get_action_space(action_dim)
        )

        # Training config
        ppo_config = ppo_config.training(
            lr=config.get('learning_rate', 3e-4),
            gamma=config.get('gamma', 0.99),
            lambda_=config.get('lambda', 0.95),
            clip_param=config.get('clip_param', 0.2),
            entropy_coeff=config.get('entropy_coeff', 0.01),
            vf_loss_coeff=config.get('vf_loss_coeff', 0.5),
            train_batch_size=config.get('train_batch_size', 4000),
            sgd_minibatch_size=config.get('sgd_minibatch_size', 128),
            num_sgd_iter=config.get('num_sgd_iter', 10),
        )

        # Resources
        ppo_config = ppo_config.resources(
            num_gpus=config.get('num_gpus', 0)
        )

        # Rollout config
        ppo_config = ppo_config.rollouts(
            num_rollout_workers=config.get('num_workers', 2),
            rollout_fragment_length=config.get('rollout_fragment_length', 200)
        )

        # Model config
        model_config = {
            'fcnet_hiddens': config.get('hidden_layers', [256, 256]),
            'fcnet_activation': config.get('activation', 'relu'),
            'use_lstm': config.get('use_lstm', False),
        }
        ppo_config = ppo_config.framework('torch')
        ppo_config.model = model_config

        return ppo_config

    def _get_observation_space(self, state_shape):
        """Create Gym observation space from state shape."""
        from gymnasium.spaces import Box
        if isinstance(state_shape, tuple):
            return Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
        else:
            return Box(low=-np.inf, high=np.inf, shape=(state_shape,), dtype=np.float32)

    def _get_action_space(self, action_dim: int):
        """Create Gym action space."""
        from gymnasium.spaces import Discrete
        return Discrete(action_dim)

    def _initialize_algorithm(self) -> None:
        """Initialize the PPO algorithm."""
        self.algorithm = self.ppo_config.build()
        self.policy = self.algorithm.get_policy()

    def select_action(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Tuple[Action, AgentMetrics]:
        """Select action using PPO policy."""
        start_time = time.time()

        # Encode state
        state_array = self.state_encoder.encode(game_state, self.agent_id)

        # Create action mask for legal actions
        action_mask = self._create_action_mask(legal_actions)

        # Compute action from policy
        if self.is_training:
            # During training, use exploration
            action_idx, _, info = self.policy.compute_single_action(
                state_array,
                explore=True,
                policy_id="default_policy"
            )
        else:
            # During evaluation, use greedy policy
            action_idx, _, info = self.policy.compute_single_action(
                state_array,
                explore=False,
                policy_id="default_policy"
            )

        # Ensure action is legal
        action_idx = int(action_idx)
        if action_mask[action_idx] == 0:
            # Fallback to random legal action if policy chose illegal action
            legal_indices = np.where(action_mask == 1)[0]
            action_idx = np.random.choice(legal_indices)

        # Decode action
        selected_action = self.action_space.decode_action(action_idx)
        if selected_action is None or selected_action not in legal_actions:
            selected_action = np.random.choice(legal_actions)

        # Create metrics
        metrics = AgentMetrics(
            decision_time=time.time() - start_time,
            confidence=info.get('action_prob', 0.0) if info else 0.0,
            exploration_rate=0.0,  # PPO doesn't use epsilon-greedy
            policy_entropy=info.get('entropy', None) if info else None,
            additional_data={
                'action_idx': action_idx,
                'action_logp': info.get('action_logp', None) if info else None,
            }
        )

        self.total_steps += 1
        return selected_action, metrics

    def _create_action_mask(self, legal_actions: List[Action]) -> np.ndarray:
        """Create binary mask for legal actions."""
        action_space_size = self.action_space.get_action_space_size()
        mask = np.zeros(action_space_size, dtype=np.float32)

        for action in legal_actions:
            action_idx = self.action_space.encode_action(action)
            if action_idx < action_space_size:
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
        """
        Update PPO policy.

        Note: RLlib handles experience collection and training internally,
        so this method is mostly for compatibility with the base interface.
        """
        # RLlib's PPO handles training internally during rollouts
        # We'll return metrics from the last training iteration
        if hasattr(self.algorithm, 'train'):
            result = self.algorithm.train()
            return {
                'policy_loss': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('policy_loss', 0.0),
                'vf_loss': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('vf_loss', 0.0),
                'entropy': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('entropy', 0.0),
            }

        return None

    def save_checkpoint(self, path: Path) -> None:
        """Save PPO agent checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        # Save RLlib checkpoint
        checkpoint_path = self.algorithm.save(str(path))

        # Save agent metadata
        metadata = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'framework': 'rllib_ppo',
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'checkpoint_path': str(checkpoint_path),
        }

        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load PPO agent checkpoint."""
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Restore RLlib checkpoint
        checkpoint_path = metadata.get('checkpoint_path', str(path))
        self.algorithm.restore(checkpoint_path)

        # Restore agent state
        self.total_steps = metadata.get('total_steps', 0)
        self.total_episodes = metadata.get('total_episodes', 0)

    def __del__(self):
        """Cleanup when agent is destroyed."""
        if hasattr(self, 'algorithm') and self.algorithm:
            self.algorithm.stop()
