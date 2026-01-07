"""Bridge between DQN framework and existing Catan game engine."""

from typing import List, Dict, Any, Tuple
import torch
from pathlib import Path

from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action
from ..game.agents.rl_agent import RLAgent
from ...agents.dqn_agent import DQNAgent, DQNAgentFactory


class DQNGameBridge:
    """
    Bridge that connects the new DQN framework to the existing game engine.

    This allows DQN agents to be used as drop-in replacements for the
    existing RLAgent while maintaining compatibility with the game loop.
    """

    def __init__(self, dqn_agent: DQNAgent, player_id: int):
        self.dqn_agent = dqn_agent
        self.player_id = player_id
        self.last_state = None
        self.last_reward = 0.0

        # Track episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'actions_taken': 0,
            'buildings_built': 0,
            'development_cards': 0
        }

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """
        Select action using DQN agent.

        This method converts the game state for DQN processing and
        handles the action selection.
        """
        # Let DQN agent select action
        selected_action, metrics = self.dqn_agent.select_action(state, legal_actions)

        # Update episode statistics
        self.episode_stats['actions_taken'] += 1

        # Store state for reward processing
        self.last_state = state

        return selected_action

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState):
        """Update the DQN agent with experience."""
        # Accumulate reward
        self.episode_stats['total_reward'] += reward
        self.last_reward = reward

        # Update agent with experience
        if self.last_state is not None:
            # Check if episode is done (someone won)
            done = next_state.winner is not None

            # Let DQN agent observe reward and next state
            self.dqn_agent.observe_reward(reward, next_state, done)

            # Update episode statistics
            self._update_episode_stats(state, next_state)

    def _update_episode_stats(self, state: GameState, next_state: GameState):
        """Update episode statistics for tracking."""
        prev_player = state.players[self.player_id]
        curr_player = next_state.players[self.player_id]

        # Count new buildings
        new_settlements = len(curr_player.settlements) - len(prev_player.settlements)
        new_cities = len(curr_player.cities) - len(prev_player.cities)
        self.episode_stats['buildings_built'] += new_settlements + new_cities

        # Count new development cards
        new_dev_cards = len(curr_player.dev_cards) - len(prev_player.dev_cards)
        self.episode_stats['development_cards'] += new_dev_cards

    def reset(self):
        """Reset for new episode."""
        self.dqn_agent.reset()
        self.last_state = None
        self.last_reward = 0.0

        # Reset episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'actions_taken': 0,
            'buildings_built': 0,
            'development_cards': 0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from both bridge and DQN agent."""
        dqn_metrics = self.dqn_agent.get_metrics()

        return {
            **dqn_metrics,
            'episode_stats': self.episode_stats,
            'last_reward': self.last_reward
        }

    def save_checkpoint(self, path: str):
        """Save DQN agent checkpoint."""
        self.dqn_agent.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Load DQN agent checkpoint."""
        self.dqn_agent.load_checkpoint(path)


class DQNAgentAdapter(RLAgent):
    """
    Adapter that makes DQN agents compatible with existing game infrastructure.

    This class implements the RLAgent interface while using a DQN agent internally.
    """

    def __init__(
        self,
        player_id: int,
        dqn_config_name: str = 'baseline',
        checkpoint_path: str = None,
        **kwargs
    ):
        # Initialize RLAgent with dummy values (won't be used)
        super().__init__(
            player_id=player_id,
            learning_rate=0.1,  # Not used
            epsilon=0.1,       # Not used
            discount=0.95,     # Not used
            seed=kwargs.get('seed'),
            model_path=kwargs.get('model_path')
        )

        # Create DQN agent based on configuration
        self.dqn_agent = self._create_dqn_agent(dqn_config_name, player_id)

        # Create bridge
        self.bridge = DQNGameBridge(self.dqn_agent, player_id)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.bridge.load_checkpoint(checkpoint_path)

    def _create_dqn_agent(self, config_name: str, player_id: int) -> DQNAgent:
        """Create DQN agent based on configuration name."""
        if config_name == 'baseline':
            return DQNAgentFactory.create_baseline_agent(player_id)
        elif config_name == 'spatial':
            return DQNAgentFactory.create_spatial_agent(player_id)
        elif config_name == 'rainbow':
            return DQNAgentFactory.create_rainbow_agent(player_id)
        else:
            raise ValueError(f"Unknown DQN config: {config_name}")

    def select_action(self, state: GameState, legal_actions: List[Action]) -> Action:
        """Select action using DQN agent via bridge."""
        return self.bridge.select_action(state, legal_actions)

    def update(self, state: GameState, action: Action, reward: float, next_state: GameState):
        """Update DQN agent via bridge."""
        self.bridge.update(state, action, reward, next_state)

    def compute_reward(self, state: GameState, next_state: GameState) -> float:
        """Use original reward computation for consistency."""
        # We can use the original RLAgent reward computation
        return super().compute_reward(state, next_state)

    def reset(self):
        """Reset DQN agent via bridge."""
        self.bridge.reset()

    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay epsilon (handled internally by DQN agent)."""
        # DQN agent handles its own epsilon decay
        pass

    def save_model(self):
        """Save DQN model."""
        checkpoint_path = self.model_path.replace('.json', '_dqn')
        self.bridge.save_checkpoint(checkpoint_path)

    def load_model(self):
        """Load DQN model."""
        checkpoint_path = self.model_path.replace('.json', '_dqn')
        if Path(checkpoint_path + '.agent.json').exists():
            self.bridge.load_checkpoint(checkpoint_path)

    def get_dqn_metrics(self) -> Dict[str, Any]:
        """Get DQN-specific metrics."""
        return self.bridge.get_metrics()

    def set_evaluation_mode(self, eval_mode: bool):
        """Set DQN agent to evaluation mode."""
        self.dqn_agent.set_evaluation_mode(eval_mode)


class ExperimentRunner:
    """
    Enhanced experiment runner that supports both DQN and traditional RL agents.

    Provides comprehensive logging and analysis capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            'episodes': [],
            'agent_metrics': {},
            'training_curves': {
                'win_rates': [],
                'average_rewards': [],
                'q_values': [],
                'losses': []
            }
        }

    def run_experiment(
        self,
        agents: List[DQNAgentAdapter],
        num_episodes: int,
        save_frequency: int = 100,
        log_frequency: int = 10
    ) -> Dict[str, Any]:
        """
        Run multi-agent experiment with comprehensive logging.

        Args:
            agents: List of DQN agents to train
            num_episodes: Number of episodes to run
            save_frequency: How often to save checkpoints
            log_frequency: How often to log progress

        Returns:
            Comprehensive results dictionary
        """
        from catan.engine.board import standard_board
        from catan.engine.game_state import initial_game_state
        from catan.engine.rules import apply_action

        print(f"Starting experiment with {len(agents)} agents for {num_episodes} episodes")

        wins_per_agent = {agent.player_id: 0 for agent in agents}
        episode_rewards = {agent.player_id: [] for agent in agents}

        for episode in range(num_episodes):
            # Reset environment
            board = standard_board(seed=episode)
            game_state = initial_game_state(board, num_players=len(agents))

            # Reset agents
            for agent in agents:
                agent.reset()

            episode_length = 0
            max_turns = 1000  # Prevent infinite games

            # Run episode
            while game_state.winner is None and episode_length < max_turns:
                current_player_id = game_state.current_player
                current_agent = agents[current_player_id]

                # Get legal actions
                legal_actions = game_state.legal_actions()
                if not legal_actions:
                    break

                # Agent selects action
                action = current_agent.select_action(game_state, legal_actions)

                # Apply action
                prev_state = game_state
                game_state = apply_action(game_state, action)

                # Compute reward and update agent
                reward = current_agent.compute_reward(prev_state, game_state)
                current_agent.update(prev_state, action, reward, game_state)

                # Track rewards
                episode_rewards[current_player_id].append(reward)

                episode_length += 1

            # Record episode results
            if game_state.winner is not None:
                wins_per_agent[game_state.winner] += 1

            # Collect episode data
            episode_data = {
                'episode': episode,
                'winner': game_state.winner,
                'episode_length': episode_length,
                'agent_rewards': {aid: sum(episode_rewards[aid]) for aid in episode_rewards},
                'agent_metrics': {aid: agents[aid].get_dqn_metrics() for aid in range(len(agents))}
            }
            self.results['episodes'].append(episode_data)

            # Update training curves
            if episode > 0:
                # Calculate rolling win rates
                window = min(50, episode + 1)
                recent_wins = sum(
                    1 for ep in self.results['episodes'][-window:]
                    if ep['winner'] == 0  # Track agent 0's performance
                )
                win_rate = recent_wins / window
                self.results['training_curves']['win_rates'].append(win_rate)

                # Track other metrics for agent 0
                agent_0_metrics = episode_data['agent_metrics'][0]
                self.results['training_curves']['average_rewards'].append(
                    episode_data['agent_rewards'][0]
                )
                self.results['training_curves']['q_values'].append(
                    agent_0_metrics.get('avg_q_value', 0)
                )
                self.results['training_curves']['losses'].append(
                    agent_0_metrics.get('avg_loss', 0)
                )

            # Logging
            if episode % log_frequency == 0:
                win_rate = wins_per_agent[0] / (episode + 1)
                print(f"Episode {episode}: Win rate = {win_rate:.3f}, "
                      f"Episode length = {episode_length}")

            # Save checkpoints
            if save_frequency > 0 and episode % save_frequency == 0:
                for agent in agents:
                    checkpoint_path = f"checkpoints/agent_{agent.player_id}_ep_{episode}"
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    agent.save_model()

            # Clear episode rewards for next episode
            episode_rewards = {agent.player_id: [] for agent in agents}

        # Final results
        self.results['final_metrics'] = {
            'total_episodes': num_episodes,
            'wins_per_agent': wins_per_agent,
            'win_rates': {aid: wins_per_agent[aid] / num_episodes for aid in wins_per_agent},
            'final_agent_metrics': {aid: agents[aid].get_dqn_metrics() for aid in range(len(agents))}
        }

        return self.results

    def save_results(self, path: str):
        """Save experiment results to file."""
        import json

        # Convert torch tensors to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def _make_json_serializable(self, obj):
        """Convert torch tensors and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert objects to strings
        else:
            return obj