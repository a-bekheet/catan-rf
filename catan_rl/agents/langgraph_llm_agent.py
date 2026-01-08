"""
LangGraph LLM Agent for Catan
==============================

Uses LangGraph workflow with LLM reasoning for strategic Catan gameplay.
This agent combines:
- LLM strategic reasoning (GPT-4/Claude)
- LangGraph state management and workflows
- Reinforcement learning from game outcomes
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
import json
import os

try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain/LangGraph not installed. Install with: pip install langchain langgraph")

from .base_rl_agent import BaseRLAgent, AgentMetrics
from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action, ActionType
from ..environments.action_space import CatanActionSpace


class LangGraphLLMAgent(BaseRLAgent):
    """
    LangGraph-powered LLM agent for Catan.

    This agent uses:
    - LLM reasoning for strategic decision-making
    - LangGraph for structured workflows
    - Experience tracking for learning from outcomes
    """

    def __init__(
        self,
        agent_id: int,
        config: Dict[str, Any]
    ):
        super().__init__(agent_id, "LangGraph LLM Agent", config)

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain/LangGraph required for LLMAgent. "
                "Install with: pip install langchain langgraph openai anthropic"
            )

        # LLM configuration
        self.llm_provider = config.get('llm_provider', 'openai')  # or 'anthropic'
        self.model_name = config.get('model_name', 'gpt-4-turbo-preview')
        self.temperature = config.get('temperature', 0.7)

        # Initialize LLM
        self._initialize_llm()

        # Action space for encoding/decoding
        self.action_space = CatanActionSpace(config.get('action_space', {}))

        # Experience memory for learning
        self.experience_memory = []
        self.max_memory_size = config.get('max_memory_size', 100)

        # Strategy preferences (learned from experience)
        self.strategy_preferences = {
            'aggressive_expansion': 0.5,
            'resource_hoarding': 0.5,
            'development_cards': 0.5,
            'trading_frequency': 0.5,
        }

    def _initialize_llm(self) -> None:
        """Initialize the LLM based on provider."""
        if self.llm_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Using mock mode.")
                self.llm = None
                self.mock_mode = True
            else:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=api_key
                )
                self.mock_mode = False

        elif self.llm_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY not found. Using mock mode.")
                self.llm = None
                self.mock_mode = True
            else:
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=api_key
                )
                self.mock_mode = False
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def select_action(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Tuple[Action, AgentMetrics]:
        """Select action using LLM reasoning."""
        start_time = time.time()

        if self.mock_mode or self.llm is None:
            # Fallback to heuristic-based selection
            return self._select_heuristic_action(game_state, legal_actions, start_time)

        try:
            # Create context for LLM
            context = self._create_game_context(game_state, legal_actions)

            # Build prompt
            prompt = self._build_decision_prompt(context)

            # Get LLM response
            response = self.llm.invoke(prompt)
            action_description = response.content

            # Parse LLM response to select action
            selected_action = self._parse_llm_action(
                action_description,
                legal_actions,
                game_state
            )

            # Fallback to heuristic if parsing failed
            if selected_action is None:
                selected_action = self._select_heuristic_action(
                    game_state,
                    legal_actions,
                    start_time
                )[0]

            metrics = AgentMetrics(
                decision_time=time.time() - start_time,
                confidence=0.8,  # High confidence for LLM decisions
                exploration_rate=self.temperature,
                additional_data={
                    'llm_response': action_description[:200],
                    'provider': self.llm_provider,
                }
            )

            self.total_steps += 1
            return selected_action, metrics

        except Exception as e:
            print(f"LLM error: {e}. Falling back to heuristic.")
            return self._select_heuristic_action(game_state, legal_actions, start_time)

    def _create_game_context(
        self,
        game_state: GameState,
        legal_actions: List[Action]
    ) -> Dict[str, Any]:
        """Create context dictionary describing the game state."""
        player = game_state.players[self.agent_id]

        # Get opponent info
        opponents = []
        for pid, opp in game_state.players.items():
            if pid != self.agent_id:
                opponents.append({
                    'id': pid,
                    'vp': opp.victory_points,
                    'settlements': len(opp.settlements),
                    'cities': len(opp.cities),
                    'roads': len(opp.roads),
                })

        # Categorize legal actions
        action_categories = {}
        for action in legal_actions:
            category = action.action_type.value
            if category not in action_categories:
                action_categories[category] = 0
            action_categories[category] += 1

        return {
            'my_vp': player.victory_points,
            'my_resources': dict(player.resources),
            'my_buildings': {
                'settlements': len(player.settlements),
                'cities': len(player.cities),
                'roads': len(player.roads),
            },
            'my_dev_cards': len(player.dev_cards),
            'opponents': opponents,
            'turn_number': game_state.turn_number,
            'available_actions': action_categories,
            'phase': game_state.phase.value,
        }

    def _build_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM decision-making."""
        prompt = f"""You are an expert Settlers of Catan player. Analyze the current game state and choose the best action.

GAME STATE:
- Your Victory Points: {context['my_vp']} (need 10 to win)
- Your Resources: {context['my_resources']}
- Your Buildings: {context['my_buildings']}
- Development Cards: {context['my_dev_cards']}
- Turn: {context['turn_number']}
- Phase: {context['phase']}

OPPONENTS:
{self._format_opponents(context['opponents'])}

AVAILABLE ACTIONS:
{self._format_actions(context['available_actions'])}

STRATEGY PREFERENCES (learned from experience):
- Aggressive Expansion: {self.strategy_preferences['aggressive_expansion']:.2f}
- Resource Hoarding: {self.strategy_preferences['resource_hoarding']:.2f}
- Development Cards: {self.strategy_preferences['development_cards']:.2f}
- Trading: {self.strategy_preferences['trading_frequency']:.2f}

Based on this information, what type of action should you take? Choose ONE:
1. BUILD_SETTLEMENT - Expand your territory
2. BUILD_CITY - Upgrade for more resources
3. BUILD_ROAD - Connect to new areas
4. BUY_DEV_CARD - Get development cards
5. TRADE - Exchange resources
6. END_TURN - Complete your turn
7. ROLL_DICE - Roll for resources

Respond with ONLY the action type (e.g., "BUILD_SETTLEMENT"). No explanation needed."""

        return prompt

    def _format_opponents(self, opponents: List[Dict]) -> str:
        """Format opponent information."""
        lines = []
        for opp in opponents:
            lines.append(
                f"  Player {opp['id']}: {opp['vp']} VP, "
                f"{opp['settlements']} settlements, {opp['cities']} cities, "
                f"{opp['roads']} roads"
            )
        return "\n".join(lines)

    def _format_actions(self, actions: Dict[str, int]) -> str:
        """Format available actions."""
        lines = []
        for action_type, count in actions.items():
            lines.append(f"  {action_type}: {count} options")
        return "\n".join(lines)

    def _parse_llm_action(
        self,
        llm_response: str,
        legal_actions: List[Action],
        game_state: GameState
    ) -> Optional[Action]:
        """Parse LLM response to select a specific action."""
        # Extract action type from response
        response_upper = llm_response.upper()

        # Map response to action type
        action_type_map = {
            'BUILD_SETTLEMENT': ActionType.BUILD_SETTLEMENT,
            'BUILD_CITY': ActionType.BUILD_CITY,
            'BUILD_ROAD': ActionType.BUILD_ROAD,
            'BUY_DEV_CARD': ActionType.BUY_DEV_CARD,
            'TRADE': ActionType.PROPOSE_TRADE,
            'END_TURN': ActionType.END_TURN,
            'ROLL_DICE': ActionType.ROLL_DICE,
        }

        # Find matching action type
        target_type = None
        for key, action_type in action_type_map.items():
            if key in response_upper:
                target_type = action_type
                break

        if target_type is None:
            return None

        # Filter legal actions by type
        matching_actions = [a for a in legal_actions if a.action_type == target_type]

        if not matching_actions:
            return None

        # Use heuristic to pick best among matching actions
        return self._select_best_action_of_type(matching_actions, game_state)

    def _select_best_action_of_type(
        self,
        actions: List[Action],
        game_state: GameState
    ) -> Action:
        """Select best action from a list of same-type actions."""
        # Simple heuristic: prefer actions that increase VP or resource production
        if not actions:
            return actions[0] if actions else None

        # For now, just pick the first one (can be enhanced with more heuristics)
        return actions[0]

    def _select_heuristic_action(
        self,
        game_state: GameState,
        legal_actions: List[Action],
        start_time: float
    ) -> Tuple[Action, AgentMetrics]:
        """Fallback heuristic action selection."""
        player = game_state.players[self.agent_id]

        # Priority order based on strategy
        priorities = [
            ActionType.BUILD_SETTLEMENT,  # Highest priority
            ActionType.BUILD_CITY,
            ActionType.BUY_DEV_CARD,
            ActionType.BUILD_ROAD,
            ActionType.PROPOSE_TRADE,
            ActionType.END_TURN,
        ]

        # Select action based on priority
        for priority_type in priorities:
            matching_actions = [a for a in legal_actions if a.action_type == priority_type]
            if matching_actions:
                selected_action = matching_actions[0]
                metrics = AgentMetrics(
                    decision_time=time.time() - start_time,
                    confidence=0.6,
                    exploration_rate=0.0,
                    additional_data={'method': 'heuristic'}
                )
                self.total_steps += 1
                return selected_action, metrics

        # Last resort: random action
        import random
        selected_action = random.choice(legal_actions)
        metrics = AgentMetrics(
            decision_time=time.time() - start_time,
            confidence=0.3,
            exploration_rate=1.0,
            additional_data={'method': 'random_fallback'}
        )
        self.total_steps += 1
        return selected_action, metrics

    def update(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """Update agent based on experience (learn strategy preferences)."""
        # Store experience
        experience = {
            'action_type': action.action_type.value,
            'reward': reward,
            'vp_change': next_state.players[self.agent_id].victory_points -
                        state.players[self.agent_id].victory_points,
            'done': done,
        }
        self.experience_memory.append(experience)

        # Keep memory bounded
        if len(self.experience_memory) > self.max_memory_size:
            self.experience_memory.pop(0)

        # Update strategy preferences based on outcomes
        if done:
            self._update_strategy_preferences()

        return {
            'avg_reward': sum(e['reward'] for e in self.experience_memory[-10:]) / min(10, len(self.experience_memory))
        }

    def _update_strategy_preferences(self) -> None:
        """Update strategy preferences based on recent experiences."""
        if len(self.experience_memory) < 10:
            return

        recent_experiences = self.experience_memory[-50:]

        # Analyze which actions led to positive outcomes
        action_rewards = {}
        for exp in recent_experiences:
            action_type = exp['action_type']
            if action_type not in action_rewards:
                action_rewards[action_type] = []
            action_rewards[action_type].append(exp['reward'])

        # Adjust preferences based on action success
        for action_type, rewards in action_rewards.items():
            avg_reward = sum(rewards) / len(rewards)

            # Map actions to strategy preferences
            if action_type in ['BUILD_SETTLEMENT', 'BUILD_CITY']:
                self.strategy_preferences['aggressive_expansion'] += avg_reward * 0.01
            elif action_type in ['BUY_DEV_CARD', 'PLAY_DEV_CARD']:
                self.strategy_preferences['development_cards'] += avg_reward * 0.01
            elif action_type in ['PROPOSE_TRADE', 'ACCEPT_TRADE']:
                self.strategy_preferences['trading_frequency'] += avg_reward * 0.01

        # Normalize preferences to [0, 1]
        for key in self.strategy_preferences:
            self.strategy_preferences[key] = max(0.0, min(1.0, self.strategy_preferences[key]))

    def save_checkpoint(self, path: Path) -> None:
        """Save agent state."""
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'framework': 'langgraph_llm',
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'strategy_preferences': self.strategy_preferences,
            'experience_memory': self.experience_memory[-100:],  # Save last 100
            'llm_provider': self.llm_provider,
            'model_name': self.model_name,
        }

        with open(path / 'llm_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        """Load agent state."""
        with open(path / 'llm_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)

        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.strategy_preferences = checkpoint.get(
            'strategy_preferences',
            self.strategy_preferences
        )
        self.experience_memory = checkpoint.get('experience_memory', [])
