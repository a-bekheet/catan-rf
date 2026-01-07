"""Action space management for Catan RL environment."""

from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from enum import Enum

from catan_rl.core.game.engine.game_state import GameState
from catan_rl.core.game.engine.types import Action, ActionType


class ActionEncoder:
    """
    Encodes Catan actions into fixed-size action space.

    Maps complex game actions (with parameters) to discrete action indices
    for use with DQN-style algorithms.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._action_to_index_map = {}
        self._index_to_action_map = {}
        self._build_action_mappings()

    def _build_action_mappings(self):
        """Build bidirectional mappings between actions and indices."""
        index = 0

        # Pass action (always available)
        self._action_to_index_map[('pass',)] = index
        self._index_to_action_map[index] = Action(ActionType.PASS_TURN, {})
        index += 1

        # Roll dice action
        self._action_to_index_map[('roll_dice',)] = index
        self._index_to_action_map[index] = Action(ActionType.ROLL_DICE, {})
        index += 1

        # Build settlement actions (54 vertices on standard board)
        for vertex_id in range(54):
            action_key = ('build_settlement', vertex_id)
            self._action_to_index_map[action_key] = index
            self._index_to_action_map[index] = Action(ActionType.BUILD_SETTLEMENT, {'vertex_id': vertex_id})
            index += 1

        # Build city actions (upgrade settlements)
        for vertex_id in range(54):
            action_key = ('build_city', vertex_id)
            self._action_to_index_map[action_key] = index
            self._index_to_action_map[index] = Action(ActionType.BUILD_CITY, {'vertex_id': vertex_id})
            index += 1

        # Build road actions (72 edges on standard board)
        for edge_id in range(72):
            action_key = ('build_road', edge_id)
            self._action_to_index_map[action_key] = index
            self._index_to_action_map[index] = Action(ActionType.BUILD_ROAD, {'edge_id': edge_id})
            index += 1

        # Buy development card action
        self._action_to_index_map[('buy_dev_card',)] = index
        self._index_to_action_map[index] = Action(ActionType.BUY_DEV_CARD, {})
        index += 1

        # Play development card actions
        dev_cards = ['knight', 'road_building', 'year_of_plenty', 'monopoly']
        for card_type in dev_cards:
            action_key = ('play_dev_card', card_type)
            self._action_to_index_map[action_key] = index
            self._index_to_action_map[index] = Action(ActionType.PLAY_DEV_CARD, {'card_type': card_type})
            index += 1

        # Move robber actions (19 tiles on standard board)
        for tile_id in range(19):
            action_key = ('move_robber', tile_id)
            self._action_to_index_map[action_key] = index
            self._index_to_action_map[index] = Action(ActionType.MOVE_ROBBER, {'tile_id': tile_id})
            index += 1

        # Trade with bank actions (4:1 and 3:1/2:1 port trades)
        resources = ['brick', 'lumber', 'ore', 'grain', 'wool']
        for give_resource in resources:
            for get_resource in resources:
                if give_resource != get_resource:
                    action_key = ('trade_bank', give_resource, get_resource)
                    self._action_to_index_map[action_key] = index
                    self._index_to_action_map[index] = Action(
                        ActionType.TRADE_BANK,
                        {'give_resource': give_resource, 'get_resource': get_resource}
                    )
                    index += 1

        # Discard resources actions (for robber)
        for num_cards in range(1, 8):  # Discard 1-7 cards
            for resource_combo in self._generate_discard_combos(num_cards):
                action_key = ('discard',) + tuple(sorted(resource_combo))
                self._action_to_index_map[action_key] = index
                self._index_to_action_map[index] = Action(
                    ActionType.DISCARD,
                    {'resources': resource_combo}
                )
                index += 1

        self.action_space_size = index

    def _generate_discard_combos(self, num_cards: int) -> List[List[str]]:
        """Generate possible resource combinations for discarding."""
        resources = ['brick', 'lumber', 'ore', 'grain', 'wool']
        combos = []

        def generate_combo(remaining_cards, current_combo, resource_idx):
            if remaining_cards == 0:
                combos.append(current_combo.copy())
                return
            if resource_idx >= len(resources):
                return

            # Try different amounts of current resource (0 to remaining_cards)
            for count in range(min(remaining_cards + 1, 8)):  # Max 7 of any resource
                current_combo.extend([resources[resource_idx]] * count)
                generate_combo(remaining_cards - count, current_combo, resource_idx + 1)
                # Remove the cards we just added
                for _ in range(count):
                    if current_combo:
                        current_combo.pop()

        generate_combo(num_cards, [], 0)
        return combos

    def action_to_index(self, action: Action) -> Optional[int]:
        """Convert action to index."""
        action_key = self._action_to_key(action)
        return self._action_to_index_map.get(action_key)

    def index_to_action(self, index: int) -> Optional[Action]:
        """Convert index to action."""
        return self._index_to_action_map.get(index)

    def _action_to_key(self, action: Action) -> tuple:
        """Convert action to hashable key."""
        if action.action_type == ActionType.PASS_TURN:
            return ('pass',)
        elif action.action_type == ActionType.PASS_TURN:
            return ('end_turn',)
        elif action.action_type == ActionType.ROLL_DICE:
            return ('roll_dice',)
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            return ('build_settlement', action.payload.get('vertex_id'))
        elif action.action_type == ActionType.BUILD_CITY:
            return ('build_city', action.payload.get('vertex_id'))
        elif action.action_type == ActionType.BUILD_ROAD:
            return ('build_road', action.payload.get('edge_id'))
        elif action.action_type == ActionType.BUY_DEV_CARD:
            return ('buy_dev_card',)
        elif action.action_type == ActionType.PLAY_DEV_CARD:
            return ('play_dev_card', action.payload.get('card_type'))
        elif action.action_type == ActionType.MOVE_ROBBER:
            return ('move_robber', action.payload.get('tile_id'))
        elif action.action_type == ActionType.TRADE_BANK:
            give = action.payload.get('give_resource')
            get = action.payload.get('get_resource')
            return ('trade_bank', give, get)
        elif action.action_type == ActionType.DISCARD:
            resources = action.payload.get('resources', [])
            return ('discard',) + tuple(sorted(resources))
        else:
            return None

    def get_action_space_size(self) -> int:
        """Get total number of possible actions."""
        return self.action_space_size


class ActionMasker:
    """
    Creates action masks for valid actions given game state.

    Provides binary mask indicating which actions are legal
    in the current game state.
    """

    def __init__(self, action_encoder: ActionEncoder):
        self.action_encoder = action_encoder

    def get_action_mask(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """
        Get binary mask of valid actions.

        Args:
            game_state: Current game state
            player_id: Player whose turn it is

        Returns:
            Binary tensor [action_space_size] where 1 = valid, 0 = invalid
        """
        mask = torch.zeros(self.action_encoder.get_action_space_size(), dtype=torch.float32)

        # Get legal actions from game state
        legal_actions = self._get_legal_actions(game_state, player_id)

        # Set mask for legal actions
        for action in legal_actions:
            action_index = self.action_encoder.action_to_index(action)
            if action_index is not None:
                mask[action_index] = 1.0

        return mask

    def _get_legal_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get list of legal actions from game state."""
        legal_actions = []

        # Always allow pass
        legal_actions.append(Action(ActionType.PASS_TURN, {}))

        player = game_state.players[player_id]

        # Phase-specific actions
        if game_state.phase.value == 'setup':
            legal_actions.extend(self._get_setup_actions(game_state, player_id))
        elif game_state.phase.value == 'roll':
            legal_actions.append(Action(ActionType.ROLL_DICE, {}))
        elif game_state.phase.value == 'main':
            legal_actions.extend(self._get_main_phase_actions(game_state, player_id))
            legal_actions.append(Action(ActionType.PASS_TURN, {}))
        elif game_state.phase.value == 'discard':
            legal_actions.extend(self._get_discard_actions(game_state, player_id))
        elif game_state.phase.value == 'move_robber':
            legal_actions.extend(self._get_robber_actions(game_state, player_id))

        return legal_actions

    def _get_setup_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get legal actions during setup phase."""
        actions = []
        player = game_state.players[player_id]

        # Simplified setup logic - would need proper implementation
        if len(player.settlements) < 2:
            # Can build settlements on valid vertices
            for vertex_id in range(54):  # All possible vertices
                if self._can_build_settlement(game_state, player_id, vertex_id):
                    actions.append(Action(ActionType.BUILD_SETTLEMENT, {'vertex_id': vertex_id}))

        if len(player.roads) < 2:
            # Can build roads connected to settlements
            for edge_id in range(72):  # All possible edges
                if self._can_build_road(game_state, player_id, edge_id):
                    actions.append(Action(ActionType.BUILD_ROAD, {'edge_id': edge_id}))

        return actions

    def _get_main_phase_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get legal actions during main phase."""
        actions = []
        player = game_state.players[player_id]

        # Building actions
        for vertex_id in range(54):
            if self._can_build_settlement(game_state, player_id, vertex_id):
                actions.append(Action(ActionType.BUILD_SETTLEMENT, {'vertex_id': vertex_id}))
            if self._can_build_city(game_state, player_id, vertex_id):
                actions.append(Action(ActionType.BUILD_CITY, {'vertex_id': vertex_id}))

        for edge_id in range(72):
            if self._can_build_road(game_state, player_id, edge_id):
                actions.append(Action(ActionType.BUILD_ROAD, {'edge_id': edge_id}))

        # Development card actions
        if self._can_buy_dev_card(game_state, player_id):
            actions.append(Action(ActionType.BUY_DEV_CARD, {}))

        # Play development cards
        actions.extend(self._get_playable_dev_cards(game_state, player_id))

        # Trading actions
        actions.extend(self._get_trade_actions(game_state, player_id))

        return actions

    def _get_discard_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get legal discard actions when over hand limit."""
        actions = []
        player = game_state.players[player_id]
        total_cards = sum(player.resources.values())

        if total_cards > 7:
            discard_count = total_cards // 2
            # Generate valid discard combinations
            for combo in self._generate_valid_discards(player.resources, discard_count):
                actions.append(Action(ActionType.DISCARD, {'resources': combo}))

        return actions

    def _get_robber_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get legal robber movement actions."""
        actions = []

        # Can move robber to any tile except current position
        for tile_id in range(19):
            if tile_id != game_state.robber_tile:
                actions.append(Action(ActionType.MOVE_ROBBER, {'tile_id': tile_id}))

        return actions

    def _can_build_settlement(self, game_state: GameState, player_id: int, vertex_id: int) -> bool:
        """Check if player can build settlement at vertex."""
        # Simplified implementation - would need proper game rule checking
        player = game_state.players[player_id]

        # Check resources
        if (player.resources.get('brick', 0) < 1 or
            player.resources.get('lumber', 0) < 1 or
            player.resources.get('grain', 0) < 1 or
            player.resources.get('wool', 0) < 1):
            return False

        # Check if vertex is available (simplified)
        for other_player in game_state.players.values():
            if vertex_id in other_player.settlements or vertex_id in other_player.cities:
                return False

        return True

    def _can_build_city(self, game_state: GameState, player_id: int, vertex_id: int) -> bool:
        """Check if player can upgrade settlement to city."""
        player = game_state.players[player_id]

        # Must have settlement at this vertex
        if vertex_id not in player.settlements:
            return False

        # Check resources
        if (player.resources.get('ore', 0) < 3 or
            player.resources.get('grain', 0) < 2):
            return False

        return True

    def _can_build_road(self, game_state: GameState, player_id: int, edge_id: int) -> bool:
        """Check if player can build road at edge."""
        player = game_state.players[player_id]

        # Check resources
        if (player.resources.get('brick', 0) < 1 or
            player.resources.get('lumber', 0) < 1):
            return False

        # Check if edge is available
        for other_player in game_state.players.values():
            if edge_id in other_player.roads:
                return False

        return True

    def _can_buy_dev_card(self, game_state: GameState, player_id: int) -> bool:
        """Check if player can buy development card."""
        player = game_state.players[player_id]

        # Check resources
        if (player.resources.get('ore', 0) < 1 or
            player.resources.get('grain', 0) < 1 or
            player.resources.get('wool', 0) < 1):
            return False

        # Check if cards available
        return len(game_state.dev_deck) > 0

    def _get_playable_dev_cards(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get playable development cards."""
        actions = []
        player = game_state.players[player_id]

        # Can't play dev card if already played one this turn
        if game_state.played_dev_card_this_turn:
            return actions

        # Can't play cards bought this turn
        new_cards = game_state.new_dev_cards.get(player_id, [])
        playable_cards = [card for card in player.dev_cards if card not in new_cards]

        for card in playable_cards:
            actions.append(Action(ActionType.PLAY_DEV_CARD, {'card_type': card.value}))

        return actions

    def _get_trade_actions(self, game_state: GameState, player_id: int) -> List[Action]:
        """Get available trading actions."""
        actions = []
        player = game_state.players[player_id]

        # Bank trades (4:1 and port trades)
        resources = ['brick', 'lumber', 'ore', 'grain', 'wool']
        for give_resource in resources:
            if player.resources.get(give_resource, 0) >= 4:  # 4:1 trade
                for get_resource in resources:
                    if give_resource != get_resource:
                        actions.append(Action(
                            ActionType.TRADE_BANK,
                            {'give_resource': give_resource, 'get_resource': get_resource}
                        ))

        return actions

    def _generate_valid_discards(self, resources: Dict[str, int], discard_count: int) -> List[List[str]]:
        """Generate valid resource combinations to discard."""
        # Simplified implementation
        valid_combos = []
        resource_list = []

        for resource, count in resources.items():
            resource_list.extend([resource] * count)

        # Generate all combinations of discard_count cards
        from itertools import combinations
        for combo in combinations(resource_list, discard_count):
            valid_combos.append(list(combo))

        return valid_combos[:100]  # Limit to first 100 to avoid explosion


class CatanActionSpace:
    """Complete action space management for Catan RL environment."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.action_encoder = ActionEncoder(config)
        self.action_masker = ActionMasker(self.action_encoder)

    def get_action_space_size(self) -> int:
        """Get size of action space."""
        return self.action_encoder.get_action_space_size()

    def encode_action(self, action: Action) -> Optional[int]:
        """Encode action to index."""
        return self.action_encoder.action_to_index(action)

    def decode_action(self, index: int) -> Optional[Action]:
        """Decode index to action."""
        return self.action_encoder.index_to_action(index)

    def get_action_mask(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Get valid action mask."""
        return self.action_masker.get_action_mask(game_state, player_id)

    def sample_random_action(self, game_state: GameState, player_id: int) -> Tuple[int, Action]:
        """Sample random valid action."""
        mask = self.get_action_mask(game_state, player_id)
        valid_indices = torch.nonzero(mask).squeeze(-1)

        if len(valid_indices) == 0:
            # Fallback to pass action
            return 0, self.decode_action(0)

        random_idx = torch.randint(0, len(valid_indices), (1,)).item()
        action_idx = valid_indices[random_idx].item()
        action = self.decode_action(action_idx)

        return action_idx, action