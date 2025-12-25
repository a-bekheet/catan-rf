from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .board import Board
from .game_state import GameState, PlayerState, ResourceBank, TurnPhase
from .types import Action, ActionType, BuildingType, ResourceType

COSTS: Dict[ActionType, ResourceBank] = {
    ActionType.BUILD_ROAD: {
        ResourceType.BRICK: 1,
        ResourceType.LUMBER: 1,
        ResourceType.ORE: 0,
        ResourceType.GRAIN: 0,
        ResourceType.WOOL: 0,
    },
    ActionType.BUILD_SETTLEMENT: {
        ResourceType.BRICK: 1,
        ResourceType.LUMBER: 1,
        ResourceType.ORE: 0,
        ResourceType.GRAIN: 1,
        ResourceType.WOOL: 1,
    },
    ActionType.BUILD_CITY: {
        ResourceType.BRICK: 0,
        ResourceType.LUMBER: 0,
        ResourceType.ORE: 3,
        ResourceType.GRAIN: 2,
        ResourceType.WOOL: 0,
    },
}

TRADE_RATE = 4


@dataclass(frozen=True)
class RuleViolation:
    reason: str


def _clone_resources(resources: ResourceBank) -> ResourceBank:
    return {key: int(value) for key, value in resources.items()}


def clone_state(state: GameState) -> GameState:
    players: Dict[int, PlayerState] = {}
    for pid, player in state.players.items():
        players[pid] = PlayerState(
            player_id=pid,
            resources=_clone_resources(player.resources),
            roads=set(player.roads),
            settlements=set(player.settlements),
            cities=set(player.cities),
            victory_points=player.victory_points,
        )

    return GameState(
        board=state.board,
        players=players,
        bank=_clone_resources(state.bank),
        current_player=state.current_player,
        phase=state.phase,
        turn_index=state.turn_index,
        robber_tile=state.robber_tile,
        vertex_occupancy=dict(state.vertex_occupancy),
        edge_occupancy=dict(state.edge_occupancy),
        setup_round=state.setup_round,
        setup_direction=state.setup_direction,
        pending_setup_vertex=state.pending_setup_vertex,
        last_roll=state.last_roll,
        pending_discards=dict(state.pending_discards),
        robber_player=state.robber_player,
        winner=state.winner,
    )


def _can_afford(resources: ResourceBank, cost: ResourceBank) -> bool:
    return all(resources[key] >= cost[key] for key in cost)

def _has_resources(resources: ResourceBank, bundle: Dict[ResourceType, int]) -> bool:
    return all(resources[res] >= amount for res, amount in bundle.items())


def _apply_cost(resources: ResourceBank, bank: ResourceBank, cost: ResourceBank) -> None:
    for key, amount in cost.items():
        if amount <= 0:
            continue
        resources[key] -= amount
        bank[key] += amount


def _award_resources(resources: ResourceBank, bank: ResourceBank, award: ResourceBank) -> None:
    for key, amount in award.items():
        if amount <= 0:
            continue
        if bank[key] < amount:
            continue
        bank[key] -= amount
        resources[key] += amount


def _resource_bank_empty() -> ResourceBank:
    return {
        ResourceType.BRICK: 0,
        ResourceType.LUMBER: 0,
        ResourceType.ORE: 0,
        ResourceType.GRAIN: 0,
        ResourceType.WOOL: 0,
    }


def _tiles_adjacent_to_vertex(board: Board, vertex_id: int) -> List[int]:
    tiles = []
    for tile_id, vertices in board.graph.hex_to_vertices.items():
        if vertex_id in vertices:
            tiles.append(tile_id)
    return tiles


def _settlement_distance_ok(board: Board, vertex_occupancy: Dict[int, Tuple[int, BuildingType]], vertex_id: int) -> bool:
    if vertex_id in vertex_occupancy:
        return False
    for neighbor in board.vertices_adjacent_to(vertex_id):
        if neighbor in vertex_occupancy:
            return False
    return True


def _player_has_road_touching(board: Board, player: PlayerState, vertex_id: int) -> bool:
    for edge_id in board.edges_for_vertex(vertex_id):
        if edge_id in player.roads:
            return True
    return False


def _edge_touches_player(board: Board, player: PlayerState, edge_id: int) -> bool:
    edge = board.graph.edges[edge_id]
    if edge.vertex_a in player.settlements or edge.vertex_b in player.settlements:
        return True
    if edge.vertex_a in player.cities or edge.vertex_b in player.cities:
        return True
    for road_id in player.roads:
        road = board.graph.edges[road_id]
        if edge.vertex_a in (road.vertex_a, road.vertex_b) or edge.vertex_b in (
            road.vertex_a,
            road.vertex_b,
        ):
            return True
    return False


def _build_settlement(state: GameState, vertex_id: int, is_setup: bool) -> None:
    player = state.players[state.current_player]
    state.vertex_occupancy[vertex_id] = (state.current_player, BuildingType.SETTLEMENT)
    player.settlements.add(vertex_id)
    player.victory_points += 1

    if not is_setup:
        _apply_cost(player.resources, state.bank, COSTS[ActionType.BUILD_SETTLEMENT])


def _build_city(state: GameState, vertex_id: int) -> None:
    player = state.players[state.current_player]
    player.settlements.remove(vertex_id)
    player.cities.add(vertex_id)
    state.vertex_occupancy[vertex_id] = (state.current_player, BuildingType.CITY)
    player.victory_points += 1
    _apply_cost(player.resources, state.bank, COSTS[ActionType.BUILD_CITY])


def _build_road(state: GameState, edge_id: int, is_setup: bool) -> None:
    player = state.players[state.current_player]
    state.edge_occupancy[edge_id] = state.current_player
    player.roads.add(edge_id)
    if not is_setup:
        _apply_cost(player.resources, state.bank, COSTS[ActionType.BUILD_ROAD])


def _distribute_resources(state: GameState, roll: int) -> None:
    award_by_player: Dict[int, ResourceBank] = {
        pid: _resource_bank_empty() for pid in state.players
    }
    for tile in state.board.tiles.values():
        if tile.number_token != roll or tile.tile_id == state.robber_tile:
            continue
        resource = tile.resource
        if resource == ResourceType.DESERT:
            continue
        vertices = state.board.graph.hex_to_vertices[tile.tile_id]
        for vertex_id in vertices:
            if vertex_id not in state.vertex_occupancy:
                continue
            owner, building = state.vertex_occupancy[vertex_id]
            if building == BuildingType.SETTLEMENT:
                award_by_player[owner][resource] += 1
            else:
                award_by_player[owner][resource] += 2

    # Apply awards if bank can cover them per resource
    for resource in [
        ResourceType.BRICK,
        ResourceType.LUMBER,
        ResourceType.ORE,
        ResourceType.GRAIN,
        ResourceType.WOOL,
    ]:
        total_needed = sum(award[resource] for award in award_by_player.values())
        if state.bank[resource] < total_needed:
            for award in award_by_player.values():
                award[resource] = 0

    for pid, award in award_by_player.items():
        _award_resources(state.players[pid].resources, state.bank, award)


def _next_setup_player(state: GameState) -> None:
    num_players = len(state.players)
    next_player = state.current_player + state.setup_direction
    if next_player < 0 or next_player >= num_players:
        if state.setup_round == 0:
            state.setup_round = 1
            state.setup_direction = -1
            next_player = num_players - 1
        else:
            state.phase = TurnPhase.ROLL
            state.current_player = 0
            state.turn_index = 0
            return
    state.current_player = next_player


def _collect_setup_resources(state: GameState, vertex_id: int) -> None:
    player = state.players[state.current_player]
    award = _resource_bank_empty()
    for tile_id in _tiles_adjacent_to_vertex(state.board, vertex_id):
        tile = state.board.tiles[tile_id]
        if tile.resource == ResourceType.DESERT:
            continue
        award[tile.resource] += 1
    _award_resources(player.resources, state.bank, award)


def _advance_turn(state: GameState) -> None:
    state.current_player = (state.current_player + 1) % len(state.players)
    state.turn_index += 1
    state.phase = TurnPhase.ROLL
    state.last_roll = None


def legal_actions(state: GameState) -> List[Action]:
    if state.winner is not None:
        return []

    actions: List[Action] = []
    player = state.players[state.current_player]

    if state.phase == TurnPhase.SETUP:
        if state.pending_setup_vertex is None:
            for vertex_id in state.board.graph.vertices:
                if _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id):
                    actions.append(
                        Action(action_type=ActionType.BUILD_SETTLEMENT, payload={"vertex_id": vertex_id})
                    )
        else:
            for edge_id in state.board.edges_for_vertex(state.pending_setup_vertex):
                if edge_id not in state.edge_occupancy:
                    actions.append(
                        Action(action_type=ActionType.BUILD_ROAD, payload={"edge_id": edge_id})
                    )
        return actions

    if state.phase == TurnPhase.ROLL:
        return [Action(action_type=ActionType.ROLL_DICE, payload={})]

    if state.phase == TurnPhase.DISCARD:
        pending = state.pending_discards.get(state.current_player, 0)
        if pending > 0:
            actions.append(
                Action(
                    action_type=ActionType.DISCARD,
                    payload={"player_id": state.current_player, "resources": {}},
                )
            )
        return actions

    if state.phase == TurnPhase.MOVE_ROBBER:
        for tile_id in state.board.tiles:
            if tile_id != state.robber_tile:
                actions.append(Action(action_type=ActionType.MOVE_ROBBER, payload={"tile_id": tile_id}))
        return actions

    if state.phase == TurnPhase.MAIN:
        actions.append(Action(action_type=ActionType.PASS_TURN, payload={}))
        if _can_afford(player.resources, COSTS[ActionType.BUILD_ROAD]):
            for edge_id in state.board.graph.edges:
                if edge_id in state.edge_occupancy:
                    continue
                if _edge_touches_player(state.board, player, edge_id):
                    actions.append(
                        Action(action_type=ActionType.BUILD_ROAD, payload={"edge_id": edge_id})
                    )
        if _can_afford(player.resources, COSTS[ActionType.BUILD_SETTLEMENT]):
            for vertex_id in state.board.graph.vertices:
                if not _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id):
                    continue
                if _player_has_road_touching(state.board, player, vertex_id):
                    actions.append(
                        Action(
                            action_type=ActionType.BUILD_SETTLEMENT,
                            payload={"vertex_id": vertex_id},
                        )
                    )
        if _can_afford(player.resources, COSTS[ActionType.BUILD_CITY]):
            for vertex_id in player.settlements:
                actions.append(
                    Action(action_type=ActionType.BUILD_CITY, payload={"vertex_id": vertex_id})
                )
        for give in [
            ResourceType.BRICK,
            ResourceType.LUMBER,
            ResourceType.ORE,
            ResourceType.GRAIN,
            ResourceType.WOOL,
        ]:
            if player.resources[give] < TRADE_RATE:
                continue
            for receive in [
                ResourceType.BRICK,
                ResourceType.LUMBER,
                ResourceType.ORE,
                ResourceType.GRAIN,
                ResourceType.WOOL,
            ]:
                if receive == give:
                    continue
                if state.bank[receive] <= 0:
                    continue
                actions.append(
                    Action(
                        action_type=ActionType.TRADE_BANK,
                        payload={
                            "give": give.value,
                            "receive": receive.value,
                            "rate": TRADE_RATE,
                        },
                    )
                )
        for other_pid, other_player in state.players.items():
            if other_pid == state.current_player:
                continue
            for give in [
                ResourceType.BRICK,
                ResourceType.LUMBER,
                ResourceType.ORE,
                ResourceType.GRAIN,
                ResourceType.WOOL,
            ]:
                if player.resources[give] <= 0:
                    continue
                for receive in [
                    ResourceType.BRICK,
                    ResourceType.LUMBER,
                    ResourceType.ORE,
                    ResourceType.GRAIN,
                    ResourceType.WOOL,
                ]:
                    if other_player.resources[receive] <= 0:
                        continue
                    actions.append(
                        Action(
                            action_type=ActionType.TRADE_PLAYER,
                            payload={
                                "to_player": other_pid,
                                "give": {give.value: 1},
                                "receive": {receive.value: 1},
                            },
                        )
                    )
        return actions

    return actions


def validate_action(state: GameState, action: Action) -> List[RuleViolation]:
    violations: List[RuleViolation] = []

    if state.winner is not None:
        violations.append(RuleViolation(reason="game_over"))
        return violations

    if state.phase == TurnPhase.SETUP:
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            vertex_id = int(action.payload.get("vertex_id", -1))
            if state.pending_setup_vertex is not None:
                violations.append(RuleViolation(reason="must_place_road"))
            elif not _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id):
                violations.append(RuleViolation(reason="invalid_settlement_location"))
        elif action.action_type == ActionType.BUILD_ROAD:
            edge_id = int(action.payload.get("edge_id", -1))
            if state.pending_setup_vertex is None:
                violations.append(RuleViolation(reason="must_place_settlement_first"))
            elif edge_id in state.edge_occupancy:
                violations.append(RuleViolation(reason="edge_occupied"))
            else:
                edge = state.board.graph.edges.get(edge_id)
                if edge is None:
                    violations.append(RuleViolation(reason="edge_not_found"))
                else:
                    if state.pending_setup_vertex not in (edge.vertex_a, edge.vertex_b):
                        violations.append(RuleViolation(reason="road_not_adjacent_to_setup"))
        else:
            violations.append(RuleViolation(reason="invalid_action_for_setup"))
        return violations

    if state.phase == TurnPhase.ROLL:
        if action.action_type != ActionType.ROLL_DICE:
            violations.append(RuleViolation(reason="must_roll"))
        return violations

    if state.phase == TurnPhase.DISCARD:
        if action.action_type != ActionType.DISCARD:
            violations.append(RuleViolation(reason="must_discard"))
            return violations
        player_id = int(action.payload.get("player_id", -1))
        if player_id != state.current_player:
            violations.append(RuleViolation(reason="not_players_turn"))
            return violations
        required = state.pending_discards.get(player_id, 0)
        resources = action.payload.get("resources", {})
        if not isinstance(resources, dict):
            violations.append(RuleViolation(reason="invalid_discard_payload"))
            return violations
        discard_total = sum(int(value) for value in resources.values())
        if discard_total != required:
            violations.append(RuleViolation(reason="discard_count_mismatch"))
        player = state.players[player_id]
        for key, value in resources.items():
            try:
                resource = ResourceType(key)
            except ValueError:
                violations.append(RuleViolation(reason="invalid_resource"))
                continue
            if player.resources[resource] < int(value):
                violations.append(RuleViolation(reason="insufficient_resource"))
        return violations

    if state.phase == TurnPhase.MOVE_ROBBER:
        if action.action_type != ActionType.MOVE_ROBBER:
            violations.append(RuleViolation(reason="must_move_robber"))
            return violations
        tile_id = int(action.payload.get("tile_id", -1))
        if tile_id not in state.board.tiles:
            violations.append(RuleViolation(reason="tile_not_found"))
        return violations

    if state.phase == TurnPhase.MAIN:
        player = state.players[state.current_player]
        if action.action_type == ActionType.PASS_TURN:
            return violations
        if action.action_type == ActionType.BUILD_ROAD:
            edge_id = int(action.payload.get("edge_id", -1))
            if edge_id in state.edge_occupancy:
                violations.append(RuleViolation(reason="edge_occupied"))
            elif edge_id not in state.board.graph.edges:
                violations.append(RuleViolation(reason="edge_not_found"))
            elif not _edge_touches_player(state.board, player, edge_id):
                violations.append(RuleViolation(reason="road_not_connected"))
            elif not _can_afford(player.resources, COSTS[ActionType.BUILD_ROAD]):
                violations.append(RuleViolation(reason="insufficient_resources"))
            return violations
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            vertex_id = int(action.payload.get("vertex_id", -1))
            if not _settlement_distance_ok(state.board, state.vertex_occupancy, vertex_id):
                violations.append(RuleViolation(reason="invalid_settlement_location"))
            elif not _player_has_road_touching(state.board, player, vertex_id):
                violations.append(RuleViolation(reason="settlement_not_connected"))
            elif not _can_afford(player.resources, COSTS[ActionType.BUILD_SETTLEMENT]):
                violations.append(RuleViolation(reason="insufficient_resources"))
            return violations
        if action.action_type == ActionType.BUILD_CITY:
            vertex_id = int(action.payload.get("vertex_id", -1))
            if vertex_id not in player.settlements:
                violations.append(RuleViolation(reason="city_requires_settlement"))
            elif not _can_afford(player.resources, COSTS[ActionType.BUILD_CITY]):
                violations.append(RuleViolation(reason="insufficient_resources"))
            return violations
        if action.action_type == ActionType.TRADE_BANK:
            give = action.payload.get("give")
            receive = action.payload.get("receive")
            rate = int(action.payload.get("rate", TRADE_RATE))
            try:
                give_res = ResourceType(str(give))
                receive_res = ResourceType(str(receive))
            except ValueError:
                violations.append(RuleViolation(reason="invalid_trade_resource"))
                return violations
            if give_res == receive_res or give_res == ResourceType.DESERT:
                violations.append(RuleViolation(reason="invalid_trade_pair"))
                return violations
            if rate <= 0:
                violations.append(RuleViolation(reason="invalid_trade_rate"))
                return violations
            if player.resources[give_res] < rate:
                violations.append(RuleViolation(reason="insufficient_resources"))
            if state.bank[receive_res] <= 0:
                violations.append(RuleViolation(reason="bank_empty"))
            return violations
        if action.action_type == ActionType.TRADE_PLAYER:
            to_player = int(action.payload.get("to_player", -1))
            if to_player not in state.players or to_player == state.current_player:
                violations.append(RuleViolation(reason="invalid_trade_target"))
                return violations
            give_payload = action.payload.get("give", {})
            receive_payload = action.payload.get("receive", {})
            try:
                give_bundle = {ResourceType(k): int(v) for k, v in give_payload.items()}
                receive_bundle = {ResourceType(k): int(v) for k, v in receive_payload.items()}
            except ValueError:
                violations.append(RuleViolation(reason="invalid_trade_resource"))
                return violations
            if not _has_resources(player.resources, give_bundle):
                violations.append(RuleViolation(reason="insufficient_resources"))
            if not _has_resources(state.players[to_player].resources, receive_bundle):
                violations.append(RuleViolation(reason="counterparty_insufficient_resources"))
            return violations

        violations.append(RuleViolation(reason="unsupported_action"))
        return violations

    return violations


def _check_winner(state: GameState) -> None:
    for pid, player in state.players.items():
        if player.victory_points >= 10:
            state.winner = pid
            state.phase = TurnPhase.END
            return


def apply_action(state: GameState, action: Action) -> GameState:
    violations = validate_action(state, action)
    if violations:
        reasons = ", ".join(v.reason for v in violations)
        raise ValueError(f"Illegal action: {reasons}")

    next_state = clone_state(state)

    if next_state.phase == TurnPhase.SETUP:
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            vertex_id = int(action.payload["vertex_id"])
            _build_settlement(next_state, vertex_id, is_setup=True)
            next_state.pending_setup_vertex = vertex_id
        elif action.action_type == ActionType.BUILD_ROAD:
            edge_id = int(action.payload["edge_id"])
            _build_road(next_state, edge_id, is_setup=True)
            if next_state.setup_round == 1:
                _collect_setup_resources(next_state, next_state.pending_setup_vertex)
            next_state.pending_setup_vertex = None
            _next_setup_player(next_state)
        return next_state

    if next_state.phase == TurnPhase.ROLL:
        if action.action_type == ActionType.ROLL_DICE:
            import random

            roll = action.payload.get("roll")
            if roll is None:
                roll = random.randint(1, 6) + random.randint(1, 6)
            next_state.last_roll = int(roll)
            if next_state.last_roll == 7:
                next_state.robber_player = next_state.current_player
                pending: Dict[int, int] = {}
                for pid, player in next_state.players.items():
                    total_cards = sum(player.resources.values())
                    if total_cards > 7:
                        pending[pid] = total_cards // 2
                next_state.pending_discards = pending
                if pending:
                    next_state.phase = TurnPhase.DISCARD
                    next_state.current_player = sorted(pending.keys())[0]
                else:
                    next_state.phase = TurnPhase.MOVE_ROBBER
            else:
                _distribute_resources(next_state, next_state.last_roll)
                next_state.phase = TurnPhase.MAIN
        return next_state

    if next_state.phase == TurnPhase.DISCARD:
        if action.action_type == ActionType.DISCARD:
            player_id = int(action.payload["player_id"])
            resources = action.payload.get("resources", {})
            player = next_state.players[player_id]
            for key, value in resources.items():
                resource = ResourceType(key)
                amount = int(value)
                player.resources[resource] -= amount
                next_state.bank[resource] += amount
            next_state.pending_discards.pop(player_id, None)
            if next_state.pending_discards:
                next_state.current_player = sorted(next_state.pending_discards.keys())[0]
            else:
                next_state.phase = TurnPhase.MOVE_ROBBER
                next_state.current_player = next_state.robber_player or next_state.current_player
        return next_state

    if next_state.phase == TurnPhase.MOVE_ROBBER:
        if action.action_type == ActionType.MOVE_ROBBER:
            tile_id = int(action.payload["tile_id"])
            next_state.robber_tile = tile_id
            next_state.phase = TurnPhase.MAIN
        return next_state

    if next_state.phase == TurnPhase.MAIN:
        if action.action_type == ActionType.PASS_TURN:
            _advance_turn(next_state)
            return next_state
        if action.action_type == ActionType.BUILD_ROAD:
            edge_id = int(action.payload["edge_id"])
            _build_road(next_state, edge_id, is_setup=False)
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            vertex_id = int(action.payload["vertex_id"])
            _build_settlement(next_state, vertex_id, is_setup=False)
        elif action.action_type == ActionType.BUILD_CITY:
            vertex_id = int(action.payload["vertex_id"])
            _build_city(next_state, vertex_id)
        elif action.action_type == ActionType.TRADE_BANK:
            give_res = ResourceType(str(action.payload["give"]))
            receive_res = ResourceType(str(action.payload["receive"]))
            rate = int(action.payload.get("rate", TRADE_RATE))
            next_state.players[next_state.current_player].resources[give_res] -= rate
            next_state.bank[give_res] += rate
            next_state.bank[receive_res] -= 1
            next_state.players[next_state.current_player].resources[receive_res] += 1
        elif action.action_type == ActionType.TRADE_PLAYER:
            to_player = int(action.payload["to_player"])
            give_bundle = {ResourceType(k): int(v) for k, v in action.payload["give"].items()}
            receive_bundle = {ResourceType(k): int(v) for k, v in action.payload["receive"].items()}
            for res, amt in give_bundle.items():
                next_state.players[next_state.current_player].resources[res] -= amt
                next_state.players[to_player].resources[res] += amt
            for res, amt in receive_bundle.items():
                next_state.players[to_player].resources[res] -= amt
                next_state.players[next_state.current_player].resources[res] += amt

        _check_winner(next_state)
        return next_state

    return next_state
