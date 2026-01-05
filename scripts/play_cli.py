from __future__ import annotations

import sys
from typing import Dict, List

from catan.engine.board import standard_board
from catan.engine.game_state import GameState, initial_game_state
from catan.engine.types import Action, ActionType, ResourceType

HELP_TEXT = """
Commands:
  help                         Show this help text
  state                        Show current game state summary
  tiles                        List tiles (id, resource, number, robber)
  vertices                     List vertices (id, coord, occupancy)
  edges                        List edges (id, v1, v2, occupancy)
  legal                        Show count of legal actions (and first 10)
  roll [value]                 Roll dice (or force a value)
  build_settlement <vertex>    Build settlement
  build_city <vertex>          Build city
  build_road <edge>            Build road
  move_robber <tile_id>        Move robber
  trade <give> <receive>       Trade 4:1 with bank
  trade_player <pid> give=a:1,b:0 receive=c:1
  discard <res>=<n>,...        Discard resources (only when prompted)
  buy_dev_card                 Buy development card (costs: ore, grain, wool)
  play_knight <tile_id>        Play knight card and move robber
  play_monopoly <resource>     Play monopoly card for a resource
  play_year_of_plenty <res1> <res2>  Play year of plenty for 2 resources
  play_road_building <edge1> <edge2>  Play road building for 2 roads
  end                          End turn
  quit                         Exit
""".strip()


def _resource_str(resources: Dict[ResourceType, int]) -> str:
    parts = []
    for res in [
        ResourceType.BRICK,
        ResourceType.LUMBER,
        ResourceType.ORE,
        ResourceType.GRAIN,
        ResourceType.WOOL,
    ]:
        parts.append(f"{res.value}:{resources[res]}")
    return ", ".join(parts)


def _print_state(state: GameState) -> None:
    print(f"Turn {state.turn_index} | Player {state.current_player} | Phase {state.phase.value}")
    if state.last_roll is not None:
        print(f"Last roll: {state.last_roll}")
    if state.winner is not None:
        print(f"Winner: Player {state.winner}")
    print(f"Robber tile: {state.robber_tile}")
    print(f"Dev deck: {len(state.dev_deck)} cards")
    for pid, player in state.players.items():
        # Count dev cards by type
        from collections import Counter
        dev_card_counts = Counter(player.dev_cards)
        dev_str = ", ".join([f"{card.value}:{count}" for card, count in dev_card_counts.items()]) or "none"

        print(
            f"Player {pid} | VP {player.victory_points} | Knights {player.knights_played} | "
            f"Roads {len(player.roads)} | Settlements {len(player.settlements)} | "
            f"Cities {len(player.cities)} | Resources: {_resource_str(player.resources)} | "
            f"Dev cards: {dev_str}"
        )


def _print_tiles(state: GameState) -> None:
    for tile in state.board.tiles.values():
        robber = "robber" if tile.tile_id == state.robber_tile else ""
        print(
            f"Tile {tile.tile_id} | {tile.resource.value} | "
            f"{tile.number_token} {robber}".strip()
        )


def _print_vertices(state: GameState) -> None:
    for vertex in state.board.graph.vertices.values():
        occ = state.vertex_occupancy.get(vertex.vertex_id)
        if occ:
            print(f"V{vertex.vertex_id} {vertex.coord} | P{occ[0]} {occ[1].value}")
        else:
            print(f"V{vertex.vertex_id} {vertex.coord} | empty")


def _print_edges(state: GameState) -> None:
    for edge in state.board.graph.edges.values():
        occ = state.edge_occupancy.get(edge.edge_id)
        if occ is None:
            print(f"E{edge.edge_id} ({edge.vertex_a}-{edge.vertex_b}) | empty")
        else:
            print(f"E{edge.edge_id} ({edge.vertex_a}-{edge.vertex_b}) | P{occ}")


def _print_legal(state: GameState) -> None:
    actions = state.legal_actions()
    print(f"Legal actions: {len(actions)}")
    for action in actions[:10]:
        print(f"- {action.action_type.value} {action.payload}")


def _parse_resources(arg: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not arg:
        return result
    for part in arg.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = int(value.strip())
    return result


def main() -> int:
    board = standard_board()
    state = initial_game_state(board)
    print("Catan CLI - type 'help' for commands")

    while True:
        prompt = f"P{state.current_player}:{state.phase.value}> "
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting")
            return 0
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        try:
            if cmd == "help":
                print(HELP_TEXT)
            elif cmd == "state":
                _print_state(state)
            elif cmd == "tiles":
                _print_tiles(state)
            elif cmd == "vertices":
                _print_vertices(state)
            elif cmd == "edges":
                _print_edges(state)
            elif cmd == "legal":
                _print_legal(state)
            elif cmd == "roll":
                payload = {}
                if len(parts) > 1:
                    payload["roll"] = int(parts[1])
                state = state.apply(Action(ActionType.ROLL_DICE, payload))
            elif cmd == "build_settlement":
                vertex_id = int(parts[1])
                state = state.apply(Action(ActionType.BUILD_SETTLEMENT, {"vertex_id": vertex_id}))
            elif cmd == "build_city":
                vertex_id = int(parts[1])
                state = state.apply(Action(ActionType.BUILD_CITY, {"vertex_id": vertex_id}))
            elif cmd == "build_road":
                edge_id = int(parts[1])
                state = state.apply(Action(ActionType.BUILD_ROAD, {"edge_id": edge_id}))
            elif cmd == "move_robber":
                tile_id = int(parts[1])
                state = state.apply(Action(ActionType.MOVE_ROBBER, {"tile_id": tile_id}))
            elif cmd == "trade":
                give = parts[1].lower()
                receive = parts[2].lower()
                state = state.apply(
                    Action(
                        ActionType.TRADE_BANK,
                        {"give": give, "receive": receive, "rate": 4},
                    )
                )
            elif cmd == "trade_player":
                target = int(parts[1])
                give = _parse_resources(parts[2].replace("give=", ""))
                receive = _parse_resources(parts[3].replace("receive=", ""))
                state = state.apply(
                    Action(
                        ActionType.TRADE_PLAYER,
                        {"to_player": target, "give": give, "receive": receive},
                    )
                )
            elif cmd == "discard":
                resources = _parse_resources(" ".join(parts[1:]))
                state = state.apply(
                    Action(
                        ActionType.DISCARD,
                        {"player_id": state.current_player, "resources": resources},
                    )
                )
            elif cmd == "buy_dev_card":
                state = state.apply(Action(ActionType.BUY_DEV_CARD, {}))
            elif cmd == "play_knight":
                tile_id = int(parts[1])
                state = state.apply(
                    Action(
                        ActionType.PLAY_DEV_CARD,
                        {"dev_card": "knight", "tile_id": tile_id},
                    )
                )
            elif cmd == "play_monopoly":
                resource = parts[1].lower()
                state = state.apply(
                    Action(
                        ActionType.PLAY_DEV_CARD,
                        {"dev_card": "monopoly", "resource": resource},
                    )
                )
            elif cmd == "play_year_of_plenty":
                resource1 = parts[1].lower()
                resource2 = parts[2].lower()
                state = state.apply(
                    Action(
                        ActionType.PLAY_DEV_CARD,
                        {"dev_card": "year_of_plenty", "resource1": resource1, "resource2": resource2},
                    )
                )
            elif cmd == "play_road_building":
                edge1 = int(parts[1])
                edge2 = int(parts[2])
                state = state.apply(
                    Action(
                        ActionType.PLAY_DEV_CARD,
                        {"dev_card": "road_building", "roads": [edge1, edge2]},
                    )
                )
            elif cmd in {"end", "pass"}:
                state = state.apply(Action(ActionType.PASS_TURN, {}))
            elif cmd == "quit":
                return 0
            else:
                print("Unknown command. Type 'help'.")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
