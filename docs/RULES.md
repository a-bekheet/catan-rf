# Rules Reference (Base Catan)

This project targets the base game (no expansions) with standard 4-player rules.

## Setup
- Standard 19 hexes, 9 ports, robber starts on desert.
- Each player places 2 settlements and 2 roads in snake order.

## Turn Sequence
1. Roll dice (unless setup).
2. Distribute resources.
3. Trade (bank, ports, players).
4. Build actions (roads, settlements, cities, dev cards).
5. End turn.

## Victory
- First player to 10 victory points wins.

## Development Cards
- Knight, Victory Point, Road Building, Year of Plenty, Monopoly.

## Robber
- On roll of 7, players with >7 cards discard half.
- Current player moves robber and may steal one resource.

## Implementation Status
- Implemented: setup placements, dice rolls, resource distribution, robber movement, building roads/settlements/cities, victory at 10 VP.
- Pending: trading, development cards, largest army, longest road, ports.

## Notes
- For training, optional rule toggles will be supported (configurable).
