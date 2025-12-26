# Architecture

## Goals
- Deterministic, replayable game engine with strict validation.
- Clean separation between rules engine and RL environment.
- Scaffolding that supports self-play and large-scale training.

## Layers
1. Engine (`src/catan/engine`)
   - Pure domain logic: board, rules, game state transitions.
   - No RNG hidden inside state transitions; all randomness comes from explicit seeds.
   - Fully serializable states for replay and testing.

2. Environment (`src/catan/env`)
   - Gym-style interface over the engine.
   - Exposes observations, reward shaping, and termination logic.

3. RL (`src/catan/rl`)
   - Agents, self-play loop, evaluation harness, and training scripts.

## Board Model
- Uses axial hex coordinates for 19 tiles (radius=2).
- Vertices and edges are derived deterministically from integer grid math.
- 54 vertices and 72 edges are generated, matching standard Catan.
- Tile adjacency is computed from axial neighbor directions and cached.
- Number tokens are assigned with basic fairness constraints (default: no adjacent 6/8).

## Determinism Policy
- `standard_board(seed)` shuffles resources and number tokens with a local RNG.
- All randomness is seeded and passed explicitly.

## Next Steps
- Implement full rules in `engine/rules.py`.
- Define action/observation schema for agents.
- Add legality checks and game termination conditions.
