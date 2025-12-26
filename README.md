# Catan-RF

Engineering-grade reinforcement learning project for training an optimal AI in Settlers of Catan.

## Goals
- Build a deterministic, fully validated Catan rules engine.
- Provide a Gym-style environment for self-play and evaluation.
- Support reproducible experiments, logging, and model checkpoints.
- Produce clear documentation suitable for a portfolio project.

## Project Layout
- `src/catan/engine`: Pure game engine (state, board, rules, validation).
- `src/catan/env`: Gym-like environment wrapper and specs.
- `src/catan/rl`: Agents, training loops, self-play, evaluation.
- `src/catan/utils`: Reproducibility, config loading, logging helpers.
- `docs`: Architecture, rules, and design decisions.
- `tests`: Unit tests for deterministic board and core rules.

## Quick Start
1. Create a virtual environment.
2. Install deps:
   - `pip install -e .[dev]`
3. Run tests:
   - `pytest`
4. Play via CLI:
   - `python scripts/play_cli.py`
5. Run web UI:
   - `uvicorn catan.web.server:app --reload --app-dir src`

## Documentation
- `docs/ARCHITECTURE.md`
- `docs/RULES.md`
- `docs/ROADMAP.md`

## Status
Playable engine + CLI are implemented with core build/roll/robber flow and constrained number placement (no adjacent 6/8). Trading and dev cards are pending.
