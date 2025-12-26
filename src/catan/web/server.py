from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from catan.engine.board import standard_board
from catan.engine.game_state import GameState, initial_game_state
from catan.engine.types import Action, ActionType

ROOT_DIR = Path(__file__).resolve().parents[3]
CARDS_DIR = ROOT_DIR / "cards"
TILES_DIR = ROOT_DIR / "tiles"

app = FastAPI(title="Catan-RF")
app.mount("/static/cards", StaticFiles(directory=CARDS_DIR), name="cards")
app.mount("/static/tiles", StaticFiles(directory=TILES_DIR), name="tiles")
app.mount("/static", StaticFiles(directory=ROOT_DIR), name="root")

_state_lock = threading.Lock()
_state: GameState = initial_game_state(standard_board(seed=42))


def _serialize_state(state: GameState) -> Dict[str, Any]:
    return {
        "turn_index": state.turn_index,
        "current_player": state.current_player,
        "phase": state.phase.value,
        "last_roll": state.last_roll,
        "winner": state.winner,
        "robber_tile": state.robber_tile,
        "pending_discards": {str(pid): count for pid, count in state.pending_discards.items()},
        "tiles": [
            {
                "tile_id": tile.tile_id,
                "axial": tile.axial,
                "resource": tile.resource.value,
                "number_token": tile.number_token,
            }
            for tile in state.board.tiles.values()
        ],
        "vertices": [
            {
                "vertex_id": vertex.vertex_id,
                "coord": vertex.coord,
                "occupancy": state.vertex_occupancy.get(vertex.vertex_id),
            }
            for vertex in state.board.graph.vertices.values()
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "vertex_a": edge.vertex_a,
                "vertex_b": edge.vertex_b,
                "occupancy": state.edge_occupancy.get(edge.edge_id),
            }
            for edge in state.board.graph.edges.values()
        ],
        "players": [
            {
                "player_id": player.player_id,
                "resources": {k.value: v for k, v in player.resources.items()},
                "roads": sorted(player.roads),
                "settlements": sorted(player.settlements),
                "cities": sorted(player.cities),
                "victory_points": player.victory_points,
                "dev_cards": [card.value for card in player.dev_cards],
                "knights_played": player.knights_played,
            }
            for player in state.players.values()
        ],
        "dev_deck_size": len(state.dev_deck),
        "played_dev_card_this_turn": state.played_dev_card_this_turn,
        "bank": {k.value: v for k, v in state.bank.items()},
        "current_player_pieces": {
            "settlements": 5 - len(state.players[state.current_player].settlements),
            "cities": 4 - len(state.players[state.current_player].cities),
            "roads": 15 - len(state.players[state.current_player].roads),
        },
        "legal_actions": [
            {"action_type": action.action_type.value, "payload": action.payload}
            for action in state.legal_actions()
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Catan-RF</title>
  <style>
    :root {
      --bg: #f3ece1;
      --ink: #2c2622;
      --accent: #a85b2d;
      --panel: #fffaf2;
      --panel-2: #f6efe3;
      --stroke: #42362d;
      --gold: #e3b15a;
      --shadow: rgba(29, 24, 21, 0.2);
      --muted: #6e5b4e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Fraunces", "Iowan Old Style", "Palatino", "Georgia", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 20%, #fff6e6 0%, rgba(255,255,255,0) 45%),
        radial-gradient(circle at 80% 0%, #f1e0cb 0%, rgba(255,255,255,0) 50%),
        linear-gradient(120deg, #f3ece1 0%, #ead9c3 100%);
      min-height: 100vh;
    }
    header {
      padding: 18px 28px;
      border-bottom: 1px solid #d2c2ae;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.6);
      backdrop-filter: blur(4px);
    }
    header h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    header .meta {
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
      color: var(--muted);
    }
    .container {
      display: grid;
      grid-template-columns: minmax(520px, 1.4fr) minmax(280px, 0.8fr);
      gap: 20px;
      padding: 20px 28px 28px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid #dac9b5;
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 30px var(--shadow);
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    #board {
      width: 100%;
      height: 70vh;
    }
    .board-shell {
      background: radial-gradient(circle at 40% 40%, #fdf8f1 0%, #efe2d2 60%);
      border-radius: 16px;
      border: 1px solid #d8c4ae;
      padding: 12px;
    }
    .actions {
      display: none;
    }
    .actions.open {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 30vh;
      overflow: auto;
      margin-top: 8px;
    }
    .actions-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid #cbb8a3;
      background: #fff7eb;
      cursor: pointer;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
    }
    button {
      border: 1px solid #cbb8a3;
      background: var(--panel-2);
      color: var(--ink);
      padding: 10px 12px;
      border-radius: 10px;
      cursor: pointer;
      text-align: left;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
    }
    button:hover {
      background: #efe2d1;
    }
    .status {
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }
    .player-card {
      background: #fff4e6;
      border-radius: 12px;
      border: 1px solid #e2cdb4;
      padding: 10px;
      display: grid;
      gap: 6px;
    }
    .player-card.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(168, 91, 45, 0.2);
    }
    .player-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
    }
    .resource-row {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
    }
    .resource {
      background: #f4eadb;
      border-radius: 8px;
      padding: 4px;
      text-align: center;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 11px;
    }
    .resource img {
      width: 100%;
      height: 42px;
      object-fit: contain;
      display: block;
    }
    .dev-cards-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(40px, 1fr));
      gap: 6px;
      margin-top: 6px;
    }
    .dev-card {
      background: #f4eadb;
      border-radius: 8px;
      padding: 4px;
      text-align: center;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 11px;
      border: 1px solid #d4be9d;
    }
    .dev-card img {
      width: 100%;
      height: 32px;
      object-fit: contain;
      display: block;
      border-radius: 4px;
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(5, minmax(35px, 1fr));
      gap: 4px;
      margin-top: 12px;
    }
    .legend-row {
      display: contents;
    }
    .legend-card {
      background: #f6efe3;
      border-radius: 8px;
      padding: 6px;
      text-align: center;
      font-size: 11px;
      border: 1px solid #e0cbb2;
    }
    .legend-card img {
      width: 100%;
      height: 36px;
      object-fit: contain;
    }
    .resource-deck-card, .building-deck-card, .dev-deck-card {
      position: relative;
    }
    .resource-count, .building-count, .dev-count {
      position: absolute;
      bottom: 2px;
      right: 2px;
      background: rgba(42, 54, 45, 0.9);
      color: white;
      font-size: 9px;
      font-weight: bold;
      padding: 1px 3px;
      border-radius: 3px;
      line-height: 1;
    }
    .footer-meta {
      margin-top: 12px;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
      color: var(--muted);
    }
    .board-overlay {
      fill: transparent;
      stroke: transparent;
      cursor: pointer;
      pointer-events: none;
    }
    .board-overlay.active {
      pointer-events: all;
      opacity: 0.9;
    }
    .board-overlay.vertex.active {
      fill: #c74a3a;
      stroke: none;
    }
    .board-overlay.edge.active {
      stroke: #c74a3a;
      stroke-width: 5;
      stroke-linecap: round;
    }
    .board-overlay.tile.active {
      fill: rgba(199, 74, 58, 0.12);
      stroke: none;
    }
    .discard-panel {
      margin-top: 12px;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid #e0cbb2;
      background: #f6efe3;
      display: grid;
      gap: 8px;
    }
    .trade-panel {
      margin-top: 12px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid #e0cbb2;
      background: #fff7eb;
      display: grid;
      gap: 10px;
    }
    .trade-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 8px;
    }
    .trade-row select, .trade-row input {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #ccb8a3;
      padding: 8px;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
      background: #fffaf2;
    }
    .trade-row input[type="number"] {
      text-align: center;
    }
    .discard-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(40px, 1fr));
      gap: 6px;
    }
    .discard-grid input {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #ccb8a3;
      padding: 6px;
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 12px;
      text-align: center;
      background: #fffaf2;
    }
    .dice-display {
      background: #f6efe3;
      border: 1px solid #e0cbb2;
      border-radius: 12px;
      padding: 12px;
      margin: 10px 0;
      text-align: center;
    }
    .dice-container {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .dice-icon {
      font-size: 24px;
    }
    .dice-result {
      font-family: "IBM Plex Mono", "Menlo", "Monaco", monospace;
      font-size: 18px;
      font-weight: bold;
      color: var(--accent);
      min-width: 24px;
    }
    .player-color-dot {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 6px;
      border: 1px solid rgba(66, 54, 45, 0.3);
      vertical-align: middle;
    }
    .player-info {
      display: flex;
      align-items: center;
    }
    .player-colors-legend {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .player-color-legend-content {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      font-size: 8px;
      text-align: center;
    }
    .legend-player-item {
      display: flex;
      align-items: center;
      gap: 3px;
      justify-content: center;
    }
    .legend-player-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      border: 1px solid rgba(66, 54, 45, 0.3);
    }
    @media (max-width: 960px) {
      .container { grid-template-columns: 1fr; }
      #board { height: 55vh; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Catan-RF</h1>
      <div class="meta" id="phase"></div>
    </div>
    <div>
      <button onclick="resetGame()">Reset Game</button>
    </div>
  </header>
  <div class="container">
    <div class="panel">
      <div class="board-shell">
        <svg id="board" preserveAspectRatio="xMidYMid meet"></svg>
      </div>
      <div class="legend">
        <!-- Row 1: Resources -->
        <div class="legend-card resource-deck-card"><img src="/static/cards/Resource_Brick.png" alt="brick"/>Brick<div id="brickCount" class="resource-count">19</div></div>
        <div class="legend-card resource-deck-card"><img src="/static/cards/Resource_Wood.png" alt="wood"/>Wood<div id="lumberCount" class="resource-count">19</div></div>
        <div class="legend-card resource-deck-card"><img src="/static/cards/Resource_Ore.png" alt="ore"/>Ore<div id="oreCount" class="resource-count">19</div></div>
        <div class="legend-card resource-deck-card"><img src="/static/cards/Resource_Grain.png" alt="grain"/>Grain<div id="grainCount" class="resource-count">19</div></div>
        <div class="legend-card resource-deck-card"><img src="/static/cards/Resource_Sheep.png" alt="sheep"/>Sheep<div id="woolCount" class="resource-count">19</div></div>

        <!-- Row 2: Buildings and Dev Cards -->
        <div class="legend-card building-deck-card"><img id="settlementImg" src="/static/cards/Piece-Settlement.png" alt="settlements"/>Settlements<div id="settlementCount" class="building-count">5</div></div>
        <div class="legend-card building-deck-card"><img id="cityImg" src="/static/cards/Piece-City.png" alt="cities"/>Cities<div id="cityCount" class="building-count">4</div></div>
        <div class="legend-card building-deck-card"><img id="roadImg" src="/static/cards/Piece-Road.png" alt="roads"/>Roads<div id="roadCount" class="building-count">15</div></div>
        <div class="legend-card dev-deck-card"><img src="/static/cards/CardBack1.png" alt="dev cards"/>Dev Cards<div id="devDeckCount" class="dev-count">25</div></div>
        <div class="legend-card player-colors-legend">
          <div style="font-size: 10px; font-weight: bold; margin-bottom: 4px;">Players</div>
          <div id="playerColorLegend" class="player-color-legend-content"></div>
        </div>
      </div>
    </div>
      <div class="panel">
        <h2>Players</h2>
        <div class="status" id="status"></div>
        <div class="dice-display" id="diceDisplay">
          <div class="dice-container">
            <div class="dice-icon">🎲</div>
            <div class="dice-result" id="diceResult">-</div>
          </div>
        </div>
        <div style="display:flex; gap:8px; margin-bottom:8px;">
          <button onclick="quickRoll()">Roll</button>
          <button onclick="quickPass()">Skip Turn</button>
          <button onclick="quickBuyDevCard()">Buy Dev Card</button>
        </div>
        <div class="actions-toggle" id="actionsToggle">Show Actions ▼</div>
        <div class="actions" id="actions"></div>
      <div class="discard-panel" id="discardPanel" style="display:none;">
        <strong>Discard Resources</strong>
        <div class="mono" id="discardHint"></div>
        <div class="discard-grid">
          <input type="number" min="0" id="discard-brick" placeholder="Brick">
          <input type="number" min="0" id="discard-lumber" placeholder="Wood">
          <input type="number" min="0" id="discard-ore" placeholder="Ore">
          <input type="number" min="0" id="discard-grain" placeholder="Grain">
          <input type="number" min="0" id="discard-wool" placeholder="Sheep">
        </div>
        <button onclick="submitDiscard()">Submit Discard</button>
      </div>
      <div class="trade-panel" id="tradePanel" style="display:none;">
        <strong>Trade</strong>
        <div class="mono" id="tradeHint"></div>
        <div><em>Bank Trade (4:1)</em></div>
        <div class="trade-row">
          <select id="bankGive"></select>
          <select id="bankReceive"></select>
        </div>
        <button onclick="submitBankTrade()">Trade with Bank</button>
        <div><em>Player Trade (1:1 proposal)</em></div>
        <div class="trade-row">
          <select id="playerTarget"></select>
          <select id="playerGive"></select>
          <select id="playerReceive"></select>
        </div>
        <button onclick="submitPlayerTrade()">Propose Player Trade</button>
      </div>
      <div class="footer-meta" id="footer"></div>
    </div>
  </div>
  <script>
    const stateEl = document.getElementById('status');
    const phaseEl = document.getElementById('phase');
    const boardEl = document.getElementById('board');
    const footerEl = document.getElementById('footer');
    const discardPanel = document.getElementById('discardPanel');
    const discardHint = document.getElementById('discardHint');
    const tradePanel = document.getElementById('tradePanel');
    const tradeHint = document.getElementById('tradeHint');
    const bankGive = document.getElementById('bankGive');
    const bankReceive = document.getElementById('bankReceive');
    const playerTarget = document.getElementById('playerTarget');
    const playerGive = document.getElementById('playerGive');
    const playerReceive = document.getElementById('playerReceive');
    const actionsToggle = document.getElementById('actionsToggle');
    const actionsEl = document.getElementById('actions');
    let currentPlayerId = 0;
    let actionsOpen = false;
    let previousRoll = null;
    let previousWinner = null;

    const resourceCards = {
      brick: '/static/cards/Resource_Brick.png',
      lumber: '/static/cards/Resource_Wood.png',
      ore: '/static/cards/Resource_Ore.png',
      grain: '/static/cards/Resource_Grain.png',
      wool: '/static/cards/Resource_Sheep.png'
    };

    const devCards = {
      knight: '/static/cards/DevCard_Knight.png',
      victory_point: '/static/cards/DevCard_1VP.png',
      monopoly: '/static/cards/DevCard_Monopoly.png',
      year_of_plenty: '/static/cards/DevCard_YearOfPlenty.png',
      road_building: '/static/cards/DevCard_RoadBuilding.png'
    };

    const tileTextures = {
      brick: '/static/tiles/Tile_Brick.jpg',
      lumber: '/static/tiles/Tile_Wood.jpg',
      ore: '/static/tiles/Tile_Ore.jpg',
      grain: '/static/tiles/Tile_Grain.jpg',
      wool: '/static/tiles/Tile_Sheep.jpg',
      desert: '/static/tiles/Tile_Desert.jpg',
      sea: '/static/tiles/Tile_Sea.jpg'
    };

    const portTextures = {
      '3-1': '/static/tiles/Tile_3-1_Port.jpg',
      brick: '/static/tiles/Tile_Brick-Port.jpg',
      lumber: '/static/tiles/Tile_Wood-Port.jpg',
      ore: '/static/tiles/Tile_Ore-Port.jpg',
      grain: '/static/tiles/Tile_Grain-Port.jpg',
      wool: '/static/tiles/Tile_Sheep-Port.jpg'
    };

    const playerColors = ['#c7503a', '#3578a7', '#2e8b57', '#c07a2a'];

    const TILE_RADIUS = 18;
    const TILE_TEXTURE_SIZE = 2.4;
    const SQRT3 = Math.sqrt(3);
    const SX = (SQRT3 / 4) * TILE_RADIUS;
    const SY = TILE_RADIUS / 4;
    const TILE_WIDTH = SQRT3 * TILE_RADIUS;
    const TILE_HEIGHT = 2 * TILE_RADIUS;

    function axialToPixel(q, r) {
      const x = TILE_RADIUS * SQRT3 * (q + r / 2);
      const y = TILE_RADIUS * 1.5 * r;
      return [x, y];
    }

    function vertexToPixel(coord) {
      return [coord[0] * SX, coord[1] * SY];
    }

    function hexPoints(cx, cy) {
      const offsets = [
        [0, TILE_RADIUS],
        [SQRT3 * TILE_RADIUS / 2, TILE_RADIUS / 2],
        [SQRT3 * TILE_RADIUS / 2, -TILE_RADIUS / 2],
        [0, -TILE_RADIUS],
        [-SQRT3 * TILE_RADIUS / 2, -TILE_RADIUS / 2],
        [-SQRT3 * TILE_RADIUS / 2, TILE_RADIUS / 2]
      ];
      return offsets.map(([dx, dy]) => `${cx + dx},${cy + dy}`).join(' ');
    }

    function ringCoords(radius) {
      const dirs = [
        [1, 0], [1, -1], [0, -1],
        [-1, 0], [-1, 1], [0, 1]
      ];
      let q = radius;
      let r = 0;
      const coords = [];
      for (let side = 0; side < 6; side++) {
        const [dq, dr] = dirs[side];
        for (let step = 0; step < radius; step++) {
          coords.push([q, r]);
          q += dq;
          r += dr;
        }
      }
      return coords;
    }

    function allTilesWithRadius(radius) {
      const coords = [];
      for (let q = -radius; q <= radius; q++) {
        for (let r = -radius; r <= radius; r++) {
          if (-radius <= q + r && q + r <= radius) {
            coords.push([q, r]);
          }
        }
      }
      return coords;
    }

    function tileColor(resource) {
      const map = {
        brick: '#b6653a',
        lumber: '#4f8a5b',
        ore: '#7b7e8b',
        grain: '#d8b45d',
        wool: '#7fbf7f',
        desert: '#d9c2a2',
        sea: '#5b7ea4'
      };
      return map[resource] || '#ccc';
    }

    function tileFill(resource) {
      if (tileTextures[resource]) {
        return `url(#tile-${resource})`;
      }
      return tileColor(resource);
    }

    function render(state) {
      phaseEl.textContent = `Turn ${state.turn_index} | Player ${state.current_player} | ${state.phase}`;
      footerEl.textContent = state.last_roll ? `Last roll: ${state.last_roll}` : '';

      // Check for game winner and play win sound
      if (state.winner !== null && state.winner !== previousWinner) {
        playWinSound();
        previousWinner = state.winner;
      }

      // Update dice display and play sound if new roll
      const diceResult = document.getElementById('diceResult');
      if (state.last_roll) {
        diceResult.textContent = state.last_roll;

        // Play dice sound if this is a new roll
        if (previousRoll !== state.last_roll && previousRoll !== null) {
          playDiceSound();
        }
        previousRoll = state.last_roll;
      } else {
        diceResult.textContent = '-';
      }

      // Update dev deck count
      document.getElementById('devDeckCount').textContent = state.dev_deck_size;

      // Update resource bank counts
      document.getElementById('brickCount').textContent = state.bank.brick;
      document.getElementById('lumberCount').textContent = state.bank.lumber;
      document.getElementById('oreCount').textContent = state.bank.ore;
      document.getElementById('grainCount').textContent = state.bank.grain;
      document.getElementById('woolCount').textContent = state.bank.wool;

      // Update building counts (current player's pieces)
      document.getElementById('settlementCount').textContent = state.current_player_pieces.settlements;
      document.getElementById('cityCount').textContent = state.current_player_pieces.cities;
      document.getElementById('roadCount').textContent = state.current_player_pieces.roads;

      // Update building icon hues based on current player
      const playerHues = [0, 210, 120, 30]; // Red, Blue, Green, Orange
      const currentHue = playerHues[state.current_player % playerHues.length];

      document.getElementById('settlementImg').style.filter = `hue-rotate(${currentHue}deg)`;
      document.getElementById('cityImg').style.filter = `hue-rotate(${currentHue}deg)`;
      document.getElementById('roadImg').style.filter = `hue-rotate(${currentHue}deg)`;

      // Update player color legend
      const playerColorLegend = document.getElementById('playerColorLegend');
      playerColorLegend.innerHTML = '';
      for (let i = 0; i < 4; i++) {
        const playerItem = document.createElement('div');
        playerItem.className = 'legend-player-item';
        const playerColor = playerColors[i % playerColors.length];
        playerItem.innerHTML = `<div class="legend-player-dot" style="background-color: ${playerColor}"></div>P${i}`;
        playerColorLegend.appendChild(playerItem);
      }

      currentPlayerId = state.current_player;
      discardPanel.style.display = state.phase === 'discard' ? 'grid' : 'none';
      if (state.phase === 'discard') {
        const required = state.pending_discards?.[String(state.current_player)] ?? 0;
        discardHint.textContent = `Discard ${required} cards.`;
      } else {
        discardHint.textContent = '';
      }
      tradePanel.style.display = state.phase === 'main' ? 'grid' : 'none';
      tradeHint.textContent = state.phase === 'main' ? 'Trade with bank (4:1) or propose a 1:1 player trade.' : '';
      if (state.phase === 'main') {
        populateTradeSelectors(state);
      }

      stateEl.innerHTML = '';
      state.players.forEach(player => {
        const div = document.createElement('div');
        div.className = 'player-card' + (player.player_id === state.current_player ? ' active' : '');

        // Add player color indicator
        const playerColor = playerColors[player.player_id % playerColors.length];
        div.style.borderLeft = `5px solid ${playerColor}`;
        div.style.boxShadow = `0 0 0 1px ${playerColor}20`; // 20% opacity color border

        const header = document.createElement('div');
        header.className = 'player-header';

        // Add colored player indicator dot
        const playerDot = document.createElement('span');
        playerDot.className = 'player-color-dot';
        playerDot.style.backgroundColor = playerColor;

        header.innerHTML = `<span class="player-info"><span class="player-color-dot" style="background-color: ${playerColor}"></span>P${player.player_id}</span><span>VP ${player.victory_points} | Knights ${player.knights_played}</span>`;

        const resources = document.createElement('div');
        resources.className = 'resource-row';
        ['brick', 'lumber', 'ore', 'grain', 'wool'].forEach(res => {
          const cell = document.createElement('div');
          cell.className = 'resource';
          cell.innerHTML = `<img src="${resourceCards[res]}" alt="${res}"/><div>${player.resources[res]}</div>`;
          resources.appendChild(cell);
        });

        // Development cards section
        const devCardCounts = {};
        player.dev_cards.forEach(card => {
          devCardCounts[card] = (devCardCounts[card] || 0) + 1;
        });

        if (Object.keys(devCardCounts).length > 0) {
          const devCardsRow = document.createElement('div');
          devCardsRow.className = 'dev-cards-row';
          Object.entries(devCardCounts).forEach(([cardType, count]) => {
            const cell = document.createElement('div');
            cell.className = 'dev-card';

            // Show cards only if it's the current player's turn, otherwise show card backs
            let imgSrc, displayCard;
            if (player.player_id === state.current_player) {
              // Current player can see their own cards
              displayCard = cardType;
              imgSrc = devCards[cardType];
            } else {
              // Other players see card backs
              displayCard = 'CardBack1';
              imgSrc = '/static/cards/CardBack1.png';
            }

            cell.innerHTML = `<img src="${imgSrc}" alt="${displayCard}"/><div>${count}</div>`;
            devCardsRow.appendChild(cell);
          });
          div.appendChild(devCardsRow);
        }

        div.appendChild(header);
        div.appendChild(resources);
        stateEl.appendChild(div);
      });

      actionsEl.innerHTML = '';
      const legalMap = {
        build_settlement: new Set(),
        build_city: new Set(),
        build_road: new Set(),
        move_robber: new Set()
      };
      state.legal_actions.forEach(action => {
        const payload = action.payload || {};
        if (action.action_type === 'build_settlement' && payload.vertex_id !== undefined) {
          legalMap.build_settlement.add(payload.vertex_id);
        }
        if (action.action_type === 'build_city' && payload.vertex_id !== undefined) {
          legalMap.build_city.add(payload.vertex_id);
        }
        if (action.action_type === 'build_road' && payload.edge_id !== undefined) {
          legalMap.build_road.add(payload.edge_id);
        }
        if (action.action_type === 'move_robber' && payload.tile_id !== undefined) {
          legalMap.move_robber.add(payload.tile_id);
        }
      });

      state.legal_actions.slice(0, 200).forEach(action => {
        const btn = document.createElement('button');
        const actionName = action.action_type.replace('_', ' ');
        btn.textContent = `${actionName} ${JSON.stringify(action.payload)}`;
        btn.onclick = () => applyAction(action);
        actionsEl.appendChild(btn);
      });
      actionsEl.className = actionsOpen ? 'actions open' : 'actions';
      actionsToggle.textContent = actionsOpen ? 'Hide Actions ▲' : 'Show Actions ▼';

      boardEl.innerHTML = '';
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      defs.innerHTML = `
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="rgba(0,0,0,0.35)" />
        </filter>
      `;
      boardEl.appendChild(defs);

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      let clipCounter = 0;

      const outerRing = ringCoords(3);
      const portIndices = [0, 2, 4, 6, 8, 10, 12, 14, 16];
      const portOrder = ['3-1', 'brick', '3-1', 'lumber', '3-1', 'ore', '3-1', 'grain', 'wool'];
      const portIndexSet = new Set(portIndices);
      const portTiles = portIndices.map((idx, i) => ({
        axial: outerRing[idx],
        portType: portOrder[i]
      }));
      const seaTiles = outerRing.filter((_, idx) => !portIndexSet.has(idx)).map(axial => ({ axial }));
      const perimeter = new Set(outerRing.map(axial => `${axial[0]},${axial[1]}`));
      const landCoords = new Set(allTilesWithRadius(2).map(axial => `${axial[0]},${axial[1]}`));

      function updateBounds(points) {
        points.forEach(([px, py]) => {
          minX = Math.min(minX, px);
          minY = Math.min(minY, py);
          maxX = Math.max(maxX, px);
          maxY = Math.max(maxY, py);
        });
      }

      function drawHexTile(cx, cy, points, textureUrl, strokeColor) {
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', points.map(([px, py]) => `${px},${py}`).join(' '));
        poly.setAttribute('fill', tileColor('sea'));
        poly.setAttribute('stroke', strokeColor || '#42362d');
        poly.setAttribute('stroke-width', '1.2');
        poly.setAttribute('filter', 'url(#shadow)');
        boardEl.appendChild(poly);

        if (!textureUrl) {
          return;
        }
        const clip = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
        const clipId = `clip-${clipCounter++}`;
        clip.setAttribute('id', clipId);
        const clipPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        clipPoly.setAttribute('points', points.map(([px, py]) => `${px},${py}`).join(' '));
        clip.appendChild(clipPoly);
        defs.appendChild(clip);

        const img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
        const imgSize = TILE_RADIUS * TILE_TEXTURE_SIZE;
        img.setAttribute('href', textureUrl);
        img.setAttribute('x', `${cx - imgSize / 2}`);
        img.setAttribute('y', `${cy - imgSize / 2}`);
        img.setAttribute('width', `${imgSize}`);
        img.setAttribute('height', `${imgSize}`);
        img.setAttribute('preserveAspectRatio', 'xMidYMid slice');
        img.setAttribute('clip-path', `url(#${clipId})`);
        boardEl.appendChild(img);
      }

      const landCenters = [];
      state.tiles.forEach(tile => {
        landCenters.push(axialToPixel(tile.axial[0], tile.axial[1]));
      });
      const landCenter = landCenters.reduce(
        (acc, [x, y]) => [acc[0] + x, acc[1] + y],
        [0, 0]
      ).map(v => v / landCenters.length);

      const ringCenters = [];
      seaTiles.forEach(tile => ringCenters.push(axialToPixel(tile.axial[0], tile.axial[1])));
      portTiles.forEach(tile => ringCenters.push(axialToPixel(tile.axial[0], tile.axial[1])));
      const ringCenter = ringCenters.reduce(
        (acc, [x, y]) => [acc[0] + x, acc[1] + y],
        [0, 0]
      ).map(v => v / ringCenters.length);

      const dx = landCenter[0] - ringCenter[0];
      const dy = landCenter[1] - ringCenter[1];

      seaTiles.forEach(tile => {
        const [cxRaw, cyRaw] = axialToPixel(tile.axial[0], tile.axial[1]);
        const cx = cxRaw + dx;
        const cy = cyRaw + dy;
        const points = [
          [cx, cy + TILE_RADIUS],
          [cx + SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2],
          [cx + SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx, cy - TILE_RADIUS],
          [cx - SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx - SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2]
        ];
        updateBounds(points);
        drawHexTile(cx, cy, points, tileTextures.sea, '#65564a');
      });

      portTiles.forEach(tile => {
        const [cxRaw, cyRaw] = axialToPixel(tile.axial[0], tile.axial[1]);
        const cx = cxRaw + dx;
        const cy = cyRaw + dy;
        const points = [
          [cx, cy + TILE_RADIUS],
          [cx + SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2],
          [cx + SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx, cy - TILE_RADIUS],
          [cx - SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx - SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2]
        ];
        updateBounds(points);
        drawHexTile(cx, cy, points, portTextures[tile.portType], '#6b4f3c');
      });

      state.tiles.forEach(tile => {
        const coordKey = `${tile.axial[0]},${tile.axial[1]}`;
        if (perimeter.has(coordKey) || !landCoords.has(coordKey)) {
          return;
        }
        const [cx, cy] = axialToPixel(tile.axial[0], tile.axial[1]);
        const points = [
          [cx, cy + TILE_RADIUS],
          [cx + SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2],
          [cx + SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx, cy - TILE_RADIUS],
          [cx - SQRT3 * TILE_RADIUS / 2, cy - TILE_RADIUS / 2],
          [cx - SQRT3 * TILE_RADIUS / 2, cy + TILE_RADIUS / 2]
        ];
        updateBounds(points);
        drawHexTile(cx, cy, points, tileTextures[tile.resource], '#42362d');

        if (tile.number_token) {
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('cx', cx);
          circle.setAttribute('cy', cy);
          circle.setAttribute('r', '7.5');
          circle.setAttribute('fill', '#f7f0e4');
          circle.setAttribute('stroke', '#41362d');
          circle.setAttribute('stroke-width', '1');
          boardEl.appendChild(circle);

          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', cx);
          text.setAttribute('y', cy + 3);
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('font-size', '7');
          text.setAttribute('font-weight', '700');
          text.setAttribute('fill', tile.number_token === 6 || tile.number_token === 8 ? '#c74a3a' : '#2c2622');
          text.textContent = tile.number_token;
          boardEl.appendChild(text);
        }

        if (tile.tile_id === state.robber_tile) {
          const robber = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          robber.setAttribute('cx', cx);
          robber.setAttribute('cy', cy);
          robber.setAttribute('r', '5');
          robber.setAttribute('fill', '#1f1c1a');
          boardEl.appendChild(robber);
        }

        if (legalMap.move_robber.has(tile.tile_id)) {
          const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
          overlay.setAttribute('points', points.map(([px, py]) => `${px},${py}`).join(' '));
          overlay.setAttribute('class', 'board-overlay tile active');
          overlay.setAttribute('fill', 'rgba(199,74,58,0.15)');
          overlay.addEventListener('click', () => {
            applyAction({ action_type: 'move_robber', payload: { tile_id: tile.tile_id } });
          });
          boardEl.appendChild(overlay);
        }
      });

      const pad = 18;
      const width = Math.max(10, maxX - minX + pad * 2);
      const height = Math.max(10, maxY - minY + pad * 2);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const size = Math.max(width, height);
      boardEl.setAttribute('viewBox', `${cx - size / 2 - pad} ${cy - size / 2 - pad} ${size + pad * 2} ${size + pad * 2}`);

      state.edges.forEach(edge => {
        const vA = state.vertices.find(v => v.vertex_id === edge.vertex_a);
        const vB = state.vertices.find(v => v.vertex_id === edge.vertex_b);
        if (!vA || !vB) {
          return;
        }
        const [x1, y1] = vertexToPixel(vA.coord);
        const [x2, y2] = vertexToPixel(vB.coord);

        const edgeLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        if (legalMap.build_road.has(edge.edge_id)) {
          edgeLine.setAttribute('class', 'board-overlay edge active');
          edgeLine.addEventListener('click', () => {
            applyAction({ action_type: 'build_road', payload: { edge_id: edge.edge_id } });
          });
        } else {
          edgeLine.setAttribute('class', 'board-overlay edge');
        }
        edgeLine.setAttribute('x1', x1);
        edgeLine.setAttribute('y1', y1);
        edgeLine.setAttribute('x2', x2);
        edgeLine.setAttribute('y2', y2);
        boardEl.appendChild(edgeLine);

        if (edge.occupancy === null || edge.occupancy === undefined) {
          return;
        }
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', playerColors[edge.occupancy % playerColors.length]);
        line.setAttribute('stroke-width', '3');
        line.setAttribute('stroke-linecap', 'round');
        boardEl.appendChild(line);
      });

      state.vertices.forEach(vertex => {
        const [x, y] = vertexToPixel(vertex.coord);
        const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        if (legalMap.build_settlement.has(vertex.vertex_id)) {
          overlay.setAttribute('class', 'board-overlay vertex active');
          overlay.addEventListener('click', () => {
            applyAction({ action_type: 'build_settlement', payload: { vertex_id: vertex.vertex_id } });
          });
        } else if (legalMap.build_city.has(vertex.vertex_id)) {
          overlay.setAttribute('class', 'board-overlay vertex active');
          overlay.addEventListener('click', () => {
            applyAction({ action_type: 'build_city', payload: { vertex_id: vertex.vertex_id } });
          });
        } else {
          overlay.setAttribute('class', 'board-overlay vertex');
        }
        overlay.setAttribute('cx', x);
        overlay.setAttribute('cy', y);
        overlay.setAttribute('r', '3.5');
        boardEl.appendChild(overlay);

        if (!vertex.occupancy) {
          return;
        }
        const owner = vertex.occupancy[0];
        const building = vertex.occupancy[1];
        const color = playerColors[owner % playerColors.length];
        const shape = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        shape.setAttribute('cx', x);
        shape.setAttribute('cy', y);
        shape.setAttribute('r', building === 'city' ? '6' : '4');
        shape.setAttribute('fill', color);
        shape.setAttribute('stroke', '#fff');
        shape.setAttribute('stroke-width', '1');
        boardEl.appendChild(shape);
      });
    }

    function playDiceSound() {
      try {
        // Create audio element and play the dice roll sound
        const audio = new Audio('/static/dice-sound.mp3');
        audio.volume = 0.5;
        audio.play().catch(err => {
          console.log('Dice sound playback failed:', err);
        });
      } catch (err) {
        console.log('Could not play dice sound:', err);
      }
    }

    function playWinSound() {
      try {
        // Create audio element and play the win sound
        const audio = new Audio('/static/win-sound.mp3');
        audio.volume = 0.7;
        audio.play().catch(err => {
          console.log('Win sound playback failed:', err);
        });
      } catch (err) {
        console.log('Could not play win sound:', err);
      }
    }

    function populateTradeSelectors(state) {
      const resources = ['brick', 'lumber', 'ore', 'grain', 'wool'];
      const players = state.players.map(p => p.player_id);

      function fillSelect(selectEl, opts) {
        selectEl.innerHTML = '';
        opts.forEach(val => {
          const opt = document.createElement('option');
          opt.value = val;
          opt.textContent = val;
          selectEl.appendChild(opt);
        });
      }

      fillSelect(bankGive, resources);
      fillSelect(bankReceive, resources);
      fillSelect(playerGive, resources);
      fillSelect(playerReceive, resources);
      fillSelect(playerTarget, players.filter(pid => pid !== currentPlayerId));
    }

    async function loadState() {
      const res = await fetch('/api/state');
      const state = await res.json();
      render(state);
    }

    async function applyAction(action) {
      await fetch('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(action)
      });
      await loadState();
    }

    async function resetGame() {
      await fetch('/api/reset', { method: 'POST' });
      await loadState();
    }

    function quickRoll() {
      applyAction({ action_type: 'roll_dice', payload: {} });
    }

    function quickPass() {
      applyAction({ action_type: 'pass_turn', payload: {} });
    }

    function quickBuyDevCard() {
      applyAction({ action_type: 'buy_dev_card', payload: {} });
    }

    function submitDiscard() {
      const payload = {
        brick: parseInt(document.getElementById('discard-brick').value || '0', 10),
        lumber: parseInt(document.getElementById('discard-lumber').value || '0', 10),
        ore: parseInt(document.getElementById('discard-ore').value || '0', 10),
        grain: parseInt(document.getElementById('discard-grain').value || '0', 10),
        wool: parseInt(document.getElementById('discard-wool').value || '0', 10)
      };
      applyAction({ action_type: 'discard', payload: { player_id: currentPlayerId, resources: payload } });
    }

    function submitBankTrade() {
      const give = bankGive.value;
      const receive = bankReceive.value;
      if (!give || !receive || give === receive) return;
      applyAction({ action_type: 'trade_bank', payload: { give, receive, rate: 4 } });
    }

    function submitPlayerTrade() {
      const target = parseInt(playerTarget.value, 10);
      const give = playerGive.value;
      const receive = playerReceive.value;
      if (!Number.isFinite(target) || !give || !receive || give === receive) return;
      applyAction({
        action_type: 'trade_player',
        payload: { to_player: target, give: { [give]: 1 }, receive: { [receive]: 1 } }
      });
    }

    actionsToggle.addEventListener('click', () => {
      actionsOpen = !actionsOpen;
      actionsEl.className = actionsOpen ? 'actions open' : 'actions';
      actionsToggle.textContent = actionsOpen ? 'Hide Actions ▲' : 'Show Actions ▼';
    });

    loadState();
  </script>
</body>
</html>
"""


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    with _state_lock:
        return _serialize_state(_state)


@app.post("/api/reset")
def reset_game() -> Dict[str, Any]:
    global _state
    with _state_lock:
        _state = initial_game_state(standard_board(seed=42))
        return _serialize_state(_state)


@app.post("/api/action")
def apply_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _state
    try:
        action_type = ActionType(payload["action_type"])
        action = Action(action_type=action_type, payload=payload.get("payload", {}))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with _state_lock:
        try:
            _state = _state.apply(action)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _serialize_state(_state)
