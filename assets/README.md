# Catan Game Assets

This directory contains all visual and audio assets for the Catan game.

## 📁 Directory Structure

```
assets/
├── images/                 # All visual assets
│   ├── cards/              # Development cards and card backs
│   │   ├── DevCard_*.png   # Development card graphics
│   │   └── CardBack*.png   # Card back designs
│   ├── dice/               # Dice graphics
│   │   ├── 1.png - 6.png   # Dice face images
│   │   └── solid_white.png # Base dice texture
│   ├── tiles/              # Hex tile graphics
│   │   ├── Tile_*.jpg      # Resource tile images
│   │   └── Port tiles      # Harbor/port graphics
│   ├── pieces/             # Game piece graphics
│   │   ├── Piece-City.png      # City piece
│   │   ├── Piece-Road.png      # Road piece
│   │   └── Piece-Settlement.png # Settlement piece
│   └── ui/                 # UI elements (future)
├── sounds/                 # Audio assets
│   ├── dice-sound.mp3      # Dice roll sound effect
│   └── win-sound.mp3       # Victory sound effect
├── fonts/                  # Typography (future)
└── data/                   # Asset metadata (future)
```

## 🎨 Asset Categories

### Development Cards
- Knight cards for robber movement
- Monopoly, Year of Plenty, Road Building
- Victory Point cards
- Card backs for hidden information

### Dice
- Standard 6-sided dice faces (1-6)
- White dice theme for clean UI

### Tiles
- Resource tiles: Brick, Lumber, Ore, Grain, Wool
- Special tiles: Desert
- Port tiles: 3:1 and 2:1 resource ports

### Game Pieces
- Settlement markers
- City markers
- Road segments

### Audio
- Dice roll sound effects
- Victory/achievement sounds

## 🔧 Usage in Code

```python
from catan_rl.utils.assets import AssetManager

# Load game assets
assets = AssetManager()
dice_image = assets.get_image("dice/3.png")
victory_sound = assets.get_sound("win-sound.mp3")
```

## 📝 Asset Guidelines

- **Images**: PNG for transparency, JPG for photos
- **Sounds**: MP3 format, < 1MB file size
- **Naming**: Descriptive, lowercase with hyphens
- **Resolution**: Optimize for web display

## 📄 License

Assets are for research and educational use only.
Original Catan game assets are property of Catan Studio/Kosmos.