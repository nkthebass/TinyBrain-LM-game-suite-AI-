# TinyBrain Game LM Studio

Train GPT-style transformers from scratch in C# to play games. Built on TorchSharp (PyTorch C# bindings) with Windows Forms UI. No Python required.

Three games included:
- **Chess** — Full chess engine with training, RL (Stockfish), board encoding, thinking/search, AI vs AI battles
- **Tic-Tac-Toe** — Nearly unbeatable at 200K params
- **Blob Shooter** — Top-down 2D arena shooter with rule-based + transformer AI

---

## System Requirements

### Minimum
| Component | Requirement |
|---|---|
| **OS** | Windows 10/11 (64-bit) |
| **GPU** | NVIDIA GTX 900 series or newer (CUDA compute 5.0+) |
| **VRAM** | 4 GB (small models up to ~10M params) |
| **RAM** | 8 GB |
| **Disk** | 2 GB for NuGet packages + model files |
| **Software** | [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) |

### Recommended (for large chess models)
| Component | Requirement |
|---|---|
| **GPU** | RTX 3070 or better |
| **VRAM** | 8+ GB (for 50M-150M param models) |
| **RAM** | 16-32 GB (for large datasets) |
| **Disk** | 20+ GB (datasets can be 6GB+) |

### Optional
- **Stockfish** — Download from [stockfishchess.org](https://stockfishchess.org/download/) for RL training. Place the exe in the Chess folder.

### CPU-Only (No NVIDIA GPU)
Replace `TorchSharp-cuda-windows` with `TorchSharp-cpu` in the `.csproj` file. Training will be significantly slower but inference is playable for small models.

---

## Quick Start (All Games)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/TinyBrain-Game-LM-Studio.git
cd TinyBrain-Game-LM-Studio

# Pick a game and run it
cd Chess
dotnet run

# Or use the batch file
run.bat
```

NuGet packages (TorchSharp + CUDA) download automatically on first build (~2GB).

---

# Chess

A full chess-playing transformer with dataset downloading, supervised training, Stockfish RL, board position encoding, and a thinking/search system.

## Setup

```bash
cd Chess
dotnet run
```

The app opens with a chess board and settings panel. First time setup:

### 1. Get Training Data

Click **Datasets** in the bottom bar. Options:

| Dataset | Quality | Size | Time |
|---|---|---|---|
| Self-Play (Quick Test) | Low | 1K games | ~10 sec |
| Lichess 2015-01 (2000+) | High | ~70K games | ~5 min download |
| **★★★ Elite 2021 Full Year** | Elite (2200+) | ~8M games | ~30 min download |
| **★★ KingBase Lite 2019** | Elite (2200+) | 1M+ games | ~5 min download |

For your first model, start with **Lichess 2015-01 (2000+)** — it's small but high quality. For serious training, use the Elite datasets.

Check **"Append to existing data"** to combine multiple datasets.

### 2. Configure the Model

In the settings panel on the right:

| Setting | Small Model | Medium Model | Large Model |
|---|---|---|---|
| Embed Dim | 64 | 256 | 1024 |
| Num Heads | 8 | 8 | 8 |
| FF Dim | 256 | 1024 | 4096 |
| Num Layers | 4 | 8 | 12 |
| Context Size | 384 | 512 | 768 |
| **~Params** | **~200K** | **~25M** | **~150M** |

**Important:** Embed Dim must be divisible by Num Heads.

### 3. Train

Click **Start Training** in the TRAINING section of the settings panel. Training settings:

| Setting | Recommendation |
|---|---|
| Steps | 20,000 for small, 60,000+ for large |
| Batch Size | 2-4 for large models, 16-32 for small |
| Grad Accum | 8-16 (effective batch = Batch × Accum) |
| Learning Rate | 0.0003 for fresh training, 0.00001 for continuing |
| Warmup Steps | 1,000-3,000 |

Training progress shows in the header bar with loss, step count, and ETA. Checkpoints save every ~1%. You can close and reopen — training resumes automatically.

### 4. Play

Click **Play White** or **Play Black** in the bottom bar.
- Click a piece to select it, click a destination to move
- Click **Draw** to end a game as a draw
- The AI uses board position encoding automatically

### 5. Improve with RL Training (Optional)

Requires [Stockfish](https://stockfishchess.org/download/) — place the exe in the Chess folder.

Click **Run RL Training** in the RL TRAINING section. The AI plays games against itself, Stockfish corrects bad moves, then the model fine-tunes on the corrections.

| Setting | What it does |
|---|---|
| SF Depth | How deep Stockfish searches (10-15 recommended) |
| Self-Play Games | Games to generate (200-1000) |
| Start at Move | Skip opening to focus on midgame/endgame (20-50) |
| CP Loss Thresh | How bad a move must be to get corrected (30-50 centipawns) |

### 6. Board Prefix Training (Optional)

Click **Run Board Training** in the BOARD PREFIX section. This teaches the model to read board position tokens, making it significantly stronger.

### 7. Enable Thinking (Optional)

In the INFERENCE section, check **Enable Thinking (lookahead search)**. This adds a 1-ply tactical search:
- The model picks its top 5 candidate moves
- For each, it simulates the opponent's best responses
- It avoids moves that lose material and finds moves that win material
- ~150ms per move on GPU — no noticeable delay

### 8. AI vs AI Battles

Click **AI vs AI** to open the battle window. Load different models on each side and watch them play. Toggle **Board** and **Think** per side to compare configurations.

### Files to Share (Trained Models)

To share a trained model, copy these 3 files:
- `chess_model.pt` — model weights
- `chess_arch.json` — architecture config
- `chess_tokenizer.bin` — tokenizer

---

# Tic-Tac-Toe

A tiny transformer that plays perfect tic-tac-toe. **A pre-trained model is included** — you can play immediately without training.

## Setup

```bash
cd TicTacToe
dotnet run
```

### Playing (Model Included)

The included model (~200K params) plays nearly perfect tic-tac-toe. Just click **Play as X** or **Play as O** and click board positions to make your move.

The board shows position numbers (0-8) for empty squares:
```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

You will almost certainly draw every game — perfect tic-tac-toe play always results in a draw.

### Training Your Own Model (Optional)

If you want to train from scratch:

1. Delete `ttt_model.pt`, `ttt_arch.json`, and `ttt_training_state.json`
2. Set model config (defaults work great: Embed=64, Heads=8, FF=256, Layers=4)
3. Click **Start Training** — generates 60K games and trains in ~2 minutes on GPU
4. Click **Play as X** or **Play as O**

### Default Config (Recommended)

| Setting | Value |
|---|---|
| Embed Dim | 64 |
| Num Heads | 8 |
| FF Dim | 256 |
| Num Layers | 4 |
| Context Size | 32 |
| Steps | 8,000 |
| Training Games | 60,000 |

This produces a ~200K param model that plays essentially perfect tic-tac-toe.

---

# Blob Shooter

A top-down 2D arena shooter where you fight an AI blob. Includes both rule-based AI and a trainable transformer AI.

## Setup

```bash
cd BlobShooter
dotnet run
```

### Controls

| Key | Action |
|---|---|
| **W/A/S/D** | Move |
| **Mouse** | Aim |
| **Left Click** | Shoot |

### Playing

1. Click **New Game** to start
2. You are the **blue** circle, the AI is **red**
3. Shoot the AI before it shoots you
4. Health bars show above each player

### Map Editing

1. Click **Edit Map** to enter edit mode
2. Click grid cells to toggle walls on/off
3. Click **Done** when finished
4. Walls block bullets and movement — use them for cover

### AI Modes

Switch between AI types using the **AI Mode** dropdown in the settings:

- **Rule-Based** — Hard-coded behavior (approach, strafe, retreat, use cover). Works immediately.
- **Transformer** — Trained neural network. Requires training first.

### Training the Transformer AI

1. Set AI Mode to **Rule-Based** first (generates training data from rule-based play)
2. Configure model settings (defaults: Embed=64, Heads=4, FF=256, Layers=4, Context=256)
3. Click **Train AI** — generates 50K self-play games and trains
4. Switch AI Mode to **Transformer**
5. Click **New Game**

### Settings

| Section | Settings |
|---|---|
| **Map** | Grid Width/Height (4-32, default 16) |
| **Gameplay** | Player/AI speed, bullet speed/damage, health, fire rate |
| **Model** | Architecture (Embed, Heads, FF, Layers, Context) |
| **Training** | Steps, games, batch size, learning rate, etc. |
| **Inference** | Temperature (lower = more deterministic) |

---

## Project Structure

Each game is a self-contained .NET 8 project:

```
TinyBrain Game LM Studio/
├── Chess/              ← Full chess with training, RL, thinking
│   ├── Chess.csproj
│   ├── ChessGame.cs        # Chess rules engine
│   ├── ChessBoardPanel.cs  # Board rendering
│   ├── Player.cs            # Model inference
│   ├── ThinkingPlayer.cs    # Lookahead search
│   ├── Trainer.cs           # Supervised training
│   ├── RLTrainer.cs         # Stockfish RL training
│   ├── BoardPrefixTrainer.cs # Board encoding training
│   ├── BoardEncoder.cs      # Position → token encoding
│   ├── StockfishEvaluator.cs # UCI Stockfish wrapper
│   ├── DataGenerator.cs     # Self-play data generation
│   ├── DatasetForm.cs       # Dataset browser/downloader
│   ├── PgnConverter.cs      # PGN → UCI conversion
│   ├── AIBattleForm.cs      # AI vs AI battle window
│   ├── MainForm.cs          # Main UI
│   └── (shared transformer files)
│
├── TicTacToe/          ← Includes pre-trained model
│   ├── TicTacToe.csproj
│   ├── TicTacToeGame.cs
│   ├── GameDataGenerator.cs
│   ├── Trainer.cs
│   ├── Player.cs
│   ├── MainForm.cs
│   ├── ttt_model.pt         # Pre-trained model (included)
│   └── (shared transformer files)
│
├── BlobShooter/        ← Top-down 2D shooter
│   ├── BlobShooter.csproj
│   ├── GamePanel.cs
│   ├── GameSimulation.cs
│   ├── RuleBasedAI.cs
│   ├── TransformerAI.cs
│   ├── StateEncoder.cs
│   ├── Trainer.cs
│   ├── MainForm.cs
│   └── (shared transformer files)
│
└── README.md
```

### Shared Transformer Files (in each project)

| File | Purpose |
|---|---|
| `TransformerModel.cs` | GPT-2 style causal transformer |
| `TransformerBlock.cs` | Pre-LayerNorm transformer layer |
| `CausalSelfAttention.cs` | Multi-head causal self-attention |
| `FeedForward.cs` | Position-wise MLP (GELU) |
| `CharTokenizer.cs` | Character-level tokenizer |

---

## License

MIT

## Credits

- **Claude (Anthropic)** — AI assistant that helped designed and write the entire codebase, from the transformer architecture to the training loops to the UI
- **TorchSharp** — .NET bindings for PyTorch
- **Lichess** — Open chess game database (CC0)
- **KingBase** — Curated high-Elo chess database
- **Stockfish** — Open source chess engine (for RL training)
