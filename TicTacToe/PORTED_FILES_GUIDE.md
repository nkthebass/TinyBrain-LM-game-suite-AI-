# Ported Files Guide — Game LLM Project

This guide explains the files ported from TinyBrainBot for use in the game-playing LLM project (starting with Tic-Tac-Toe). Read this before modifying anything.

---

## Architecture Overview

These files implement a small GPT-2 style causal transformer using **TorchSharp** (PyTorch C# bindings). The stack is:

```
CharTokenizer  →  TransformerModel
                      ├── TransformerBlock × N
                      │       ├── CausalSelfAttention
                      │       └── FeedForward
                      └── (token embed + pos embed + final LN + lm head)
```

---

## File-by-File

### `CharTokenizer.cs`
Simple character-level tokenizer. Maps each unique character to an integer ID.

**Key methods:**
- `Build(string text)` — scans text and assigns IDs to all unique chars
- `BuildAscii()` — fixed vocab of tab, newline, and printable ASCII (0x20–0x7E), 97 tokens total. Use this if you want a stable vocab without scanning data.
- `Encode(string text)` → `int[]` — turns a string into token IDs (skips unknown chars silently)
- `Decode(IEnumerable<int> ids)` → `string` — turns token IDs back into a string
- `Save(string path)` / `Load(string path)` — binary serialization

**For tic-tac-toe:**
Don't use `BuildAscii()`. Instead call `Build(...)` with a string containing every character your board format uses (e.g., `"XO. /\n"`) so the vocab stays tiny (maybe 6–10 tokens). Smaller vocab = smaller embedding table = faster training.

**`VocabSize`** is just `CharToId.Count` — use this when constructing `TransformerModel`.

---

### `TransformerModel.cs`
The full GPT-style model. Takes a batch of token ID sequences and outputs logits (unnormalized scores over vocab) for the next token at every position.

**Constructor:**
```csharp
new TransformerModel(
    name: "ttt_model",
    vocabSize: tokenizer.VocabSize,   // must match your tokenizer
    contextSize: 64,                  // max sequence length — keep small for tic-tac-toe
    embedDim: 64,                     // hidden dimension — tiny is fine
    numHeads: 4,                      // must divide embedDim evenly
    ffDim: 256,                       // typically 4× embedDim
    numLayers: 4,                     // number of transformer blocks
    dropout: 0.0                      // use 0 for inference, small value (0.1) for training
)
```

**Forward pass:**
```csharp
// input: (batch_size, seq_len) tensor of int64 token IDs
// output: (batch_size, seq_len, vocab_size) tensor of float32 logits
Tensor logits = model.forward(inputIds);
```

**Loss:**
```csharp
var lossFn = nn.CrossEntropyLoss();
Tensor loss = model.ComputeLoss(logits, targets, lossFn);
// targets: same shape as input (B, T), shifted by 1 for next-token prediction
```

**Important:**
- Call `model.to(device)` before training (device = `CUDA` or `CPU`)
- Call `model.train()` during training, `model.eval()` during inference
- Save/load with `model.save(path)` / `model.load(path)`
- The first forward pass prints tensor shapes to console (debug mode) — this is intentional and disables itself after the first call

**Recommended config for tic-tac-toe:**
```
contextSize: 64     (a full game is ~50 chars)
embedDim:    64
numHeads:    4
ffDim:       256
numLayers:   4
```
This gives roughly ~500K parameters — more than enough to memorize all tic-tac-toe strategy.

---

### `TransformerBlock.cs`
One transformer layer. Pre-LayerNorm layout (GPT-2 style):
```
x → LayerNorm → Attention → + x
x → LayerNorm → FeedForward → + x
```
Pre-LN is more stable to train than post-LN. You don't need to modify this file.

---

### `CausalSelfAttention.cs`
Multi-head self-attention with a causal mask — tokens can only attend to previous positions (not future ones). This is what makes it a language model rather than a bidirectional encoder.

**Causal mask note:** Built once at construction for `maxSeqLen` positions, then sliced to actual sequence length at runtime. It auto-moves to the correct device (CPU/GPU) on first use.

You don't need to modify this file for tic-tac-toe.

---

### `FeedForward.cs`
Two-layer MLP applied position-wise:
```
x → Linear(embedDim → ffDim) → GELU → Dropout → Linear(ffDim → embedDim)
```
You don't need to modify this file.

---

### `ModelConfig.cs`
Dataclass for all hyperparameters. Loads/saves from `model_config.json` next to the executable.

**Fields to care about for game LLM:**
- `EmbedDim`, `NumHeads`, `FfDim`, `NumLayers`, `ContextSize` — architecture
- `Steps`, `LearningRate`, `BatchSize`, `WarmupSteps`, `GradientClip`, `WeightDecay`, `GradAccumSteps` — training
- `Temperature`, `TopK`, `RepetitionPenalty`, `RepPenaltyWindow` — generation
- `BaseVocabSize`, `BpeMerges` — **not relevant** if you're using CharTokenizer instead of BPE. You can repurpose these fields or just hardcode vocab size.

**`EstimatedVramGB`** is useful — check it before training to make sure the config fits in VRAM.

---

## What You Need to Write

The ported files handle the model itself. You need to write:

1. **Training data generator** — produces game sequences as strings (board states + moves)
2. **Training loop** — batch sampling, forward pass, loss, backward, optimizer step, checkpoint save
3. **Inference / generation loop** — autoregressively sample next tokens given a board prompt
4. **Game logic** — validate moves, detect win/draw, enforce rules
5. **UI** — see the UI section below

The training loop pattern from `TinyLanguageBot.cs` is a good reference, but for tic-tac-toe it can be much simpler — no chunked file loading, no BPE training, just a tight loop over generated game sequences.

---

## Namespace

All ported files use `namespace TinyBrainBot`. Either keep this namespace or do a find-and-replace to rename it (e.g., `namespace TicTacLLM`). Renaming is safe — nothing external depends on it.

---

## Dependencies

- **TorchSharp** NuGet package (same version as TinyBrainBot)
- **TorchSharp-cuda-windows** or **libtorch-cpu** depending on whether you're running on GPU
- No other external dependencies for the ported files

---

---

# UI Guide — Building a Similar Interface

TinyBrainBot's UI is a **Windows Forms** app with a dark theme, built entirely in code (no designer files). Here's how to replicate the same structure for a game LLM.

---

## Overall Layout

The window uses nested `Panel` controls with `Dock` properties — no layout managers needed:

```
Form (dark background)
├── Header Panel          (DockStyle.Top, ~50px)  — title, status label, buttons
├── Content Area          (DockStyle.Fill)
│   ├── Left Side Panel   (DockStyle.Left, collapsible) — optional settings
│   ├── Main Panel        (DockStyle.Fill)               — game board + log
│   └── Right Side Panel  (DockStyle.Right, collapsible) — optional stats
└── Input Panel           (DockStyle.Bottom, ~50px) — input box + send/action buttons
```

---

## Dark Theme Colors

These are the exact colors from TinyBrainBot. Copy them directly:

```csharp
// Backgrounds
Color bgDark  = Color.FromArgb(30, 31, 36);   // main background
Color bgLight = Color.FromArgb(40, 42, 48);   // panels, header
Color bgInput = Color.FromArgb(50, 52, 60);   // input boxes, cards

// Text
Color textMain  = Color.FromArgb(230, 232, 236);  // primary text
Color textMuted = Color.FromArgb(100, 106, 118);  // secondary/dim text

// Accents
Color accentGreen  = Color.FromArgb(0, 132, 80);   // user / positive / action
Color accentYellow = Color.FromArgb(180, 140, 40); // warning / pause
Color accentBlue   = Color.FromArgb(100, 180, 255); // info / highlight
Color accentRed    = Color.FromArgb(200, 60, 60);   // error / negative
```

---

## Header Panel

```csharp
var headerPanel = new Panel
{
    Dock = DockStyle.Top,
    Height = 50,
    BackColor = Color.FromArgb(40, 42, 48),
    Padding = new Padding(10, 0, 10, 0)
};

var titleLabel = new Label
{
    Text = "Tic-Tac-Toe LLM",
    Font = new Font("Segoe UI Semibold", 13f),
    ForeColor = Color.FromArgb(230, 232, 236),
    AutoSize = true,
    // Position with titleLabel.Location = new Point(10, 12);
};

var statusLabel = new Label
{
    Text = "Ready",
    Font = new Font("Segoe UI", 9f),
    ForeColor = Color.FromArgb(100, 106, 118),
    AutoSize = true,
    // Position next to title
};
```

Buttons in the header use this pattern (copy for Train, Pause, etc.):
```csharp
var trainButton = new Button
{
    Text = "Start Training",
    Size = new Size(120, 32),
    FlatStyle = FlatStyle.Flat,
    Font = new Font("Segoe UI", 9f, FontStyle.Bold),
    BackColor = Color.FromArgb(60, 140, 80),  // green
    ForeColor = Color.White,
    Cursor = Cursors.Hand
};
trainButton.FlatAppearance.BorderSize = 0;  // removes the border
trainButton.Click += (s, e) => StartTraining();
```

---

## Training Progress Bar

TinyBrainBot draws a custom progress bar directly on the header panel using `OnPaint`. For simplicity, use a standard `ProgressBar` with a custom color:

```csharp
var progressBar = new ProgressBar
{
    Dock = DockStyle.Bottom,
    Height = 4,
    Style = ProgressBarStyle.Continuous,
    Minimum = 0,
    Maximum = totalSteps,
    Value = 0
};
// Update from training thread:
this.Invoke(() => {
    progressBar.Value = currentStep;
    statusLabel.Text = $"Training... step {currentStep}/{totalSteps} — loss {loss:F3}";
});
```

---

## Log / Output Area

For a training log or game move history, use a `RichTextBox`:

```csharp
var logBox = new RichTextBox
{
    Dock = DockStyle.Fill,
    BackColor = Color.FromArgb(30, 31, 36),
    ForeColor = Color.FromArgb(230, 232, 236),
    Font = new Font("Consolas", 9.5f),
    ReadOnly = true,
    BorderStyle = BorderStyle.None,
    ScrollBars = RichTextBoxScrollBars.Vertical
};

// Append colored log lines:
void Log(string text, Color color)
{
    logBox.Invoke(() => {
        logBox.SelectionStart = logBox.TextLength;
        logBox.SelectionLength = 0;
        logBox.SelectionColor = color;
        logBox.AppendText(text + "\n");
        logBox.ScrollToCaret();
    });
}
```

---

## Game Board (Tic-Tac-Toe)

Draw the board on a custom `Panel` using `OnPaint`. This gives full control over appearance:

```csharp
public class BoardPanel : Panel
{
    public char[] Board = new char[9];  // ' ', 'X', or 'O'

    public BoardPanel()
    {
        DoubleBuffered = true;
        BackColor = Color.FromArgb(30, 31, 36);
        Size = new Size(300, 300);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var g = e.Graphics;
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

        int cellSize = Width / 3;
        var gridPen = new Pen(Color.FromArgb(60, 65, 75), 3);
        var xPen = new Pen(Color.FromArgb(100, 180, 255), 4);
        var oBrush = new SolidBrush(Color.FromArgb(0, 180, 100));

        // Draw grid lines
        for (int i = 1; i < 3; i++)
        {
            g.DrawLine(gridPen, i * cellSize, 10, i * cellSize, Height - 10);
            g.DrawLine(gridPen, 10, i * cellSize, Width - 10, i * cellSize);
        }

        // Draw pieces
        for (int i = 0; i < 9; i++)
        {
            int col = i % 3, row = i / 3;
            var rect = new Rectangle(col * cellSize + 20, row * cellSize + 20,
                                     cellSize - 40, cellSize - 40);
            if (Board[i] == 'X')
            {
                g.DrawLine(xPen, rect.Left, rect.Top, rect.Right, rect.Bottom);
                g.DrawLine(xPen, rect.Right, rect.Top, rect.Left, rect.Bottom);
            }
            else if (Board[i] == 'O')
            {
                g.DrawEllipse(new Pen(oBrush.Color, 4), rect);
            }
        }
    }

    // Call board.Invalidate() to redraw after updating Board[]
}
```

---

## Threading — Training on Background Thread

Training must run on a background thread so the UI stays responsive. TinyBrainBot uses `Task.Run` and marshals UI updates back with `Invoke`:

```csharp
private CancellationTokenSource _cts;
private bool isTraining = false;

void StartTraining()
{
    if (isTraining) return;
    isTraining = true;
    trainButton.Enabled = false;
    _cts = new CancellationTokenSource();

    Task.Run(() =>
    {
        try
        {
            // your training loop here
            // update UI like this:
            this.Invoke(() => statusLabel.Text = $"Step {step}...");
        }
        finally
        {
            this.Invoke(() => {
                isTraining = false;
                trainButton.Enabled = true;
                statusLabel.Text = "Training complete";
            });
        }
    }, _cts.Token);
}

void StopTraining()
{
    _cts?.Cancel();
}
```

---

## Collapsible Side Panels

TinyBrainBot has collapsible left/right panels (persona panel, probe panel). The pattern is simple:

```csharp
bool panelOpen = false;
const int PANEL_WIDTH = 220;

void ToggleSidePanel()
{
    panelOpen = !panelOpen;
    sidePanel.Width = panelOpen ? PANEL_WIDTH : 0;
    toggleBtn.Text = panelOpen ? "▶" : "◀";
}
```

---

## Program.cs Entry Point

```csharp
[STAThread]
static void Main()
{
    // Check for GPU
    if (TorchSharp.torch.cuda.is_available())
        TorchSharp.torch.InitializeDeviceType(TorchSharp.DeviceType.CUDA);

    Application.EnableVisualStyles();
    Application.SetCompatibleTextRenderingDefault(false);
    Application.Run(new MainForm());
}
```

---

## Quick Checklist for Claude

When building the UI, tell Claude:
- "Use Windows Forms, no designer files, all code-based layout"
- "Use DockStyle for layout — Top/Bottom/Fill/Left/Right"
- "Dark theme — background `Color.FromArgb(30, 31, 36)`, see color palette in this guide"
- "Training runs on `Task.Run`, UI updates go through `this.Invoke()`"
- "FlatStyle.Flat buttons with `FlatAppearance.BorderSize = 0` for borderless look"
- "DoubleBuffered = true on any custom-drawn panel to prevent flicker"
