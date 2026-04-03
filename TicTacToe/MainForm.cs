using System;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TicTacLLM
{
    public class MainForm : Form
    {
        // ── Dark theme palette ───────────────────────────────────────────────
        static readonly Color BgDark   = Color.FromArgb(30, 31, 36);
        static readonly Color BgLight  = Color.FromArgb(40, 42, 48);
        static readonly Color BgInput  = Color.FromArgb(50, 52, 60);
        static readonly Color TextMain = Color.FromArgb(230, 232, 236);
        static readonly Color TextMute = Color.FromArgb(100, 106, 118);
        static readonly Color Green    = Color.FromArgb(0, 132, 80);
        static readonly Color Yellow   = Color.FromArgb(180, 140, 40);
        static readonly Color Blue     = Color.FromArgb(100, 180, 255);
        static readonly Color Red      = Color.FromArgb(200, 60, 60);

        // ── Controls ─────────────────────────────────────────────────────────
        Label       _statusLabel = null!;
        Button      _trainButton = null!;
        Button      _playXButton = null!;
        Button      _playOButton = null!;
        ProgressBar _progressBar = null!;
        BoardPanel  _board       = null!;
        RichTextBox _log         = null!;

        // ── Settings controls ────────────────────────────────────────────────
        NumericUpDown _sEmbedDim = null!, _sNumHeads = null!, _sFfDim     = null!;
        NumericUpDown _sNumLayers = null!, _sCtxSize = null!;
        NumericUpDown _sSteps = null!, _sWarmup = null!, _sBatch  = null!;
        NumericUpDown _sGames = null!, _sLR     = null!, _sGradClip = null!;
        NumericUpDown _sWdecay = null!, _sLogEvery = null!, _sTemp = null!;

        // ── State ─────────────────────────────────────────────────────────────
        TicTacToeGame?           _game;
        Player?                  _player;
        string                   _history = "|";
        char                     _humanSide;
        bool                     _isTraining;
        CancellationTokenSource? _cts;
        DateTime                 _trainStart;

        public MainForm()
        {
            BuildUI();
            if (File.Exists(Trainer.ModelPath))
            {
                SetStatus("Model found — ready to play.", TextMute);
                _playXButton.Enabled = true;
                _playOButton.Enabled = true;
            }
            else
            {
                SetStatus("Train the model first.", TextMute);
            }
        }

        // ── UI Construction ───────────────────────────────────────────────────

        void BuildUI()
        {
            Text            = "Tic-Tac-Toe LM";
            ClientSize      = new Size(980, 560);
            MinimumSize     = new Size(820, 480);
            BackColor       = BgDark;
            ForeColor       = TextMain;
            Font            = new Font("Segoe UI", 9f);
            FormBorderStyle = FormBorderStyle.Sizable;

            BuildHeader();
            BuildBottomBar();

            // ── Settings panel (docked right) ────────────────────────────────
            var settingsPanel = new Panel
            {
                Dock       = DockStyle.Right,
                Width      = 340,
                BackColor  = BgLight,
                AutoScroll = true,
                Padding    = new Padding(0, 6, 0, 6)
            };
            BuildSettings(settingsPanel);

            // ── Board area (docked left inside content) ───────────────────────
            var boardArea = new Panel
            {
                Dock      = DockStyle.Left,
                Width     = 332,
                BackColor = BgDark
            };
            _board = new BoardPanel { Location = new Point(16, 16), Size = new Size(300, 300) };
            _board.CellClicked += OnCellClicked;
            Array.Fill(_board.Board, '.');
            boardArea.Controls.Add(_board);

            // ── Log (fills remaining space) ───────────────────────────────────
            var logArea = new Panel { Dock = DockStyle.Fill, BackColor = BgDark, Padding = new Padding(8) };
            _log = new RichTextBox
            {
                Dock        = DockStyle.Fill,
                BackColor   = BgInput,
                ForeColor   = TextMain,
                Font        = new Font("Consolas", 9.5f),
                ReadOnly    = true,
                BorderStyle = BorderStyle.None,
                ScrollBars  = RichTextBoxScrollBars.Vertical
            };
            logArea.Controls.Add(_log);

            // ── Content panel ─────────────────────────────────────────────────
            var content = new Panel { Dock = DockStyle.Fill, BackColor = BgDark };
            content.Controls.Add(logArea);    // Fill — must be added before Left
            content.Controls.Add(boardArea);  // Left

            // Add to form: Right before Fill
            Controls.Add(settingsPanel);
            Controls.Add(content);
        }

        void BuildHeader()
        {
            var header = new Panel
            {
                Dock      = DockStyle.Top,
                Height    = 46,
                BackColor = BgLight
            };

            header.Controls.Add(new Label
            {
                Text      = "Tic-Tac-Toe LM",
                Font      = new Font("Segoe UI Semibold", 13f),
                ForeColor = TextMain,
                AutoSize  = true,
                Location  = new Point(12, 10)
            });

            _statusLabel = new Label
            {
                Text         = "",
                Font         = new Font("Segoe UI", 9f),
                ForeColor    = TextMute,
                AutoSize     = false,
                Size         = new Size(600, 18),
                Location     = new Point(180, 14),
                AutoEllipsis = true,
                Anchor       = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right
            };
            header.Controls.Add(_statusLabel);

            _progressBar = new ProgressBar
            {
                Dock    = DockStyle.Bottom,
                Height  = 4,
                Style   = ProgressBarStyle.Continuous,
                Minimum = 0,
                Maximum = 100,
                Value   = 0
            };
            header.Controls.Add(_progressBar);

            Controls.Add(header);
        }

        void BuildBottomBar()
        {
            var bar = new Panel { Dock = DockStyle.Bottom, Height = 54, BackColor = BgLight };

            _trainButton = MakeButton("Train",     Green, 140, 36);
            _playXButton = MakeButton("Play as X", Blue,  120, 36);
            _playOButton = MakeButton("Play as O", Blue,  120, 36);
            _playXButton.Enabled = false;
            _playOButton.Enabled = false;
            _trainButton.Click += (_, __) => OnTrainClicked();
            _playXButton.Click += (_, __) => StartGame('X');
            _playOButton.Click += (_, __) => StartGame('O');

            bar.Controls.AddRange(new Control[] { _trainButton, _playXButton, _playOButton });

            // Center the three buttons horizontally whenever the bar resizes
            void Layout()
            {
                int total = _trainButton.Width + 14 + _playXButton.Width + 14 + _playOButton.Width;
                int x = Math.Max(10, (bar.Width - total) / 2);
                int y = (bar.Height - _trainButton.Height) / 2;
                _trainButton.Location = new Point(x, y);
                _playXButton.Location = new Point(x + _trainButton.Width + 14, y);
                _playOButton.Location = new Point(x + _trainButton.Width + 14 + _playXButton.Width + 14, y);
            }
            bar.SizeChanged += (_, __) => Layout();
            Load += (_, __) => Layout();

            Controls.Add(bar);
        }

        void BuildSettings(Panel p)
        {
            int y = 10;

            void Section(string title)
            {
                var lbl = new Label
                {
                    Text      = title,
                    Location  = new Point(10, y),
                    ForeColor = TextMute,
                    Font      = new Font("Segoe UI", 8f, FontStyle.Bold),
                    AutoSize  = true
                };
                p.Controls.Add(lbl);
                y += 22;
            }

            NumericUpDown Row(string label, decimal min, decimal max, decimal val, decimal inc, int dp = 0)
            {
                p.Controls.Add(new Label
                {
                    Text      = label,
                    Location  = new Point(10, y + 3),
                    ForeColor = TextMain,
                    AutoSize  = true
                });
                var ctrl = new NumericUpDown
                {
                    Location      = new Point(158, y),
                    Size          = new Size(162, 22),
                    Minimum       = min,
                    Maximum       = max,
                    Value         = val,
                    Increment     = inc,
                    DecimalPlaces = dp,
                    BackColor     = BgInput,
                    ForeColor     = TextMain,
                    BorderStyle   = BorderStyle.FixedSingle
                };
                p.Controls.Add(ctrl);
                y += 26;
                return ctrl;
            }

            Section("MODEL  (requires retraining)");
            _sEmbedDim  = Row("Embed Dim",    8,   1024,   64,   8);
            _sNumHeads  = Row("Num Heads",    1,     32,    8,   1);
            _sFfDim     = Row("FF Dim",      16,   4096,  256,  16);
            _sNumLayers = Row("Num Layers",   1,     24,    4,   1);
            _sCtxSize   = Row("Context Size", 8,    256,   32,   8);

            // Live parameter count
            var paramLabel = new Label
            {
                Location  = new Point(10, y),
                Size      = new Size(310, 18),
                ForeColor = Blue,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Bold)
            };
            p.Controls.Add(paramLabel);
            y += 24;

            void UpdateParams()
            {
                int e = (int)_sEmbedDim.Value, f = (int)_sFfDim.Value;
                int l = (int)_sNumLayers.Value, c = (int)_sCtxSize.Value;
                const int v = 14; // fixed vocab size
                long perBlock = 4L * e * e + 2L * e * f + 6L * e + f;
                long total    = perBlock * l
                              + (long)v * e   // token embed
                              + (long)c * e   // pos embed
                              + (long)v * e + v  // lm head
                              + 2L * e;          // final LN
                string fmt = total >= 1_000_000
                    ? $"{total / 1_000_000.0:F2}M"
                    : $"{total / 1_000.0:F1}K";
                paramLabel.Text = $"≈ {fmt} parameters";
            }
            _sEmbedDim.ValueChanged  += (_, __) => UpdateParams();
            _sFfDim.ValueChanged     += (_, __) => UpdateParams();
            _sNumLayers.ValueChanged += (_, __) => UpdateParams();
            _sCtxSize.ValueChanged   += (_, __) => UpdateParams();
            UpdateParams();

            y += 6;
            Section("TRAINING");
            _sSteps    = Row("Steps",               100, 200_000,   8_000,   500);
            _sGames    = Row("Training Games",     1_000, 500_000,  60_000, 5_000);
            _sBatch    = Row("Batch Size",             1,   1_024,      64,     8);
            _sWarmup   = Row("Warmup Steps",           0,  10_000,     400,   100);
            _sLR       = Row("Learning Rate",  0.00001m, 0.01m, 0.0003m, 0.00005m, 5);
            _sGradClip = Row("Grad Clip",         0.1m,   10m,    1.0m,    0.1m, 2);
            _sWdecay   = Row("Weight Decay",         0m,  0.5m,   0.01m,  0.001m, 3);
            _sLogEvery = Row("Log Every N Steps",     1,  5_000,     200,     50);

            y += 6;
            Section("INFERENCE");
            _sTemp = Row("Temperature", 0.01m, 2m, 0.3m, 0.05m, 2);
        }

        Button MakeButton(string text, Color bg, int width, int height = 32)
        {
            var btn = new Button
            {
                Text      = text,
                Size      = new Size(width, height),
                FlatStyle = FlatStyle.Flat,
                Font      = new Font("Segoe UI", 9f, FontStyle.Bold),
                BackColor = bg,
                ForeColor = Color.White,
                Cursor    = Cursors.Hand
            };
            btn.FlatAppearance.BorderSize = 0;
            return btn;
        }

        // ── Training ──────────────────────────────────────────────────────────

        void OnTrainClicked()
        {
            if (_isTraining) { _cts?.Cancel(); return; }

            if ((int)_sEmbedDim.Value % (int)_sNumHeads.Value != 0)
            {
                MessageBox.Show(
                    $"Embed Dim ({_sEmbedDim.Value}) must be divisible by Num Heads ({_sNumHeads.Value}).",
                    "Invalid settings", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _isTraining            = true;
            _trainButton.Text      = "Stop";
            _trainButton.BackColor = Yellow;
            _playXButton.Enabled   = false;
            _playOButton.Enabled   = false;
            _progressBar.Value     = 0;
            _trainStart            = DateTime.Now;
            _log.Clear();

            SetStatus("Generating training data...", Yellow);

            _cts = new CancellationTokenSource();

            var trainer = new Trainer
            {
                Arch = new ArchConfig
                {
                    EmbedDim    = (int)_sEmbedDim.Value,
                    NumHeads    = (int)_sNumHeads.Value,
                    FfDim       = (int)_sFfDim.Value,
                    NumLayers   = (int)_sNumLayers.Value,
                    ContextSize = (int)_sCtxSize.Value
                },
                NumTrainingGames = (int)_sGames.Value,
                TotalSteps       = (int)_sSteps.Value,
                WarmupSteps      = (int)_sWarmup.Value,
                LearningRate     = (float)_sLR.Value,
                BatchSize        = (int)_sBatch.Value,
                GradientClip     = (float)_sGradClip.Value,
                WeightDecay      = (float)_sWdecay.Value,
                LogEvery         = (int)_sLogEvery.Value,

                OnLog = (msg, color) => AppendLog(msg, color),

                OnProgress = (step, total, loss) =>
                {
                    int    pct     = (int)(100.0 * step / total);
                    double elapsed = (DateTime.Now - _trainStart).TotalSeconds;
                    double rate    = step / Math.Max(elapsed, 0.001);
                    var    eta     = TimeSpan.FromSeconds((total - step) / Math.Max(rate, 0.001));
                    string etaStr  = eta.TotalSeconds < 60
                        ? $"{(int)eta.TotalSeconds}s"
                        : $"{(int)eta.TotalMinutes}m {eta.Seconds:D2}s";

                    UIInvoke(() =>
                    {
                        _progressBar.Value = Math.Min(pct, 100);
                        SetStatus($"Training  {pct}%  —  step {step}/{total}  loss {loss:F4}  ETA {etaStr}", Yellow);
                    });
                },

                OnComplete = () => UIInvoke(() =>
                {
                    _isTraining            = false;
                    _trainButton.Text      = "Train Again";
                    _trainButton.BackColor = Green;
                    _playXButton.Enabled   = true;
                    _playOButton.Enabled   = true;
                    _progressBar.Value     = 100;
                    _player                = null;

                    double secs = (DateTime.Now - _trainStart).TotalSeconds;
                    string t = secs < 60 ? $"{(int)secs}s" : $"{(int)(secs/60)}m {(int)(secs%60):D2}s";
                    SetStatus($"Done in {t} — ready to play!", Color.FromArgb(0, 180, 100));
                })
            };

            Task.Run(() => trainer.Run(_cts.Token));
        }

        // ── Game flow ─────────────────────────────────────────────────────────

        void StartGame(char humanSide)
        {
            if (_player == null)
            {
                try { _player = new Player { Temperature = (float)_sTemp.Value }; }
                catch (FileNotFoundException)
                {
                    MessageBox.Show("No trained model found.\nClick 'Train' first.",
                        "Model not found", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
            }
            else
            {
                _player.Temperature = (float)_sTemp.Value;
            }

            _humanSide       = humanSide;
            _game            = new TicTacToeGame();
            _history         = "|";
            _board.Board     = _game.Board;
            _board.WinLine   = null;
            _board.Clickable = false;
            _board.Invalidate();

            AppendLog($"─── New game — you play {humanSide} ───", TextMute);
            SetStatus($"Your turn ({humanSide}).", Blue);

            if (humanSide == 'O') DoAIMove();
            else _board.Clickable = true;
        }

        void OnCellClicked(object? sender, int pos)
        {
            if (_game == null || !_game.IsValidMove(pos)) return;

            _board.Clickable = false;
            _game.MakeMove(pos);
            _history        += (char)('0' + pos);
            _board.Board     = _game.Board;
            _board.Invalidate();
            AppendLog($"You play {pos}.", TextMain);

            if (!CheckGameOver()) DoAIMove();
        }

        void DoAIMove()
        {
            if (_game == null || _player == null) return;
            SetStatus("AI is thinking...", TextMute);

            Task.Run(() =>
            {
                int move = _player.GetMove(_game, _history);
                UIInvoke(() =>
                {
                    if (_game == null) return;
                    _game.MakeMove(move);
                    _history       += (char)('0' + move);
                    _board.Board    = _game.Board;
                    _board.Invalidate();
                    AppendLog($"AI plays {move}.", Blue);

                    if (!CheckGameOver())
                    {
                        SetStatus($"Your turn ({_humanSide}).", Blue);
                        _board.Clickable = true;
                    }
                });
            });
        }

        bool CheckGameOver()
        {
            if (_game == null || !_game.IsOver()) return false;

            _board.Clickable = false;
            _board.WinLine   = _game.GetWinLine();
            _board.Invalidate();

            var winner = _game.CheckWinner();
            if      (winner == _humanSide) { SetStatus("You win!",  Green);  AppendLog("You win!",  Green);  }
            else if (winner != null)        { SetStatus("AI wins!",  Red);    AppendLog("AI wins!",  Red);    }
            else                            { SetStatus("Draw!",     Yellow); AppendLog("Draw!",     Yellow); }

            AppendLog("Click Play as X / Play as O to play again.", TextMute);
            return true;
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        void SetStatus(string text, Color color)
        {
            _statusLabel.Text      = text;
            _statusLabel.ForeColor = color;
        }

        void AppendLog(string text, Color color)
        {
            UIInvoke(() =>
            {
                _log.SelectionStart  = _log.TextLength;
                _log.SelectionLength = 0;
                _log.SelectionColor  = color;
                _log.AppendText(text + "\n");
                _log.ScrollToCaret();
            });
        }

        void UIInvoke(Action action)
        {
            if (InvokeRequired) Invoke(action);
            else action();
        }
    }
}
