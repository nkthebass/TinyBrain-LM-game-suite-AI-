using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ChessLLM
{
    public class MainForm : Form
    {
        static readonly Color BgDark  = Color.FromArgb(30, 31, 36);
        static readonly Color BgLight = Color.FromArgb(40, 42, 48);
        static readonly Color BgInput = Color.FromArgb(50, 52, 60);
        static readonly Color TextMain = Color.FromArgb(230, 232, 236);
        static readonly Color TextMute = Color.FromArgb(100, 106, 118);
        static readonly Color Green   = Color.FromArgb(0, 132, 80);
        static readonly Color Yellow  = Color.FromArgb(180, 140, 40);
        static readonly Color Blue    = Color.FromArgb(100, 180, 255);
        static readonly Color Red     = Color.FromArgb(200, 60, 60);

        // Controls
        Label          _statusLabel = null!;
        Button         _dataBtn = null!, _trainButton = null!, _rlButton = null!, _bpButton = null!, _battleBtn = null!, _playWButton = null!, _playBButton = null!, _drawBtn = null!;
        ProgressBar    _progressBar = null!;
        ChessBoardPanel _board      = null!;
        RichTextBox    _log         = null!;

        // Settings
        NumericUpDown _sEmbedDim = null!, _sNumHeads = null!, _sFfDim = null!;
        NumericUpDown _sNumLayers = null!, _sCtxSize = null!;
        NumericUpDown _sSteps = null!, _sWarmup = null!, _sBatch = null!;
        NumericUpDown _sGames = null!, _sAccum = null!, _sLR = null!, _sGradClip = null!;
        NumericUpDown _sWdecay = null!, _sLogEvery = null!, _sTemp = null!;

        // RL settings
        NumericUpDown _sSfDepth = null!, _sSfParallel = null!, _sSelfPlayGames = null!;
        NumericUpDown _sStartMove = null!, _sCpThreshold = null!, _sRlSteps = null!, _sRlLR = null!;
        NumericUpDown _sBpGames = null!, _sBpEveryN = null!;
        NumericUpDown _sCandidates = null!, _sResponses = null!;
        CheckBox      _thinkCheck = null!;

        // Pre-game piece dragging
        int _dragFrom = -1; // square being dragged from, -1 = none

    // Game state
        ChessGame? _game;
        ThinkingPlayer? _player;
        string     _history = "|";
        bool       _humanIsWhite;
        int        _selectedSq = -1;
        List<string> _legalFromSelected = new();

        // Training state
        bool _isTraining;
        CancellationTokenSource? _cts;
        DateTime _trainStart;

        public MainForm()
        {
            BuildUI();
            LoadSettings();
            UpdateTrainButton();

            // Board is always editable before a game starts — move pieces freely
            _game = new ChessGame();
            _board.Board = _game.Board;
            _board.Clickable = true;
            _board.SquareClicked += OnPreGameSquareClicked;
            _board.SquareRightClicked += OnPreGameRightClick;
            _board.Invalidate();
        }

        void UpdateTrainButton()
        {
            var (canResume, step, totalSteps, completed) = Trainer.CheckState();

            if (canResume && completed)
            {
                _trainButton.Text = "Continue";
                SetStatus($"Trained model ready ({step:N0} steps). Play or continue training.", TextMute);
                _playWButton.Enabled = true;
                _playBButton.Enabled = true;
            }
            else if (canResume && !completed)
            {
                _trainButton.Text = "Resume";
                SetStatus($"Interrupted at step {step:N0}/{totalSteps:N0} — click Resume.", Color.FromArgb(180, 140, 40));
                _playWButton.Enabled = File.Exists(Trainer.ModelPath);
                _playBButton.Enabled = File.Exists(Trainer.ModelPath);
            }
            else if (File.Exists(Trainer.ModelPath))
            {
                SetStatus("Model found — ready to play.", TextMute);
                _playWButton.Enabled = true;
                _playBButton.Enabled = true;
            }
            else
            {
                _trainButton.Text = "Train";
                SetStatus("Train the model first.", TextMute);
            }
        }

        // ── UI ───────────────────────────────────────────────────────────────

        void BuildUI()
        {
            Text        = "Chess LM";
            ClientSize  = new Size(1200, 700);
            MinimumSize = new Size(900, 550);
            BackColor   = BgDark;
            ForeColor   = TextMain;
            Font        = new Font("Segoe UI", 9f);
            FormBorderStyle = FormBorderStyle.Sizable;

            // Settings panel — right side
            var settingsPanel = new Panel
            {
                Dock = DockStyle.Right, Width = 340, BackColor = BgLight, AutoScroll = true
            };
            BuildSettings(settingsPanel);

            // Log — right of the board, left of settings
            var logArea = new Panel { Dock = DockStyle.Fill, BackColor = BgDark, Padding = new Padding(8) };
            _log = new RichTextBox
            {
                Dock = DockStyle.Fill, BackColor = BgInput, ForeColor = TextMain,
                Font = new Font("Consolas", 9.5f), ReadOnly = true,
                BorderStyle = BorderStyle.None, ScrollBars = RichTextBoxScrollBars.Vertical,
                WordWrap = true
            };
            logArea.Controls.Add(_log);

            // Board — fixed width on left
            var boardArea = new Panel { Dock = DockStyle.Left, Width = 520, BackColor = BgDark };
            _board = new ChessBoardPanel
            {
                Dock = DockStyle.Fill
            };
            _board.SquareClicked += OnSquareClicked;
            _board.Board = new ChessGame().Board;
            boardArea.Padding = new Padding(8);
            boardArea.Controls.Add(_board);

            // Content = board (left) + log (fill the rest)
            var content = new Panel { Dock = DockStyle.Fill, BackColor = BgDark };
            content.Controls.Add(logArea);
            content.Controls.Add(boardArea);

            // WinForms dock order: LAST added gets priority.
            // Add Fill first, then Right/Top/Bottom so they carve out their space.
            Controls.Add(content);
            Controls.Add(settingsPanel);
            BuildHeader();
            BuildBottomBar();
        }

        void BuildHeader()
        {
            var header = new Panel { Dock = DockStyle.Top, Height = 46, BackColor = BgLight };
            header.Controls.Add(new Label
            {
                Text = "Chess LM", Font = new Font("Segoe UI Semibold", 13f),
                ForeColor = TextMain, AutoSize = true, Location = new Point(12, 10)
            });
            _statusLabel = new Label
            {
                Location = new Point(130, 14), Size = new Size(500, 18),
                ForeColor = TextMute, AutoEllipsis = true
            };
            header.Controls.Add(_statusLabel);
            _progressBar = new ProgressBar
            {
                Dock = DockStyle.Bottom, Height = 4,
                Style = ProgressBarStyle.Continuous, Minimum = 0, Maximum = 100
            };
            header.Controls.Add(_progressBar);
            Controls.Add(header);
        }

        void BuildBottomBar()
        {
            var bar = new Panel { Dock = DockStyle.Bottom, Height = 54, BackColor = BgLight };

            _dataBtn     = MkBtn("Datasets", Color.FromArgb(90, 80, 140), 90, 36);
            _battleBtn   = MkBtn("AI vs AI", Color.FromArgb(140, 80, 140), 90, 36);
            _playWButton = MkBtn("Play White", Blue, 110, 36);
            _playBButton = MkBtn("Play Black", Blue, 110, 36);
            _drawBtn     = MkBtn("Draw", Color.FromArgb(120, 110, 50), 65, 36);
            _playWButton.Enabled = false;
            _playBButton.Enabled = false;
            _drawBtn.Visible = false;
            _dataBtn.Click     += (_, __) => { var f = new DatasetForm(); f.ShowDialog(this); };
            _battleBtn.Click   += (_, __) => { var f = new AIBattleForm(); f.Show(this); };
            _playWButton.Click += (_, __) => StartGame(true);
            _playBButton.Click += (_, __) => StartGame(false);
            _drawBtn.Click     += (_, __) => ClaimDraw();

            bar.Controls.AddRange(new Control[] { _dataBtn, _battleBtn, _playWButton, _playBButton, _drawBtn });

            void Layout()
            {
                int total = 90 + 12 + 90 + 12 + 110 + 12 + 110 + 12 + 65;
                int x = Math.Max(10, (bar.Width - total) / 2);
                int y = (bar.Height - 36) / 2;
                _dataBtn.Location     = new Point(x, y);
                _battleBtn.Location   = new Point(x + 102, y);
                _playWButton.Location = new Point(x + 204, y);
                _playBButton.Location = new Point(x + 326, y);
                _drawBtn.Location     = new Point(x + 448, y);
            }
            bar.SizeChanged += (_, __) => Layout();
            Load += (_, __) => Layout();
            Controls.Add(bar);
        }

        void BuildSettings(Panel p)
        {
            int y = 10;
            void Section(string t) { p.Controls.Add(new Label { Text = t, Location = new Point(10, y), ForeColor = TextMute, Font = new Font("Segoe UI", 8f, FontStyle.Bold), AutoSize = true }); y += 22; }

            NumericUpDown Row(string label, decimal min, decimal max, decimal val, decimal inc, int dp = 0)
            {
                p.Controls.Add(new Label { Text = label, Location = new Point(10, y + 3), ForeColor = TextMain, AutoSize = true });
                var c = new NumericUpDown
                {
                    Location = new Point(158, y), Size = new Size(162, 22),
                    Minimum = min, Maximum = max, Value = val, Increment = inc,
                    DecimalPlaces = dp, BackColor = BgInput, ForeColor = TextMain,
                    BorderStyle = BorderStyle.FixedSingle
                };
                p.Controls.Add(c); y += 26; return c;
            }

            Section("MODEL  (requires retraining)");
            _sEmbedDim  = Row("Embed Dim",    8,   1024, 128,   8);
            _sNumHeads  = Row("Num Heads",    1,     32,   4,   1);
            _sFfDim     = Row("FF Dim",      16,   8192, 512,  16);
            _sNumLayers = Row("Num Layers",   1,     32,   6,   1);
            _sCtxSize   = Row("Context Size", 32,  2048, 512,  32);

            var paramLabel = new Label { Location = new Point(10, y), Size = new Size(310, 18), ForeColor = Blue, Font = new Font("Segoe UI", 8.5f, FontStyle.Bold) };
            p.Controls.Add(paramLabel); y += 24;

            void UpdateParams()
            {
                int e = (int)_sEmbedDim.Value, f2 = (int)_sFfDim.Value;
                int l = (int)_sNumLayers.Value, c = (int)_sCtxSize.Value;
                const int v = 26;
                long perBlock = 4L * e * e + 2L * e * f2 + 6L * e + f2;
                long total = perBlock * l + (long)v * e + (long)c * e + (long)v * e + v + 2L * e;
                string fmt = total >= 1_000_000 ? $"{total / 1_000_000.0:F2}M" : $"{total / 1_000.0:F1}K";
                paramLabel.Text = $"≈ {fmt} parameters";
            }
            _sEmbedDim.ValueChanged  += (_, __) => UpdateParams();
            _sFfDim.ValueChanged     += (_, __) => UpdateParams();
            _sNumLayers.ValueChanged += (_, __) => UpdateParams();
            _sCtxSize.ValueChanged   += (_, __) => UpdateParams();
            UpdateParams();

            y += 6;
            Section("TRAINING");
            _trainButton = MkBtn("Start Training", Green, 280, 30);
            _trainButton.Location = new Point(10, y);
            _trainButton.Click += (_, __) => OnTrainClicked();
            p.Controls.Add(_trainButton);
            y += 36;
            _sSteps    = Row("Steps",            1_000, 1_000_000, 20_000,  1_000);
            _sGames    = Row("Training Games",     100,   200_000,  50_000,   5_000);
            _sBatch    = Row("Batch Size",           1,       512,     32,      4);
            _sAccum    = Row("Grad Accum Steps",     1,        16,      2,      1);
            _sWarmup   = Row("Warmup Steps",         0,    50_000,  1_000,    200);
            _sLR       = Row("Learning Rate", 0.00001m, 0.01m, 0.0003m, 0.00005m, 5);
            _sGradClip = Row("Grad Clip",       0.1m,  10m, 1.0m, 0.1m, 2);
            _sWdecay   = Row("Weight Decay",      0m, 0.5m, 0.01m, 0.001m, 3);
            _sLogEvery = Row("Log Every",          1, 10_000, 200, 50);

            y += 6;
            Section("INFERENCE");
            _sTemp = Row("Temperature", 0.01m, 2m, 0.5m, 0.05m, 2);

            // Thinking (search) settings
            _thinkCheck = new CheckBox
            {
                Text = "Enable Thinking (lookahead search)",
                Location = new Point(10, y), Size = new Size(270, 20),
                ForeColor = TextMain, Checked = false
            };
            p.Controls.Add(_thinkCheck);
            y += 24;
            _sCandidates = Row("Candidates", 1, 10, 5, 1);
            _sResponses  = Row("Responses",  1, 20, 10, 1);

            y += 6;
            Section("RL TRAINING  (Stockfish)");
            _rlButton = MkBtn("Run RL Training", Color.FromArgb(180, 100, 40), 280, 30);
            _rlButton.Location = new Point(10, y);
            _rlButton.Click += (_, __) => OnRLClicked();
            p.Controls.Add(_rlButton);
            y += 36;
            _sSfDepth      = Row("SF Depth",        5,     25,   10,   1);
            _sSfParallel   = Row("SF Parallel",     1,     16,    4,   1);
            _sSelfPlayGames= Row("Self-Play Games", 10,  5000,  200,  50);
            _sStartMove    = Row("Start at Move",    0,   80,     0,   5);
            _sCpThreshold  = Row("CP Loss Thresh",  10,   500,   50,  10);
            _sRlSteps      = Row("RL Fine-Tune Steps", 500, 50_000, 5000, 500);
            _sRlLR         = Row("RL Learning Rate", 0.00001m, 0.001m, 0.0001m, 0.00001m, 5);

            y += 6;
            Section("BOARD PREFIX TRAINING");
            _bpButton = MkBtn("Run Board Training", Color.FromArgb(60, 140, 160), 280, 30);
            _bpButton.Location = new Point(10, y);
            _bpButton.Click += (_, __) => OnBPClicked();
            p.Controls.Add(_bpButton);
            y += 36;
            _sBpGames    = Row("BP Games",       100, 50_000, 5_000, 500);
            _sBpEveryN   = Row("Prefix Every N",   1,     20,     5,   1);

            // Auto-save settings whenever any value changes
            foreach (var ctrl in new NumericUpDown[] {
                _sEmbedDim, _sNumHeads, _sFfDim, _sNumLayers, _sCtxSize,
                _sSteps, _sGames, _sBatch, _sAccum, _sWarmup,
                _sLR, _sGradClip, _sWdecay, _sLogEvery, _sTemp,
                _sSfDepth, _sSfParallel, _sSelfPlayGames, _sStartMove, _sCpThreshold, _sRlSteps, _sRlLR,
                _sBpGames, _sBpEveryN, _sCandidates, _sResponses })
            {
                ctrl.ValueChanged += (_, __) => SaveSettings();
            }
            _thinkCheck.CheckedChanged += (_, __) => SaveSettings();
        }

        void LoadSettings()
        {
            var s = UISettings.Load();
            _sEmbedDim.Value  = Math.Clamp(s.EmbedDim, _sEmbedDim.Minimum, _sEmbedDim.Maximum);
            _sNumHeads.Value  = Math.Clamp(s.NumHeads, _sNumHeads.Minimum, _sNumHeads.Maximum);
            _sFfDim.Value     = Math.Clamp(s.FfDim, _sFfDim.Minimum, _sFfDim.Maximum);
            _sNumLayers.Value = Math.Clamp(s.NumLayers, _sNumLayers.Minimum, _sNumLayers.Maximum);
            _sCtxSize.Value   = Math.Clamp(s.CtxSize, _sCtxSize.Minimum, _sCtxSize.Maximum);
            _sSteps.Value     = Math.Clamp(s.Steps, _sSteps.Minimum, _sSteps.Maximum);
            _sGames.Value     = Math.Clamp(s.Games, _sGames.Minimum, _sGames.Maximum);
            _sBatch.Value     = Math.Clamp(s.Batch, _sBatch.Minimum, _sBatch.Maximum);
            _sAccum.Value     = Math.Clamp(s.Accum, _sAccum.Minimum, _sAccum.Maximum);
            _sWarmup.Value    = Math.Clamp(s.Warmup, _sWarmup.Minimum, _sWarmup.Maximum);
            _sLR.Value        = Math.Clamp(s.LR, _sLR.Minimum, _sLR.Maximum);
            _sGradClip.Value  = Math.Clamp(s.GradClip, _sGradClip.Minimum, _sGradClip.Maximum);
            _sWdecay.Value    = Math.Clamp(s.WDecay, _sWdecay.Minimum, _sWdecay.Maximum);
            _sLogEvery.Value  = Math.Clamp(s.LogEvery, _sLogEvery.Minimum, _sLogEvery.Maximum);
            _sTemp.Value      = Math.Clamp(s.Temp, _sTemp.Minimum, _sTemp.Maximum);
            _sStartMove.Value     = Math.Clamp(s.StartMove, _sStartMove.Minimum, _sStartMove.Maximum);
            _sSfDepth.Value       = Math.Clamp(s.SfDepth, _sSfDepth.Minimum, _sSfDepth.Maximum);
            _sSfParallel.Value    = Math.Clamp(s.SfParallel, _sSfParallel.Minimum, _sSfParallel.Maximum);
            _sSelfPlayGames.Value = Math.Clamp(s.SelfPlayGames, _sSelfPlayGames.Minimum, _sSelfPlayGames.Maximum);
            _sCpThreshold.Value   = Math.Clamp(s.CpThreshold, _sCpThreshold.Minimum, _sCpThreshold.Maximum);
            _sRlSteps.Value       = Math.Clamp(s.RlSteps, _sRlSteps.Minimum, _sRlSteps.Maximum);
            _sRlLR.Value          = Math.Clamp(s.RlLR, _sRlLR.Minimum, _sRlLR.Maximum);
            _sBpGames.Value       = Math.Clamp(s.BpGames, _sBpGames.Minimum, _sBpGames.Maximum);
            _sBpEveryN.Value      = Math.Clamp(s.BpEveryN, _sBpEveryN.Minimum, _sBpEveryN.Maximum);
            _thinkCheck.Checked   = s.ThinkEnabled;
            _sCandidates.Value    = Math.Clamp(s.Candidates, _sCandidates.Minimum, _sCandidates.Maximum);
            _sResponses.Value     = Math.Clamp(s.Responses, _sResponses.Minimum, _sResponses.Maximum);
        }

        void SaveSettings()
        {
            new UISettings
            {
                EmbedDim = (int)_sEmbedDim.Value, NumHeads = (int)_sNumHeads.Value,
                FfDim = (int)_sFfDim.Value, NumLayers = (int)_sNumLayers.Value,
                CtxSize = (int)_sCtxSize.Value,
                Steps = (int)_sSteps.Value, Games = (int)_sGames.Value,
                Batch = (int)_sBatch.Value, Accum = (int)_sAccum.Value,
                Warmup = (int)_sWarmup.Value, LR = _sLR.Value,
                GradClip = _sGradClip.Value, WDecay = _sWdecay.Value,
                LogEvery = (int)_sLogEvery.Value, Temp = _sTemp.Value,
                StartMove = (int)_sStartMove.Value,
                SfDepth = (int)_sSfDepth.Value, SfParallel = (int)_sSfParallel.Value,
                SelfPlayGames = (int)_sSelfPlayGames.Value, CpThreshold = (int)_sCpThreshold.Value,
                RlSteps = (int)_sRlSteps.Value, RlLR = _sRlLR.Value,
                BpGames = (int)_sBpGames.Value, BpEveryN = (int)_sBpEveryN.Value,
                ThinkEnabled = _thinkCheck.Checked,
                Candidates = (int)_sCandidates.Value, Responses = (int)_sResponses.Value
            }.Save();
        }

        Button MkBtn(string text, Color bg, int w, int h)
        {
            var b = new Button
            {
                Text = text, Size = new Size(w, h), FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                BackColor = bg, ForeColor = Color.White, Cursor = Cursors.Hand
            };
            b.FlatAppearance.BorderSize = 0;
            return b;
        }

        // ── Training ──────────────────────────────────────────────────────────

        void OnTrainClicked()
        {
            if (_isTraining) { _cts?.Cancel(); return; }

            if ((int)_sEmbedDim.Value % (int)_sNumHeads.Value != 0)
            {
                MessageBox.Show($"Embed Dim must be divisible by Num Heads.",
                    "Invalid", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _isTraining = true;
            _trainButton.Text = "Stop"; _trainButton.BackColor = Yellow;
            _playWButton.Enabled = false; _playBButton.Enabled = false;
            _progressBar.Value = 0; _log.Clear(); _trainStart = DateTime.Now;
            SetStatus("Generating training data...", Yellow);

            _cts = new CancellationTokenSource();

            var trainer = new Trainer
            {
                Arch = new ArchConfig
                {
                    EmbedDim = (int)_sEmbedDim.Value, NumHeads = (int)_sNumHeads.Value,
                    FfDim = (int)_sFfDim.Value, NumLayers = (int)_sNumLayers.Value,
                    ContextSize = (int)_sCtxSize.Value
                },
                NumTrainingGames = (int)_sGames.Value, TotalSteps = (int)_sSteps.Value,
                WarmupSteps = (int)_sWarmup.Value, LearningRate = (float)_sLR.Value,
                BatchSize = (int)_sBatch.Value, GradientClip = (float)_sGradClip.Value,
                WeightDecay = (float)_sWdecay.Value, LogEvery = (int)_sLogEvery.Value,

                OnLog = (msg, color) => AppendLog(msg, color),
                OnProgress = (step, total, loss) =>
                {
                    int pct = (int)(100.0 * step / total);
                    double elapsed = (DateTime.Now - _trainStart).TotalSeconds;
                    double rate = step / Math.Max(elapsed, 0.001);
                    var eta = TimeSpan.FromSeconds((total - step) / Math.Max(rate, 0.001));
                    string etaStr = eta.TotalSeconds < 60 ? $"{(int)eta.TotalSeconds}s" : $"{(int)eta.TotalMinutes}m {eta.Seconds:D2}s";
                    UIInvoke(() => { _progressBar.Value = Math.Min(pct, 100); SetStatus($"Training  {pct}%  step {step}/{total}  loss {loss:F4}  ETA {etaStr}", Yellow); });
                },
                OnComplete = () => UIInvoke(() =>
                {
                    _isTraining = false;
                    _trainButton.BackColor = Green;
                    _progressBar.Value = 100;
                    _player?.Dispose(); _player = null;
                    double s = (DateTime.Now - _trainStart).TotalSeconds;
                    string t = s < 60 ? $"{(int)s}s" : $"{(int)(s / 60)}m {(int)(s % 60):D2}s";
                    SetStatus($"Done in {t} — ready to play!", Color.FromArgb(0, 180, 100));
                    UpdateTrainButton();
                })
            };

            Task.Run(() => trainer.Run(_cts.Token));
        }

        // ── RL Training ───────────────────────────────────────────────────────

        void OnRLClicked()
        {
            if (_isTraining) { _cts?.Cancel(); return; }

            // Check Stockfish exists
            string sfPath = FindStockfish();
            if (sfPath == "")
            {
                MessageBox.Show(
                    "Stockfish not found.\n\n" +
                    "Download from https://stockfishchess.org/download/\n" +
                    "Place stockfish.exe in the Chess folder or add it to PATH.",
                    "Stockfish Required", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            if (!File.Exists(Trainer.ModelPath))
            {
                MessageBox.Show("Train a supervised model first (click Train), then use RL to improve it.",
                    "No Model", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _isTraining = true;
            _rlButton.Text = "Stop"; _rlButton.BackColor = Yellow;
            _trainButton.Enabled = false; _playWButton.Enabled = false; _playBButton.Enabled = false;
            _progressBar.Value = 0;
            _trainStart = DateTime.Now;
            SetStatus("Starting RL training with Stockfish...", Color.FromArgb(180, 100, 40));

            _cts = new CancellationTokenSource();

            var rl = new RLTrainer
            {
                StockfishPath = sfPath,
                SfDepth = (int)_sSfDepth.Value,
                SfParallel = (int)_sSfParallel.Value,
                NumSelfPlayGames = (int)_sSelfPlayGames.Value,
                StartAtMove = (int)_sStartMove.Value,
                CpLossThreshold = (int)_sCpThreshold.Value,
                TotalSteps = (int)_sRlSteps.Value,
                LearningRate = (float)_sRlLR.Value,
                BatchSize = (int)_sBatch.Value,

                OnLog = (msg, color) => AppendLog(msg, color),
                OnProgress = (step, total, loss) =>
                {
                    int pct = (int)(100.0 * step / total);
                    this.Invoke(() =>
                    {
                        _progressBar.Value = Math.Min(pct, 100);
                        SetStatus($"RL Training {pct}%  step {step}/{total}  loss {loss:F4}", Color.FromArgb(180, 100, 40));
                    });
                },
                OnComplete = () => this.Invoke(() =>
                {
                    _isTraining = false;
                    _rlButton.Text = "RL Train"; _rlButton.BackColor = Color.FromArgb(180, 100, 40);
                    _trainButton.Enabled = true;
                    _progressBar.Value = 100;
                    _player?.Dispose(); _player = null;
                    double s = (DateTime.Now - _trainStart).TotalSeconds;
                    string t = s < 60 ? $"{(int)s}s" : $"{(int)(s / 60)}m {(int)(s % 60):D2}s";
                    SetStatus($"RL done in {t}! Model improved with Stockfish corrections.", Color.FromArgb(0, 180, 100));
                    UpdateTrainButton();
                })
            };

            Task.Run(async () =>
            {
                // Phase 1: Generate self-play games (skip if data already exists)
                if (!File.Exists(RLTrainer.RLDataFile))
                {
                    rl.GenerateCorrectedData(_cts.Token);
                    if (_cts.Token.IsCancellationRequested) return;
                }

                // Phase 2: Fine-tune on corrected data
                rl.RunFineTune(_cts.Token);
            });
        }

        // ── Board Prefix Training ─────────────────────────────────────────────

        void OnBPClicked()
        {
            if (_isTraining) { _cts?.Cancel(); return; }

            if (!File.Exists(Trainer.ModelPath))
            {
                MessageBox.Show("Train a base model first.", "No Model", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _isTraining = true;
            _bpButton.Text = "Stop"; _bpButton.BackColor = Yellow;
            _trainButton.Enabled = false; _rlButton.Enabled = false;
            _playWButton.Enabled = false; _playBButton.Enabled = false;
            _progressBar.Value = 0;
            _trainStart = DateTime.Now;
            SetStatus("Board prefix training...", Color.FromArgb(60, 140, 160));

            _cts = new CancellationTokenSource();

            var bp = new BoardPrefixTrainer
            {
                NumGames = (int)_sBpGames.Value,
                PrefixEveryN = (int)_sBpEveryN.Value,
                TotalSteps = (int)_sRlSteps.Value,
                LearningRate = (float)_sRlLR.Value,
                BatchSize = (int)_sBatch.Value,

                OnLog = (msg, color) => AppendLog(msg, color),
                OnProgress = (step, total, loss) =>
                {
                    int pct = (int)(100.0 * step / total);
                    this.Invoke(() =>
                    {
                        _progressBar.Value = Math.Min(pct, 100);
                        SetStatus($"BP Training {pct}%  step {step}/{total}  loss {loss:F4}",
                            Color.FromArgb(60, 140, 160));
                    });
                },
                OnComplete = () => this.Invoke(() =>
                {
                    _isTraining = false;
                    _bpButton.Text = "Board"; _bpButton.BackColor = Color.FromArgb(60, 140, 160);
                    _trainButton.Enabled = true; _rlButton.Enabled = true;
                    _progressBar.Value = 100;
                    _player?.Dispose(); _player = null;
                    double s = (DateTime.Now - _trainStart).TotalSeconds;
                    string t = s < 60 ? $"{(int)s}s" : $"{(int)(s / 60)}m {(int)(s % 60):D2}s";
                    SetStatus($"Board prefix training done in {t}!", Color.FromArgb(0, 180, 100));
                    UpdateTrainButton();
                })
            };

            Task.Run(() =>
            {
                if (!File.Exists(BoardPrefixTrainer.BPDataFile))
                    bp.GeneratePrefixData(_cts.Token);
                if (!_cts.Token.IsCancellationRequested)
                    bp.RunFineTune(_cts.Token);
            });
        }

        static string FindStockfish()
        {
            // Search for any file matching stockfish*.exe in multiple locations
            string[] searchDirs = {
                ".",
                AppDomain.CurrentDomain.BaseDirectory,
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Stockfish"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Stockfish"),
            };

            foreach (string dir in searchDirs)
            {
                if (!Directory.Exists(dir)) continue;
                foreach (string file in Directory.GetFiles(dir, "stockfish*.exe"))
                    return file;
            }

            // Check PATH
            string? pathVar = Environment.GetEnvironmentVariable("PATH");
            if (pathVar != null)
            {
                foreach (string dir in pathVar.Split(';'))
                {
                    string d = dir.Trim();
                    if (!Directory.Exists(d)) continue;
                    foreach (string file in Directory.GetFiles(d, "stockfish*.exe"))
                        return file;
                }
            }

            return "";
        }

        // ── Game ──────────────────────────────────────────────────────────────

        void StartGame(bool humanWhite)
        {
            if (_player == null)
            {
                try
                {
                    _player = new ThinkingPlayer
                    {
                        Temperature = (float)_sTemp.Value,
                        ThinkingEnabled = _thinkCheck.Checked,
                        NumCandidates = (int)_sCandidates.Value,
                        NumResponses = (int)_sResponses.Value,
                        BoardEncoding = true
                    };
                }
                catch (FileNotFoundException)
                {
                    MessageBox.Show("Train a model first.", "No model", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
            }
            _player.Temperature = (float)_sTemp.Value;
            _player.ThinkingEnabled = _thinkCheck.Checked;
            _player.NumCandidates = (int)_sCandidates.Value;
            _player.NumResponses = (int)_sResponses.Value;

            _humanIsWhite = humanWhite;

            // Switch from pre-game editing to game mode
            _board.SquareClicked -= OnPreGameSquareClicked;
            _board.SquareRightClicked -= OnPreGameRightClick;
            _board.SquareClicked -= OnSquareClicked; // prevent double-attach
            _board.SquareClicked += OnSquareClicked;
            _dragFrom = -1;

            // Keep current board position (user may have moved pieces around)
            if (_game != null)
            {
                string fen = _game.ToFen();
                _game = ChessGame.FromFen(fen);
            }
            else
            {
                _game = new ChessGame();
            }
            _history = "|";
            _selectedSq = -1;
            _legalFromSelected.Clear();

            _board.Board = _game.Board;
            _board.FlipBoard = !humanWhite;
            _board.SelectedSquare = -1;
            _board.LegalTargets.Clear();
            _board.LastMoveFrom = -1;
            _board.LastMoveTo = -1;
            _board.Clickable = true;
            _drawBtn.Visible = true;
            _board.Invalidate();

            AppendLog($"─── New game — you play {(humanWhite ? "White" : "Black")} ───", TextMute);

            if (!humanWhite)
            {
                SetStatus("AI is thinking...", TextMute);
                DoAIMove();
            }
            else
            {
                SetStatus("Your turn (White).", Blue);
            }
        }

        void OnSquareClicked(object? sender, int sq)
        {
            if (_game == null || _game.IsGameOver()) return;

            bool isHumanTurn = _game.WhiteToMove == _humanIsWhite;
            if (!isHumanTurn) return;

            if (_selectedSq == -1)
            {
                // Select a friendly piece
                if (_game.IsFriendly(_game.Board[sq]))
                    SelectSquare(sq);
            }
            else
            {
                // Try to move to clicked square
                string from = ChessGame.SqName(_selectedSq);
                string to = ChessGame.SqName(sq);
                var matches = _legalFromSelected.Where(m => m.StartsWith(from + to)).ToList();

                if (matches.Count == 1)
                {
                    MakeHumanMove(matches[0]);
                }
                else if (matches.Count > 1)
                {
                    // Promotion — pick queen by default (most common)
                    MakeHumanMove(matches.First(m => m.EndsWith("q")));
                }
                else if (_game.IsFriendly(_game.Board[sq]))
                {
                    SelectSquare(sq); // switch selection
                }
                else
                {
                    ClearSelection();
                }
            }
        }

        void SelectSquare(int sq)
        {
            _selectedSq = sq;
            var allLegal = _game!.GetLegalMoves();
            string sqName = ChessGame.SqName(sq);
            _legalFromSelected = allLegal.Where(m => m[..2] == sqName).ToList();

            _board.SelectedSquare = sq;
            _board.LegalTargets = new HashSet<int>(_legalFromSelected.Select(m => ChessGame.Sq(m[2..4])));
            _board.Invalidate();
        }

        void ClearSelection()
        {
            _selectedSq = -1;
            _legalFromSelected.Clear();
            _board.SelectedSquare = -1;
            _board.LegalTargets.Clear();
            _board.Invalidate();
        }

        void MakeHumanMove(string uci)
        {
            _board.Clickable = false;
            ClearSelection();

            _game!.MakeMove(uci);
            _history += (_history == "|" ? "" : " ") + uci;

            _board.Board = _game.Board;
            _board.LastMoveFrom = ChessGame.Sq(uci[..2]);
            _board.LastMoveTo = ChessGame.Sq(uci[2..4]);
            _board.Invalidate();

            AppendLog($"You: {uci}", TextMain);

            if (CheckGameOver()) return;
            DoAIMove();
        }

        void DoAIMove()
        {
            if (_game == null || _player == null) return;
            SetStatus("AI is thinking...", TextMute);
            _board.Clickable = false;

            Task.Run(() =>
            {
                string move = _player.GetMove(_game, _history);
                UIInvoke(() =>
                {
                    if (_game == null) return;
                    _game.MakeMove(move);
                    _history += (_history == "|" ? "" : " ") + move;

                    _board.Board = _game.Board;
                    _board.LastMoveFrom = ChessGame.Sq(move[..2]);
                    _board.LastMoveTo = ChessGame.Sq(move[2..4]);
                    _board.Invalidate();

                    AppendLog($"AI: {move}", Blue);

                    if (!CheckGameOver())
                    {
                        SetStatus($"Your turn ({(_humanIsWhite ? "White" : "Black")}).", Blue);
                        _board.Clickable = true;
                    }
                });
            });
        }

        // ── Pre-game piece moving ────────────────────────────────────────────
        // When no game is active, click a piece then click a destination to move it.
        // This lets you set up any position before clicking Play.

        void RestorePreGameEditing()
        {
            // Free model from GPU when game ends
            _player?.Dispose(); _player = null;

            _game = new ChessGame();
            _board.Board = _game.Board;
            _board.SelectedSquare = -1;
            _board.LegalTargets.Clear();
            _board.LastMoveFrom = -1;
            _board.LastMoveTo = -1;
            _board.Clickable = true;
            _dragFrom = -1;

            // Remove all handlers then attach pre-game ones cleanly
            _board.SquareClicked -= OnSquareClicked;
            _board.SquareClicked -= OnPreGameSquareClicked;
            _board.SquareRightClicked -= OnPreGameRightClick;
            _board.SquareClicked += OnPreGameSquareClicked;
            _board.SquareRightClicked += OnPreGameRightClick;
            _board.Invalidate();
        }

        void OnPreGameSquareClicked(object? sender, int sq)
        {
            if (_game == null)
            {
                _game = new ChessGame();
                _board.Board = _game.Board;
            }

            if (_dragFrom == -1)
            {
                // Pick up a piece
                if (_game.Board[sq] != '.')
                {
                    _dragFrom = sq;
                    _board.SelectedSquare = sq;
                    _board.Invalidate();
                    SetStatus($"Moving {_game.Board[sq]} — click destination (right-click to remove piece).", Yellow);
                }
            }
            else
            {
                // Drop the piece
                _game.Board[sq] = _game.Board[_dragFrom];
                _game.Board[_dragFrom] = '.';
                _dragFrom = -1;
                _board.SelectedSquare = -1;
                _board.Board = _game.Board;
                _board.Invalidate();
                SetStatus("Move pieces around, then click Play.", TextMute);
            }
        }

        void OnPreGameRightClick(object? sender, int sq)
        {
            if (_game == null) return;
            // Right-click removes a piece
            _game.Board[sq] = '.';
            _dragFrom = -1;
            _board.SelectedSquare = -1;
            _board.Board = _game.Board;
            _board.Invalidate();
        }

        void ClaimDraw()
        {
            if (_game == null || _game.IsGameOver()) return;
            _board.Clickable = false;
            _drawBtn.Visible = false;
            SetStatus("Draw by agreement.", Yellow);
            AppendLog("Draw by agreement.", Yellow);
            AppendLog("Click Play as White / Black to start a new game.", TextMute);
            RestorePreGameEditing();
        }

        bool CheckGameOver()
        {
            if (_game == null || !_game.IsGameOver()) return false;
            _board.Clickable = false;
            _drawBtn.Visible = false;

            string result = _game.GetResult();
            bool humanWon = (_humanIsWhite && result == "W") || (!_humanIsWhite && result == "L");
            bool aiWon    = (_humanIsWhite && result == "L") || (!_humanIsWhite && result == "W");

            if (humanWon) { SetStatus("You win!", Green); AppendLog("You win!", Green); }
            else if (aiWon) { SetStatus("AI wins!", Red); AppendLog("AI wins!", Red); }
            else { SetStatus("Draw!", Yellow); AppendLog("Draw!", Yellow); }

            if (_game.IsCheckmate()) AppendLog("Checkmate!", Yellow);
            else if (_game.IsStalemate()) AppendLog("Stalemate.", TextMute);

            AppendLog("Move pieces around or click Play to start a new game.", TextMute);
            RestorePreGameEditing();
            return true;
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        void SetStatus(string text, Color color) { _statusLabel.Text = text; _statusLabel.ForeColor = color; }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _cts?.Cancel();
            _player?.Dispose(); _player = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            base.OnFormClosing(e);
        }

        void AppendLog(string text, Color color)
        {
            UIInvoke(() =>
            {
                _log.SelectionStart = _log.TextLength;
                _log.SelectionLength = 0;
                _log.SelectionColor = color;
                _log.AppendText(text + "\n");
                _log.ScrollToCaret();
            });
        }

        void UIInvoke(Action a) { if (InvokeRequired) Invoke(a); else a(); }
    }
}
