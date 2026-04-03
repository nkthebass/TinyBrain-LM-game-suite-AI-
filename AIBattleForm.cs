using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;
using Color = System.Drawing.Color;
using Size = System.Drawing.Size;
using Point = System.Drawing.Point;

namespace ChessLLM
{
    public class AIBattleForm : Form
    {
        static readonly Color BgDark  = Color.FromArgb(30, 31, 36);
        static readonly Color BgLight = Color.FromArgb(40, 42, 48);
        static readonly Color BgInput = Color.FromArgb(50, 52, 60);
        static readonly Color TextMain = Color.FromArgb(230, 232, 236);
        static readonly Color TextMute = Color.FromArgb(100, 106, 118);
        static readonly Color Green   = Color.FromArgb(0, 132, 80);
        static readonly Color Blue    = Color.FromArgb(100, 180, 255);
        static readonly Color Red     = Color.FromArgb(200, 60, 60);
        static readonly Color Yellow  = Color.FromArgb(180, 140, 40);

        ChessBoardPanel _board = null!;
        RichTextBox     _log   = null!;
        Button _loadWhiteBtn = null!, _loadBlackBtn = null!, _whiteBoardBtn = null!, _blackBoardBtn = null!;
        Button _whiteThinkBtn = null!, _blackThinkBtn = null!;
        Button _startBtn = null!, _stepBtn = null!, _autoBtn = null!, _maxSpeedBtn = null!;
        bool _whiteBoardEnc = true, _blackBoardEnc = true;
        bool _whiteThinking = false, _blackThinking = false;
        Label  _whiteLabel = null!, _blackLabel = null!, _statusLabel = null!;

        // AI models
        string? _whitePath, _blackPath;
        TransformerModel? _whiteModel, _blackModel;
        CharTokenizer? _whiteTokenizer, _blackTokenizer;
        int _whiteCtx, _blackCtx;
        Device _device = null!;

        // Game
        ChessGame? _game;
        string _history = "|";
        int _moveCount;
        bool _autoPlaying;
        bool _maxSpeed;
        System.Windows.Forms.Timer _autoTimer = null!;

        public AIBattleForm()
        {
            BuildUI();
            _device = cuda.is_available() ? CUDA : CPU;
        }

        void BuildUI()
        {
            Text = "AI vs AI Battle";
            ClientSize = new Size(900, 640);
            BackColor = BgDark;
            ForeColor = TextMain;
            Font = new Font("Segoe UI", 9f);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterParent;

            // Board
            _board = new ChessBoardPanel
            {
                Location = new Point(10, 10),
                Size = new Size(480, 480)
            };
            _board.Board = new ChessGame().Board;
            Controls.Add(_board);

            // Right panel
            int rx = 510;

            // White AI
            Controls.Add(new Label { Text = "WHITE AI", Location = new Point(rx, 10), ForeColor = TextMute, Font = new Font("Segoe UI", 8f, FontStyle.Bold), AutoSize = true });
            _loadWhiteBtn = MkBtn("Load Model...", Blue, new Point(rx, 30), 160, 30);
            _loadWhiteBtn.Click += (_, __) => LoadModel(true);
            Controls.Add(_loadWhiteBtn);
            _whiteBoardBtn = MkBtn("Board: ON", Green, new Point(rx + 170, 30), 85, 30);
            _whiteBoardBtn.Click += (_, __) => ToggleBoardEnc(true);
            Controls.Add(_whiteBoardBtn);
            _whiteThinkBtn = MkBtn("Think: OFF", Color.FromArgb(80, 80, 90), new Point(rx + 265, 30), 85, 30);
            _whiteThinkBtn.Click += (_, __) => ToggleThinking(true);
            Controls.Add(_whiteThinkBtn);
            _whiteLabel = new Label { Location = new Point(rx, 65), Size = new Size(370, 18), ForeColor = TextMute, Text = "No model loaded" };
            Controls.Add(_whiteLabel);

            // Black AI
            Controls.Add(new Label { Text = "BLACK AI", Location = new Point(rx, 95), ForeColor = TextMute, Font = new Font("Segoe UI", 8f, FontStyle.Bold), AutoSize = true });
            _loadBlackBtn = MkBtn("Load Model...", Red, new Point(rx, 115), 160, 30);
            _loadBlackBtn.Click += (_, __) => LoadModel(false);
            Controls.Add(_loadBlackBtn);
            _blackBoardBtn = MkBtn("Board: ON", Green, new Point(rx + 170, 115), 85, 30);
            _blackBoardBtn.Click += (_, __) => ToggleBoardEnc(false);
            Controls.Add(_blackBoardBtn);
            _blackThinkBtn = MkBtn("Think: OFF", Color.FromArgb(80, 80, 90), new Point(rx + 265, 115), 85, 30);
            _blackThinkBtn.Click += (_, __) => ToggleThinking(false);
            Controls.Add(_blackThinkBtn);
            _blackLabel = new Label { Location = new Point(rx, 150), Size = new Size(370, 18), ForeColor = TextMute, Text = "No model loaded" };
            Controls.Add(_blackLabel);

            // Controls
            _startBtn = MkBtn("New Game", Green, new Point(rx, 185), 95, 32);
            _stepBtn  = MkBtn("Step", Yellow, new Point(rx + 100, 185), 60, 32);
            _autoBtn  = MkBtn("Auto", Color.FromArgb(90, 80, 140), new Point(rx + 165, 185), 60, 32);
            _maxSpeedBtn = MkBtn("Max Speed", Color.FromArgb(200, 80, 40), new Point(rx + 230, 185), 90, 32);
            _startBtn.Click += (_, __) => StartBattle();
            _stepBtn.Click  += (_, __) => PlayOneMove();
            _autoBtn.Click  += (_, __) => ToggleAutoPlay();
            _maxSpeedBtn.Click += (_, __) => ToggleMaxSpeed();
            _stepBtn.Enabled = false;
            _autoBtn.Enabled = false;
            _maxSpeedBtn.Enabled = false;
            Controls.Add(_startBtn);
            Controls.Add(_stepBtn);
            Controls.Add(_autoBtn);
            Controls.Add(_maxSpeedBtn);

            _statusLabel = new Label { Location = new Point(rx, 225), Size = new Size(370, 20), ForeColor = TextMute };
            Controls.Add(_statusLabel);

            // Log
            _log = new RichTextBox
            {
                Location = new Point(rx, 250), Size = new Size(370, 370),
                BackColor = BgInput, ForeColor = TextMain,
                Font = new Font("Consolas", 9.5f), ReadOnly = true,
                BorderStyle = BorderStyle.None, ScrollBars = RichTextBoxScrollBars.Vertical
            };
            Controls.Add(_log);

            // Auto-play timer
            _autoTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _autoTimer.Tick += (_, __) => PlayOneMove();
        }

        Button MkBtn(string text, Color bg, Point loc, int w, int h)
        {
            var b = new Button
            {
                Text = text, Location = loc, Size = new Size(w, h),
                FlatStyle = FlatStyle.Flat, Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                BackColor = bg, ForeColor = Color.White, Cursor = Cursors.Hand
            };
            b.FlatAppearance.BorderSize = 0;
            return b;
        }

        // ── Model loading ────────────────────────────────────────────────────

        void ToggleBoardEnc(bool forWhite)
        {
            if (forWhite)
            {
                _whiteBoardEnc = !_whiteBoardEnc;
                _whiteBoardBtn.Text = _whiteBoardEnc ? "Board: ON" : "Board: OFF";
                _whiteBoardBtn.BackColor = _whiteBoardEnc ? Green : Color.FromArgb(80, 80, 90);
            }
            else
            {
                _blackBoardEnc = !_blackBoardEnc;
                _blackBoardBtn.Text = _blackBoardEnc ? "Board: ON" : "Board: OFF";
                _blackBoardBtn.BackColor = _blackBoardEnc ? Green : Color.FromArgb(80, 80, 90);
            }
        }

        void ToggleThinking(bool forWhite)
        {
            if (forWhite)
            {
                _whiteThinking = !_whiteThinking;
                _whiteThinkBtn.Text = _whiteThinking ? "Think: ON" : "Think: OFF";
                _whiteThinkBtn.BackColor = _whiteThinking ? Color.FromArgb(60, 140, 160) : Color.FromArgb(80, 80, 90);
            }
            else
            {
                _blackThinking = !_blackThinking;
                _blackThinkBtn.Text = _blackThinking ? "Think: ON" : "Think: OFF";
                _blackThinkBtn.BackColor = _blackThinking ? Color.FromArgb(60, 140, 160) : Color.FromArgb(80, 80, 90);
            }
        }

        async void LoadModel(bool forWhite)
        {
            // Run file dialog on a fresh STA thread to avoid CUDA/COM conflict
            string? modelFile = null;
            var thread = new System.Threading.Thread(() =>
            {
                using var dlg = new OpenFileDialog
                {
                    Title = $"Select chess_model.pt for {(forWhite ? "White" : "Black")} AI",
                    Filter = "Model files (*.pt)|*.pt|All files|*.*"
                };
                if (dlg.ShowDialog() == DialogResult.OK)
                    modelFile = dlg.FileName;
            });
            thread.SetApartmentState(System.Threading.ApartmentState.STA);
            thread.Start();
            thread.Join();

            if (modelFile == null) return;
            string folder = Path.GetDirectoryName(modelFile) ?? ".";
            string archFile = Path.Combine(folder, "chess_arch.json");
            string tokFile  = Path.Combine(folder, "chess_tokenizer.bin");

            var label = forWhite ? _whiteLabel : _blackLabel;
            label.Text = "Loading...";
            label.ForeColor = Yellow;
            _loadWhiteBtn.Enabled = false;
            _loadBlackBtn.Enabled = false;

            try
            {
                // Load on background thread so UI stays responsive
                var (model, tokenizer, arch, paramStr) = await Task.Run(() =>
                {
                    ArchConfig a;
                    if (File.Exists(archFile))
                    {
                        string json = File.ReadAllText(archFile);
                        a = System.Text.Json.JsonSerializer.Deserialize<ArchConfig>(json) ?? new ArchConfig();
                    }
                    else a = new ArchConfig();

                    var tok = new CharTokenizer();
                    if (File.Exists(tokFile)) tok.Load(tokFile);
                    else tok.Build("| abcdefgh12345678qrbnWLD\n");

                    var m = new TransformerModel("battle", tok.VocabSize,
                        a.ContextSize, a.EmbedDim, a.NumHeads, a.FfDim, a.NumLayers, 0.0);
                    m.load(modelFile);
                    m.to(_device);
                    m.eval();

                    long pc = m.parameters().Sum(p => p.numel());
                    string ps = pc >= 1_000_000 ? $"{pc / 1_000_000.0:F1}M" : $"{pc / 1_000.0:F0}K";

                    return (m, tok, a, ps);
                });

                if (forWhite)
                {
                    _whiteModel?.Dispose();
                    _whiteModel = model; _whiteTokenizer = tokenizer;
                    _whiteCtx = arch.ContextSize; _whitePath = folder;
                }
                else
                {
                    _blackModel?.Dispose();
                    _blackModel = model; _blackTokenizer = tokenizer;
                    _blackCtx = arch.ContextSize; _blackPath = folder;
                }

                label.Text = $"{Path.GetFileName(folder)} ({paramStr} params)";
                label.ForeColor = Color.FromArgb(0, 180, 100);
                AppendLog($"Loaded {(forWhite ? "White" : "Black")}: {Path.GetFileName(folder)} ({paramStr})", Color.FromArgb(0, 180, 100));
            }
            catch (Exception ex)
            {
                label.Text = "Failed to load";
                label.ForeColor = Red;
                MessageBox.Show($"Failed to load model: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                _loadWhiteBtn.Enabled = true;
                _loadBlackBtn.Enabled = true;
            }
        }

        // ── Battle ───────────────────────────────────────────────────────────

        void StartBattle()
        {
            if (_whiteModel == null || _blackModel == null)
            {
                MessageBox.Show("Load both White and Black models first.", "Missing model", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _game = new ChessGame();
            _history = "|";
            _board.Board = _game.Board;
            _board.SelectedSquare = -1;
            _board.LegalTargets.Clear();
            _board.LastMoveFrom = -1;
            _board.LastMoveTo = -1;
            _board.Invalidate();

            _moveCount = 0;
            _stepBtn.Enabled = true;
            _autoBtn.Enabled = true;
            _maxSpeedBtn.Enabled = true;
            _statusLabel.Text = "White to move.";
            _statusLabel.ForeColor = TextMute;

            _log.Clear();
            AppendLog("=== New Battle ===", TextMute);
            AppendLog($"White: {Path.GetFileName(_whitePath ?? "?")}", Blue);
            AppendLog($"Black: {Path.GetFileName(_blackPath ?? "?")}", Red);
            AppendLog("", TextMain);
        }

        void PlayOneMove()
        {
            if (_game == null || _game.IsGameOver())
            {
                StopAutoPlay();
                return;
            }

            var legalMoves = _game.GetLegalMoves();
            if (legalMoves.Count == 0) { CheckResult(); return; }

            bool isWhite = _game.WhiteToMove;
            var model     = isWhite ? _whiteModel! : _blackModel!;
            var tokenizer = isWhite ? _whiteTokenizer! : _blackTokenizer!;
            int ctx       = isWhite ? _whiteCtx : _blackCtx;

            bool useBoard = isWhite ? _whiteBoardEnc : _blackBoardEnc;
            bool useThink = isWhite ? _whiteThinking : _blackThinking;
            string move = GetBestMove(model, tokenizer, ctx, _history, legalMoves, _game, useBoard, useThink);

            int from = ChessGame.Sq(move[..2]);
            int to   = ChessGame.Sq(move[2..4]);

            _game.MakeMove(move);
            _history += (_history == "|" ? "" : " ") + move;

            _board.Board = _game.Board;
            _board.LastMoveFrom = from;
            _board.LastMoveTo = to;
            _board.Invalidate();

            _moveCount++;
            string side = isWhite ? "White" : "Black";
            int moveNum = (_game.FullMoves);
            AppendLog($"{moveNum}. {side}: {move}", isWhite ? Blue : Red);

            if (_game.IsGameOver())
            {
                CheckResult();
                return;
            }

            _statusLabel.Text = $"{(_game.WhiteToMove ? "White" : "Black")} to move. (Move {_game.FullMoves})";
        }

        void CheckResult()
        {
            StopAutoPlay();
            _stepBtn.Enabled = false;
            _autoBtn.Enabled = false;

            string result = _game!.GetResult();
            if (result == "W")
            {
                _statusLabel.Text = "White wins!"; _statusLabel.ForeColor = Blue;
                AppendLog("\nWhite wins!", Blue);
            }
            else if (result == "L")
            {
                _statusLabel.Text = "Black wins!"; _statusLabel.ForeColor = Red;
                AppendLog("\nBlack wins!", Red);
            }
            else
            {
                _statusLabel.Text = "Draw!"; _statusLabel.ForeColor = Yellow;
                AppendLog("\nDraw!", Yellow);
                if (_game.IsStalemate()) AppendLog("Stalemate.", TextMute);
                else if (_game.Is50MoveRule()) AppendLog("50-move rule.", TextMute);
            }
        }

        void ToggleAutoPlay()
        {
            if (_autoPlaying) StopAutoPlay();
            else
            {
                _autoPlaying = true;
                _maxSpeed = false;
                _autoBtn.Text = "Stop";
                _autoBtn.BackColor = Red;
                _stepBtn.Enabled = false;
                _maxSpeedBtn.Enabled = false;
                _autoTimer.Interval = 500;
                _autoTimer.Start();
            }
        }

        void ToggleMaxSpeed()
        {
            if (_maxSpeed) { StopAutoPlay(); return; }

            _maxSpeed = true;
            _autoPlaying = true;
            _maxSpeedBtn.Text = "Stop";
            _maxSpeedBtn.BackColor = Red;
            _autoBtn.Enabled = false;
            _stepBtn.Enabled = false;

            // Run moves as fast as possible on a background thread
            Task.Run(() =>
            {
                while (_maxSpeed && _game != null && !_game.IsGameOver())
                {
                    try { this.Invoke(() => PlayOneMove()); }
                    catch { break; }
                }
                try { this.Invoke(() => StopAutoPlay()); } catch { }
            });
        }

        void StopAutoPlay()
        {
            _autoPlaying = false;
            _maxSpeed = false;
            _autoTimer.Stop();
            _autoBtn.Text = "Auto";
            _autoBtn.BackColor = Color.FromArgb(90, 80, 140);
            _autoBtn.Enabled = true;
            _maxSpeedBtn.Text = "Max Speed";
            _maxSpeedBtn.BackColor = Color.FromArgb(200, 80, 40);
            _maxSpeedBtn.Enabled = true;
            bool gameActive = _game != null && !_game.IsGameOver();
            _stepBtn.Enabled = gameActive;
            _autoBtn.Enabled = gameActive;
            _maxSpeedBtn.Enabled = gameActive;
        }

        // ── Model inference ──────────────────────────────────────────────────

        string GetBestMove(TransformerModel model, CharTokenizer tokenizer, int ctxSize,
                          string history, List<string> legalMoves, ChessGame game,
                          bool useBoardEnc, bool useThinking)
        {
            if (legalMoves.Count == 1) return legalMoves[0];

            var scored = ScoreMoves(model, tokenizer, ctxSize, history, legalMoves, game, useBoardEnc);

            if (!useThinking)
            {
                // Simple sampling
                float temp = 0.5f;
                var scores = scored.Select(s => s.score / temp).ToArray();
                float max = scores.Max();
                float[] exps = scores.Select(s => MathF.Exp(s - max)).ToArray();
                float sum = exps.Sum();
                double r = Random.Shared.NextDouble();
                double cum = 0;
                for (int i = 0; i < exps.Length; i++)
                {
                    cum += exps[i] / sum;
                    if (r <= cum) return scored[i].move;
                }
                return scored[0].move;
            }

            // ── Tactical thinking: trust model instinct, correct tactical errors ──
            bool isWhiteSide = game.WhiteToMove;
            float ourMaterial = CountMat(game, isWhiteSide);

            var candidates = scored.OrderByDescending(s => s.score).Take(5).ToList();
            float bestTotal = float.NegativeInfinity;
            string bestMove = candidates[0].move;

            foreach (var cand in candidates)
            {
                var vGame = game.Clone();
                vGame.MakeMove(cand.move);
                if (vGame.IsCheckmate()) return cand.move;
                if (vGame.IsGameOver()) { if (-100f > bestTotal) { bestTotal = -100f; bestMove = cand.move; } continue; }

                float matAfterOur = CountMat(vGame, isWhiteSide);
                float matGain = matAfterOur - ourMaterial;

                float worstLoss = 0;
                bool opCanMate = false;

                string vHistory = history + " " + cand.move;
                var opMoves = vGame.GetLegalMoves();
                var opScored = ScoreMoves(model, tokenizer, ctxSize, vHistory, opMoves, vGame, useBoardEnc);

                foreach (var resp in opScored.OrderByDescending(s => s.score).Take(10))
                {
                    var aGame = vGame.Clone();
                    aGame.MakeMove(resp.move);
                    if (aGame.IsCheckmate()) { opCanMate = true; break; }
                    float matAfterResp = CountMat(aGame, isWhiteSide);
                    worstLoss = Math.Max(worstLoss, matAfterOur - matAfterResp);
                }

                if (opCanMate) { float s2 = cand.score - 1000f; if (s2 > bestTotal) { bestTotal = s2; bestMove = cand.move; } continue; }

                float total = cand.score * 2f + matGain * 8f - worstLoss * 6f + (vGame.IsInCheck() ? 1.5f : 0f);
                if (total > bestTotal) { bestTotal = total; bestMove = cand.move; }
            }

            return bestMove;
        }

        static float CountMat(ChessGame game, bool forWhite)
        {
            float m = 0;
            foreach (char c in game.Board)
                m += c switch { 'P'=>1f,'N'=>3.2f,'B'=>3.3f,'R'=>5f,'Q'=>9f,'p'=>-1f,'n'=>-3.2f,'b'=>-3.3f,'r'=>-5f,'q'=>-9f,_=>0f };
            return forWhite ? m : -m;
        }

        List<(string move, float score)> ScoreMoves(TransformerModel model, CharTokenizer tokenizer,
            int ctxSize, string history, List<string> legalMoves, ChessGame game, bool useBoardEnc)
        {
            string boardPrefix = useBoardEnc ? BoardEncoder.Encode(game) : "";
            string prefix = boardPrefix + history + " ";
            var results = new List<(string, float)>(legalMoves.Count);

            using (no_grad())
            using (torch.NewDisposeScope())
            {
                int[] ids = tokenizer.Encode(prefix);
                if (ids.Length > ctxSize) ids = ids.Skip(ids.Length - ctxSize).ToArray();

                long[] data = ids.Select(x => (long)x).ToArray();
                var input = tensor(data, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
                var logits = model.forward(input);
                var logitArr = logits[0, ids.Length - 1].data<float>().ToArray();

                foreach (string uci in legalMoves)
                {
                    float score = 0;
                    for (int i = 0; i < uci.Length && i < 4; i++)
                        if (tokenizer.CharToId.TryGetValue(uci[i], out int id) && id < logitArr.Length)
                            score += logitArr[id] * (i == 0 ? 1f : 0.3f);
                    results.Add((uci, score));
                }
            }
            return results;
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        void AppendLog(string text, Color color)
        {
            _log.SelectionStart = _log.TextLength;
            _log.SelectionLength = 0;
            _log.SelectionColor = color;
            _log.AppendText(text + "\n");
            _log.ScrollToCaret();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _autoTimer.Stop();
            _whiteModel?.Dispose();
            _blackModel?.Dispose();
            base.OnFormClosing(e);
        }
    }
}
