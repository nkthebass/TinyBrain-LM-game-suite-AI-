using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace ChessLLM
{
    /// <summary>
    /// Reinforcement Learning trainer using Stockfish as teacher.
    ///
    /// Stage 1: Supervised (already done via Trainer.cs — just predict human moves)
    /// Stage 2: Stockfish-corrected self-play (DAgger)
    ///          - Model plays games against itself
    ///          - Stockfish evaluates each move at depth 10
    ///          - If centipawn loss > threshold: replace with Stockfish's best move
    ///          - Train on the corrected game sequences
    /// Stage 3: Hard RL (REINFORCE-style)
    ///          - Model plays, Stockfish at depth 18+ evaluates
    ///          - Policy gradient: good moves get reinforced, bad moves get penalized
    ///          - Steep penalty for moves >50cp worse than Stockfish's best
    /// </summary>
    public class RLTrainer
    {
        public const string RLDataFile = "chess_rl_data.txt";

        // Config
        public ArchConfig Arch        { get; set; } = new();
        public string StockfishPath   { get; set; } = "stockfish.exe";
        public int    SfDepth         { get; set; } = 10;
        public int    SfParallel      { get; set; } = 4;  // number of parallel SF instances
        public int    NumSelfPlayGames{ get; set; } = 500;
        public int    StartAtMove     { get; set; } = 0;   // 0=full game, 20=start at move 20 (midgame), 40+=endgame
        public int    CpLossThreshold { get; set; } = 50;  // centipawns — moves worse than this get corrected
        public int    TotalSteps      { get; set; } = 10_000;
        public int    WarmupSteps     { get; set; } = 500;
        public float  LearningRate    { get; set; } = 1e-4f;  // lower LR for fine-tuning
        public int    BatchSize       { get; set; } = 16;
        public float  GradientClip    { get; set; } = 1.0f;
        public float  WeightDecay     { get; set; } = 0.01f;
        public int    LogEvery        { get; set; } = 100;

        // Callbacks
        public Action<string, Color>? OnLog        { get; set; }
        public Action<int, int, float>? OnProgress { get; set; }
        public Action? OnComplete                  { get; set; }

        const string Vocab = "| abcdefgh12345678qrbnWLD\n";
        int _ctxSize;

        // ── Stage 2: Generate Stockfish-corrected self-play data ──────────────

        public void GenerateCorrectedData(CancellationToken cancel = default)
        {
            Log("Loading model for self-play...", Color.FromArgb(100, 106, 118));

            var tokenizer = new CharTokenizer();
            tokenizer.Build(Vocab);

            // Load the trained model for self-play
            if (!File.Exists(Trainer.ModelPath))
            {
                Log("No trained model found. Run supervised training first (Stage 1).", Color.FromArgb(200, 60, 60));
                OnComplete?.Invoke();
                return;
            }

            var arch = ArchConfig.Load();
            _ctxSize = arch.ContextSize;
            var device = cuda.is_available() ? CUDA : CPU;

            var model = new TransformerModel("chess", tokenizer.VocabSize,
                arch.ContextSize, arch.EmbedDim, arch.NumHeads, arch.FfDim, arch.NumLayers, 0.0);
            model.load(Trainer.ModelPath);
            model.to(device);
            model.eval();

            Log($"Model loaded. Generating {NumSelfPlayGames} self-play games with SF depth {SfDepth}...",
                Color.FromArgb(100, 106, 118));
            Log($"Parallel Stockfish instances: {SfParallel}", Color.FromArgb(100, 106, 118));
            Log($"Centipawn loss threshold: {CpLossThreshold}cp", Color.FromArgb(100, 106, 118));

            // Generate games in parallel batches
            int completed = 0;
            var allGames = new List<string>();
            var lockObj = new object();

            // ── Phase 1: Fast self-play on GPU (no Stockfish) ─────────────────
            Log("Phase 1: Fast self-play (GPU only)...", Color.FromArgb(100, 180, 255));

            var rawGames = new List<(List<string> moves, List<string> fens)>();

            for (int i = 0; i < NumSelfPlayGames; i++)
            {
                if (cancel.IsCancellationRequested) break;

                var (moves, fens) = PlayGameFast(model, tokenizer, device);
                if (moves.Count >= 6)
                    rawGames.Add((moves, fens));

                if ((i + 1) % 5 == 0 || i == NumSelfPlayGames - 1)
                    Log($"  Self-play: {i + 1}/{NumSelfPlayGames} games ({rawGames.Count} valid)",
                        Color.FromArgb(100, 106, 118));
            }

            GC.Collect(); GC.WaitForPendingFinalizers();

            // ── Phase 2: Batch Stockfish correction (CPU) ────────────────────
            Log($"Phase 2: Stockfish correction at depth {SfDepth} ({rawGames.Count} games)...",
                Color.FromArgb(100, 180, 255));

            StockfishEvaluator? sf = null;
            try
            {
                sf = new StockfishEvaluator(StockfishPath, SfDepth);

                for (int g = 0; g < rawGames.Count; g++)
                {
                    if (cancel.IsCancellationRequested) break;

                    var (moves, fens) = rawGames[g];
                    int corrections = 0;

                    // Correct bad moves with Stockfish
                    for (int m = StartAtMove; m < moves.Count && m < fens.Count; m++)
                    {
                        try
                        {
                            var (bestMove, _) = sf.Evaluate(fens[m]);
                            if (!string.IsNullOrEmpty(bestMove) && bestMove != moves[m])
                            {
                                // Verify bestMove is legal at that position
                                var tempGame = ChessGame.FromFen(fens[m]);
                                if (tempGame.GetLegalMoves().Contains(bestMove))
                                {
                                    moves[m] = bestMove;
                                    corrections++;
                                }
                            }
                        }
                        catch { }
                    }

                    // Build the game string
                    string result = "D"; // default
                    var verifyGame = new ChessGame();
                    foreach (var mv in moves)
                    {
                        if (!verifyGame.MakeMove(mv)) break;
                    }
                    string r = verifyGame.GetResult();
                    if (r != "") result = r;

                    allGames.Add("|" + string.Join(' ', moves) + result);
                    completed++;

                    if ((g + 1) % 5 == 0 || g == rawGames.Count - 1)
                        Log($"  SF correction: {g + 1}/{rawGames.Count} games ({corrections} corrections this game)",
                            Color.FromArgb(100, 106, 118));
                }
            }
            finally
            {
                sf?.Dispose();
            }

            // Append to existing training data
            bool hasExisting = File.Exists(DatasetForm.DataFile);
            using (var writer = new StreamWriter(RLDataFile, false, Encoding.UTF8))
            {
                foreach (var game in allGames)
                    writer.WriteLine(game);
            }

            // Free the self-play model from GPU
            model.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Log($"Generated {allGames.Count} corrected games → {RLDataFile}", Color.FromArgb(0, 180, 100));
        }

        /// <summary>
        /// Play one game with the model, correct bad moves with Stockfish.
        /// Returns UCI game string in training format, or null on error.
        /// </summary>
        /// <summary>Play a full game using only the model — no Stockfish. Very fast.</summary>
        (List<string> moves, List<string> fens) PlayGameFast(
            TransformerModel model, CharTokenizer tokenizer, Device device)
        {
            var game = new ChessGame();
            var moves = new List<string>();
            var fens = new List<string>();
            string history = "|";

            // Skip opening with random legal moves if StartAtMove > 0
            for (int skip = 0; skip < StartAtMove && !game.IsGameOver(); skip++)
            {
                var legal = game.GetLegalMoves();
                if (legal.Count == 0) break;
                string move = legal[Random.Shared.Next(legal.Count)];
                moves.Add(move);
                fens.Add(game.ToFen());
                game.MakeMove(move);
                history += (history == "|" ? "" : " ") + move;
            }

            // Model plays the rest
            for (int m = moves.Count; m < 150 && !game.IsGameOver(); m++)
            {
                var legalMoves = game.GetLegalMoves();
                if (legalMoves.Count == 0) break;

                fens.Add(game.ToFen());
                string move = GetModelMove(model, tokenizer, device, game, history, legalMoves);
                moves.Add(move);
                game.MakeMove(move);
                history += (history == "|" ? "" : " ") + move;
            }

            return (moves, fens);
        }

        /// <summary>Old method kept for reference — not used in the fast pipeline.</summary>
        string? PlayAndCorrectGame(TransformerModel model, CharTokenizer tokenizer,
                                    Device device, StockfishEvaluator sf)
        {
            var game = new ChessGame();
            var moves = new List<string>();
            string history = "|";
            int maxMoves = 150;
            int corrections = 0;

            // Fast-forward to StartAtMove using random Stockfish moves
            // This lets the model train on midgame/endgame positions specifically
            if (StartAtMove > 0)
            {
                for (int skip = 0; skip < StartAtMove && !game.IsGameOver(); skip++)
                {
                    var legal = game.GetLegalMoves();
                    if (legal.Count == 0) break;

                    string move;
                    try
                    {
                        var (best, _) = sf.Evaluate(game.ToFen());
                        move = (!string.IsNullOrEmpty(best) && legal.Contains(best)) ? best : legal[0];
                    }
                    catch { move = legal[Random.Shared.Next(legal.Count)]; }

                    moves.Add(move);
                    game.MakeMove(move);
                    history += (history == "|" ? "" : " ") + move;
                }

                if (game.IsGameOver()) return null; // game ended during skip
            }

            for (int m = moves.Count; m < maxMoves && !game.IsGameOver(); m++)
            {
                var legalMoves = game.GetLegalMoves();
                if (legalMoves.Count == 0) break;

                // Model picks a move (single forward pass — fast)
                string modelMove = GetModelMove(model, tokenizer, device, game, history, legalMoves);

                // Stockfish evaluates: just get the best move (ONE call, not two)
                string fen = game.ToFen();
                string finalMove = modelMove;

                try
                {
                    // ONE Stockfish call: get best move + score from current position
                    var (bestMove, scoreBefore) = sf.Evaluate(fen);

                    if (modelMove != bestMove && !string.IsNullOrEmpty(bestMove) && legalMoves.Contains(bestMove))
                    {
                        // Make model's move, evaluate resulting position (opponent's perspective)
                        var tempGame = game.Clone();
                        tempGame.MakeMove(modelMove);
                        var (_, scoreAfter) = sf.Evaluate(tempGame.ToFen());

                        // Centipawn loss = how much worse our move is
                        // scoreBefore is from our perspective, scoreAfter is from opponent's
                        int cpLoss = scoreBefore + scoreAfter; // both should be positive if we're losing ground

                        if (cpLoss > CpLossThreshold)
                        {
                            finalMove = bestMove;
                            corrections++;
                        }
                    }
                }
                catch { }

                moves.Add(finalMove);
                game.MakeMove(finalMove);
                history += (history == "|" ? "" : " ") + finalMove;
            }

            if (moves.Count < 6) return null;

            string result = game.GetResult();
            if (result == "") result = "D";
            return "|" + string.Join(' ', moves) + result;
        }

        /// <summary>
        /// Pick a move using ONE forward pass instead of scoring every legal move.
        /// Feed the history + space, get logits for the first char of the next move,
        /// then greedily pick the best legal move starting with that char.
        /// Falls back to random if the model's preference isn't legal.
        /// </summary>
        string GetModelMove(TransformerModel model, CharTokenizer tokenizer, Device device,
                           ChessGame game, string history, List<string> legalMoves)
        {
            if (legalMoves.Count == 1) return legalMoves[0];

            using (no_grad())
            using (torch.NewDisposeScope())
            {
                string prefix = history + " ";
                int[] ids = tokenizer.Encode(prefix);
                int ctxSize = _ctxSize;
                if (ids.Length > ctxSize)
                    ids = ids.Skip(ids.Length - ctxSize).ToArray();

                long[] data = ids.Select(x => (long)x).ToArray();
                var input = tensor(data, dtype: ScalarType.Int64, device: device).unsqueeze(0);
                var logits = model.forward(input);

                // Get logits at the last position (predicts first char of next move)
                var lastLogits = logits[0, ids.Length - 1];
                var logitArr = lastLogits.data<float>().ToArray();

                // Score legal moves by their first character's probability
                // Then break ties by second char, etc.
                string bestMove = legalMoves[0];
                float bestScore = float.NegativeInfinity;

                foreach (string uci in legalMoves)
                {
                    float score = 0;
                    // Score first char (from the already-computed logits)
                    if (tokenizer.CharToId.TryGetValue(uci[0], out int id0))
                        score += logitArr[id0];
                    // Add small bonus for subsequent chars to break ties
                    for (int i = 1; i < uci.Length && i < 3; i++)
                    {
                        if (tokenizer.CharToId.TryGetValue(uci[i], out int id))
                            score += logitArr[id] * 0.1f; // diminishing weight
                    }

                    if (score > bestScore) { bestScore = score; bestMove = uci; }
                }

                return bestMove;
            }
        }

        // ── Stage 2/3: Fine-tune on corrected data ───────────────────────────

        public void RunFineTune(CancellationToken cancel = default)
        {
            var device = cuda.is_available() ? CUDA : CPU;
            Log($"Device: {(cuda.is_available() ? "CUDA" : "CPU")}", Color.FromArgb(100, 106, 118));

            var tokenizer = new CharTokenizer();
            tokenizer.Build(Vocab);

            // Tokenize ONLY the RL data — don't retokenize the full 6GB supervised dataset.
            // The RL data is small (~1000 games) so this is instant.
            if (!File.Exists(RLDataFile))
            {
                Log("No RL data found. Run self-play first.", Color.FromArgb(200, 60, 60));
                OnComplete?.Invoke();
                return;
            }

            Log("Tokenizing RL data...", Color.FromArgb(100, 106, 118));
            string rlText = File.ReadAllText(RLDataFile);
            // Oversample 3x — corrected games are high value
            string tripled = rlText + "\n" + rlText + "\n" + rlText;
            int[] enc = tokenizer.Encode(tripled);
            byte[] tokens = new byte[enc.Length];
            for (int i = 0; i < enc.Length; i++) tokens[i] = (byte)enc[i];
            int tokenCount = tokens.Length;
            Log($"RL Tokens: {tokenCount:N0} ({tokenCount / 1_000_000.0:F1}M) — 3x oversampled", Color.FromArgb(100, 106, 118));

            // Load existing model (fine-tune, don't start from scratch)
            var arch = ArchConfig.Load();
            var model = new TransformerModel("chess", tokenizer.VocabSize,
                arch.ContextSize, arch.EmbedDim, arch.NumHeads, arch.FfDim, arch.NumLayers, 0.05);

            if (File.Exists(Trainer.ModelPath))
            {
                model.load(Trainer.ModelPath);
                Log("Loaded existing model for fine-tuning.", Color.FromArgb(100, 180, 255));
            }
            else
            {
                Log("No existing model — training from scratch.", Color.FromArgb(180, 140, 40));
            }

            GC.Collect(); GC.WaitForPendingFinalizers();
            model.to(device); model.train();
            Log($"Parameters: {model.parameters().Sum(p => p.numel()):N0}", Color.FromArgb(100, 106, 118));
            Log($"Fine-tuning with LR={LearningRate} for {TotalSteps} steps...",
                Color.FromArgb(0, 180, 100));

            // Lower LR for fine-tuning to not destroy supervised knowledge
            var optimizer = optim.AdamW(model.parameters(), lr: LearningRate, weight_decay: WeightDecay);
            var lossFn = nn.CrossEntropyLoss();
            var rng = new Random(42);
            float smoothedLoss = float.NaN;
            int ctx = arch.ContextSize;

            long[] inpBuf = new long[BatchSize * ctx];
            long[] tgtBuf = new long[BatchSize * ctx];

            for (int step = 1; step <= TotalSteps; step++)
            {
                if (cancel.IsCancellationRequested) break;

                float lr;
                if (step <= WarmupSteps)
                    lr = LearningRate * (float)step / WarmupSteps;
                else
                {
                    float t = (float)(step - WarmupSteps) / (TotalSteps - WarmupSteps);
                    lr = LearningRate * (0.1f + 0.9f * 0.5f * (1f + MathF.Cos(MathF.PI * t)));
                }
                foreach (var pg in optimizer.ParamGroups) pg.LearningRate = lr;

                for (int b = 0; b < BatchSize; b++)
                {
                    int off = rng.Next(0, tokenCount - ctx - 1);
                    for (int t2 = 0; t2 < ctx; t2++)
                    {
                        inpBuf[b * ctx + t2] = tokens[off + t2];
                        tgtBuf[b * ctx + t2] = tokens[off + t2 + 1];
                    }
                }

                optimizer.zero_grad();
                using (torch.NewDisposeScope())
                {
                    var it = tensor(inpBuf, dtype: ScalarType.Int64, device: device).view(BatchSize, ctx);
                    var tt = tensor(tgtBuf, dtype: ScalarType.Int64, device: device).view(BatchSize, ctx);
                    var lg = model.forward(it);
                    var ls = model.ComputeLoss(lg, tt, lossFn);

                    if (step % LogEvery == 0 || step == 1)
                    {
                        float lv = ls.item<float>();
                        smoothedLoss = float.IsNaN(smoothedLoss) ? lv : smoothedLoss * 0.95f + lv * 0.05f;
                    }
                    ls.backward();
                }
                nn.utils.clip_grad_norm_(model.parameters(), GradientClip);
                optimizer.step();

                if (step % LogEvery == 0)
                {
                    Log($"  RL step {step,6}/{TotalSteps}  loss {smoothedLoss:F4}  lr {lr:F6}",
                        Color.FromArgb(230, 232, 236));
                    OnProgress?.Invoke(step, TotalSteps, smoothedLoss);
                }

                if (step % Math.Max(100, TotalSteps / 20) == 0 || step == TotalSteps)
                {
                    model.save(Trainer.ModelPath);
                    ArchConfig.Load().Save(); // preserve arch
                    Log($"  → RL checkpoint (step {step})", Color.FromArgb(100, 180, 255));
                }
            }

            bool cancelled = cancel.IsCancellationRequested;
            if (!cancelled) model.save(Trainer.ModelPath);

            // Free GPU memory — model + optimizer hold ~3x model size on GPU
            model.Dispose();
            optimizer.Dispose();
            lossFn.Dispose();
            tokens = null!;
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Log(cancelled ? "\nRL training stopped." : "\nRL fine-tuning complete!",
                cancelled ? Color.FromArgb(180, 140, 40) : Color.FromArgb(0, 180, 100));
            OnComplete?.Invoke();
        }

        static byte[] StreamTokenize(string path, CharTokenizer tokenizer)
        {
            bool hasNl = tokenizer.CharToId.ContainsKey('\n');
            int count = 0;
            using (var r = new StreamReader(path, Encoding.UTF8, true, 1024 * 1024))
            {
                string? line;
                while ((line = r.ReadLine()) != null)
                {
                    foreach (char c in line) if (tokenizer.CharToId.ContainsKey(c)) count++;
                    if (hasNl) count++;
                }
            }
            var tokens = new byte[count];
            int idx = 0;
            using (var r = new StreamReader(path, Encoding.UTF8, true, 1024 * 1024))
            {
                string? line;
                while ((line = r.ReadLine()) != null)
                {
                    foreach (char c in line)
                        if (tokenizer.CharToId.TryGetValue(c, out int id)) tokens[idx++] = (byte)id;
                    if (hasNl && tokenizer.CharToId.TryGetValue('\n', out int nlId)) tokens[idx++] = (byte)nlId;
                }
            }
            return tokens;
        }

        void Log(string text, Color color) => OnLog?.Invoke(text, color);
    }
}
