using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace ChessLLM
{
    /// <summary>
    /// Wraps the base Player with a minimax search layer.
    ///
    /// 1. Score all legal moves → pick top N candidates
    /// 2. For each candidate: simulate opponent's top responses
    /// 3. For each response: evaluate our resulting position
    /// 4. Pick the candidate with the best worst-case outcome
    ///
    /// All virtual games are disposable — zero context pollution.
    /// </summary>
    public class ThinkingPlayer : IDisposable
    {
        readonly TransformerModel _model;
        readonly CharTokenizer    _tokenizer;
        readonly Device           _device;
        readonly int              _contextSize;

        public float Temperature    { get; set; } = 0.5f;
        public int   NumCandidates  { get; set; } = 5;
        public int   NumResponses   { get; set; } = 10;
        public bool  ThinkingEnabled { get; set; } = true;
        public bool  BoardEncoding  { get; set; } = true;

        public ThinkingPlayer()
        {
            if (!File.Exists(Trainer.ModelPath))
                throw new FileNotFoundException("No trained model.", Trainer.ModelPath);

            var arch = ArchConfig.Load();
            _device = cuda.is_available() ? CUDA : CPU;
            _contextSize = arch.ContextSize;

            _tokenizer = new CharTokenizer();
            if (!_tokenizer.Load(Trainer.TokenizerPath))
                _tokenizer.Build("| abcdefgh12345678qrbnWLD\n");

            _model = new TransformerModel("chess", _tokenizer.VocabSize,
                arch.ContextSize, arch.EmbedDim, arch.NumHeads, arch.FfDim, arch.NumLayers, 0.0);
            _model.load(Trainer.ModelPath);
            _model.to(_device);
            _model.eval();
        }

        public string GetMove(ChessGame game, string history)
        {
            var legalMoves = game.GetLegalMoves();
            if (legalMoves.Count == 0) return "";
            if (legalMoves.Count == 1) return legalMoves[0];

            // Without thinking, just pick the best-scored move.
            // NOTE: Thinking is experimental — the model's raw instinct (trained on 12M+
            // elite games) is currently stronger than the search evaluation.
            // Enable thinking only for testing.
            if (!ThinkingEnabled)
                return PickBestMove(game, history, legalMoves);

            // ── Tactical thinking: model picks the move, search checks for blunders ──
            // The model's instinct (from 12M+ elite games) is strong.
            // The search only overrides it to: avoid losing material, find winning captures,
            // prevent walking into checkmate, and find checkmate.

            float ourMaterial = CountMaterial(game, game.WhiteToMove);

            var scored = ScoreAllMoves(game, history, legalMoves);
            var candidates = scored
                .OrderByDescending(s => s.score)
                .Take(Math.Min(NumCandidates, scored.Count))
                .ToList();

            if (candidates.Count <= 1)
                return candidates[0].move;

            float bestTotal = float.NegativeInfinity;
            string bestMove = candidates[0].move;

            foreach (var cand in candidates)
            {
                var vGame = game.Clone();
                vGame.MakeMove(cand.move);

                // Instant checkmate — always pick
                if (vGame.IsCheckmate()) return cand.move;

                // Stalemate — avoid unless no better option
                if (vGame.IsGameOver())
                {
                    if (-100f > bestTotal) { bestTotal = -100f; bestMove = cand.move; }
                    continue;
                }

                // Material after our move (did we capture something?)
                float materialAfterOurMove = CountMaterial(vGame, game.WhiteToMove);
                float materialGain = materialAfterOurMove - ourMaterial;

                // Check if opponent can punish this move
                float worstMaterialLoss = 0;
                bool opponentCanMate = false;

                var opMoves = vGame.GetLegalMoves();
                // Only check opponent's top responses (by model score) to save time
                string vHistory = history + " " + cand.move;
                var opScored = ScoreAllMoves(vGame, vHistory, opMoves);
                var topOp = opScored
                    .OrderByDescending(s => s.score)
                    .Take(Math.Min(NumResponses, opScored.Count));

                foreach (var resp in topOp)
                {
                    var aGame = vGame.Clone();
                    aGame.MakeMove(resp.move);

                    if (aGame.IsCheckmate()) { opponentCanMate = true; break; }

                    // How much material do we lose after opponent's response?
                    float materialAfterResp = CountMaterial(aGame, game.WhiteToMove);
                    float loss = materialAfterOurMove - materialAfterResp;
                    worstMaterialLoss = Math.Max(worstMaterialLoss, loss);
                }

                if (opponentCanMate)
                {
                    // This move allows checkmate — massive penalty, but don't completely veto
                    // (might be the only non-losing move)
                    float score = cand.score - 1000f;
                    if (score > bestTotal) { bestTotal = score; bestMove = cand.move; }
                    continue;
                }

                // Score: model confidence + material gain - worst material loss
                // Model confidence is the base (trust the model's instinct)
                // Material adjustments correct tactical errors
                float totalScore = cand.score * 2f          // trust model instinct heavily
                    + materialGain * 8f                      // winning material is very good
                    - worstMaterialLoss * 6f                 // losing material is bad
                    + (vGame.IsInCheck() ? 1.5f : 0f);      // giving check is a small bonus

                if (totalScore > bestTotal)
                {
                    bestTotal = totalScore;
                    bestMove = cand.move;
                }
            }

            return bestMove;
        }

        // ── Scoring helpers ──────────────────────────────────────────────────

        /// <summary>Score all legal moves by model probability. Returns (move, score) pairs.</summary>
        List<(string move, float score)> ScoreAllMoves(ChessGame game, string history, List<string> legalMoves)
        {
            string boardPrefix = BoardEncoding ? BoardEncoder.Encode(game) : "";
            string prefix = boardPrefix + history + " ";
            var results = new List<(string, float)>(legalMoves.Count);

            using (no_grad())
            {
                // Single forward pass for the prefix — get logits at last position
                int[] prefixIds = _tokenizer.Encode(prefix);
                if (prefixIds.Length > _contextSize)
                    prefixIds = prefixIds.Skip(prefixIds.Length - _contextSize).ToArray();

                float[] logitArr;
                using (torch.NewDisposeScope())
                {
                    long[] data = prefixIds.Select(x => (long)x).ToArray();
                    var input = tensor(data, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
                    var logits = _model.forward(input);
                    var lastLogits = logits[0, prefixIds.Length - 1];
                    logitArr = lastLogits.data<float>().ToArray();
                }

                // Score each move using the logits
                foreach (string uci in legalMoves)
                {
                    float score = 0;
                    for (int i = 0; i < uci.Length && i < 4; i++)
                    {
                        if (_tokenizer.CharToId.TryGetValue(uci[i], out int id) && id < logitArr.Length)
                            score += logitArr[id] * (i == 0 ? 1f : 0.3f);
                    }
                    results.Add((uci, score));
                }
            }

            return results;
        }

        /// <summary>Count material for the specified side.</summary>
        static float CountMaterial(ChessGame game, bool forWhite)
        {
            float material = 0;
            foreach (char c in game.Board)
            {
                material += c switch
                {
                    'P' => 1f, 'N' => 3.2f, 'B' => 3.3f, 'R' => 5f, 'Q' => 9f,
                    'p' => -1f, 'n' => -3.2f, 'b' => -3.3f, 'r' => -5f, 'q' => -9f,
                    _ => 0f
                };
            }
            return forWhite ? material : -material;
        }

        /// <summary>Without thinking — just pick the highest-scored move.</summary>
        string PickBestMove(ChessGame game, string history, List<string> legalMoves)
        {
            var scored = ScoreAllMoves(game, history, legalMoves);

            // Temperature sampling
            float temp = Temperature;
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

        public void Dispose()
        {
            _model?.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
