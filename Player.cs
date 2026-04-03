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
    /// Scores every legal move by its probability under the model, picks the best.
    /// For each candidate move: encode (history + " " + move), run one forward pass,
    /// sum up the log-probabilities of the move's characters.
    /// </summary>
    public class Player : IDisposable
    {
        public float Temperature { get; set; } = 0.5f;

        private readonly TransformerModel _model;
        private readonly CharTokenizer    _tokenizer;
        private readonly Device           _device;
        private readonly int              _contextSize;

        public Player()
        {
            (_model, _tokenizer, _device, _contextSize) = LoadModel();
        }

        public string GetMove(ChessGame game, string history)
        {
            var legalMoves = game.GetLegalMoves();
            if (legalMoves.Count == 0) return "";
            if (legalMoves.Count == 1) return legalMoves[0]; // forced move

            // Prepend board state so the model always knows where every piece is
            string boardState = BoardEncoder.Encode(game);
            string prefix = boardState + history + " ";
            string bestMove = legalMoves[0];
            float bestScore = float.NegativeInfinity;

            using (no_grad())
            {
                foreach (string uci in legalMoves)
                {
                    // DisposeScope frees all forward-pass intermediates per move scored
                    using (torch.NewDisposeScope())
                    {
                        float score = ScoreMove(prefix, uci);
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestMove = uci;
                        }
                    }
                }
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();

            return bestMove;
        }

        float ScoreMove(string prefix, string move)
        {
            string full = prefix + move;
            int[] ids = _tokenizer.Encode(full);

            // Truncate from the LEFT to fit context
            if (ids.Length > _contextSize)
                ids = ids.Skip(ids.Length - _contextSize).ToArray();

            long[] data = ids.Select(x => (long)x).ToArray();
            var input  = tensor(data, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
            var logits = _model.forward(input);

            int moveStart = ids.Length - move.Length;
            float score = 0;

            for (int i = 0; i < move.Length; i++)
            {
                int pos = moveStart + i - 1;
                if (pos < 0 || pos >= ids.Length - 1) continue;

                var posLogits = logits[0, pos];
                var logitArr = posLogits.data<float>().ToArray();

                if (!_tokenizer.CharToId.TryGetValue(move[i], out int targetId)) continue;

                // Log-softmax with temperature
                float max = logitArr.Max();
                float sumExp = logitArr.Sum(l => MathF.Exp((l - max) / Temperature));
                score += (logitArr[targetId] - max) / Temperature - MathF.Log(sumExp);
            }

            return score;
        }

        static (TransformerModel, CharTokenizer, Device, int) LoadModel()
        {
            if (!File.Exists(Trainer.ModelPath))
                throw new FileNotFoundException("No trained model. Train first.", Trainer.ModelPath);

            var arch   = ArchConfig.Load();
            var device = cuda.is_available() ? CUDA : CPU;

            var tokenizer = new CharTokenizer();
            if (!tokenizer.Load(Trainer.TokenizerPath))
                tokenizer.Build("| abcdefgh12345678qrbnWLD\n");

            var model = new TransformerModel(
                name:        "chess",
                vocabSize:   tokenizer.VocabSize,
                contextSize: arch.ContextSize,
                embedDim:    arch.EmbedDim,
                numHeads:    arch.NumHeads,
                ffDim:       arch.FfDim,
                numLayers:   arch.NumLayers,
                dropout:     0.0);

            model.load(Trainer.ModelPath);
            model.to(device);
            model.eval();

            return (model, tokenizer, device, arch.ContextSize);
        }

        public void Dispose()
        {
            _model?.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
