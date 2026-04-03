using System;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace TicTacLLM
{
    public class Player
    {
        public float Temperature { get; set; } = 0.3f;

        private readonly TransformerModel _model;
        private readonly CharTokenizer    _tokenizer;
        private readonly Device           _device;
        private readonly int              _contextSize;

        /// <exception cref="FileNotFoundException">If no trained model exists.</exception>
        public Player()
        {
            (_model, _tokenizer, _device, _contextSize) = LoadModel();
        }

        public int GetMove(TicTacToeGame game, string history)
        {
            int[] validMoves = game.GetValidMoves();

            int[] ids = _tokenizer.Encode(history);
            if (ids.Length > _contextSize)
                ids = ids.Skip(ids.Length - _contextSize).ToArray();

            long[] inputData = ids.Select(x => (long)x).ToArray();

            using (no_grad())
            using (torch.NewDisposeScope())
            {
                var inputTensor = tensor(inputData, dtype: ScalarType.Int64, device: _device).unsqueeze(0);
                var logits      = _model.forward(inputTensor);
                var lastLogits  = logits[0, ids.Length - 1];

                int vocabSize = _tokenizer.VocabSize;

                float[] masked = Enumerable.Repeat(float.NegativeInfinity, vocabSize).ToArray();
                foreach (int pos in validMoves)
                {
                    int id = _tokenizer.CharToId[(char)('0' + pos)];
                    masked[id] = lastLogits[id].item<float>() / Temperature;
                }

                float   max  = masked.Where(float.IsFinite).Max();
                float[] exps = masked.Select(l => float.IsNegativeInfinity(l) ? 0f : MathF.Exp(l - max)).ToArray();
                float   sum  = exps.Sum();
                float[] prob = exps.Select(e => e / sum).ToArray();

                double r = Random.Shared.NextDouble(), cum = 0.0;
                for (int i = 0; i < vocabSize; i++)
                {
                    cum += prob[i];
                    if (r <= cum && prob[i] > 0f)
                        return _tokenizer.IdToChar[i] - '0';
                }

                return validMoves
                    .OrderByDescending(pos => lastLogits[_tokenizer.CharToId[(char)('0' + pos)]].item<float>())
                    .First();
            }
        }

        private static (TransformerModel, CharTokenizer, Device, int) LoadModel()
        {
            if (!File.Exists(Trainer.ModelPath))
                throw new FileNotFoundException("No trained model found. Please train first.", Trainer.ModelPath);

            // Load the arch config that was saved when training completed
            var arch   = ArchConfig.Load();
            var device = cuda.is_available() ? CUDA : CPU;

            var tokenizer = new CharTokenizer();
            if (!tokenizer.Load(Trainer.TokenizerPath))
                tokenizer.Build("|012345678WLD\n");

            var model = new TransformerModel(
                name:        "ttt",
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
    }
}
