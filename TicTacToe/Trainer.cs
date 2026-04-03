using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace TicTacLLM
{
    public class Trainer
    {
        // ── File paths ───────────────────────────────────────────────────────
        public const string ModelPath     = "ttt_model.pt";
        public const string TokenizerPath = "ttt_tokenizer.bin";

        // ── Architecture (set from UI before calling Run) ────────────────────
        public ArchConfig Arch { get; set; } = new();

        // ── Training hyperparams (set from UI before calling Run) ────────────
        public int   NumTrainingGames { get; set; } = 40_000;
        public int   TotalSteps       { get; set; } = 6_000;
        public int   WarmupSteps      { get; set; } = 400;
        public float LearningRate     { get; set; } = 3e-4f;
        public int   BatchSize        { get; set; } = 32;
        public float GradientClip     { get; set; } = 1.0f;
        public float WeightDecay      { get; set; } = 0.01f;
        public int   CheckpointEvery  { get; set; } = 1_000;
        public int   LogEvery         { get; set; } = 200;

        // ── Callbacks ────────────────────────────────────────────────────────
        public Action<string, Color>? OnLog        { get; set; }
        public Action<int, int, float>? OnProgress { get; set; }
        public Action? OnComplete                  { get; set; }

        public void Run(CancellationToken cancel = default)
        {
            var device = cuda.is_available() ? CUDA : CPU;
            Log($"Device: {(cuda.is_available() ? "CUDA" : "CPU")}", Color.FromArgb(100, 106, 118));

            if (cuda.is_available())
                Log($"GPU: {cuda.device_count()} CUDA device(s) found", Color.FromArgb(100, 106, 118));

            // Tokenizer
            var tokenizer = new CharTokenizer();
            tokenizer.Build("|012345678WLD\n");
            tokenizer.Save(TokenizerPath);
            Log($"Vocab size: {tokenizer.VocabSize}", Color.FromArgb(100, 106, 118));

            // Training data (generated in parallel across all CPU cores)
            Log($"Generating {NumTrainingGames:N0} games...", Color.FromArgb(100, 106, 118));
            int[] tokens = tokenizer.Encode(
                new GameDataGenerator(seed: 42).GenerateDataset(NumTrainingGames));
            int tokenCount = tokens.Length;
            Log($"Tokens: {tokenCount:N0}", Color.FromArgb(100, 106, 118));

            // ── Move all tokens to GPU once — no per-step CPU→GPU transfer ───
            long[] tokenLongs = new long[tokenCount];
            for (int i = 0; i < tokenCount; i++) tokenLongs[i] = tokens[i];
            using var allTokens = tensor(tokenLongs, dtype: ScalarType.Int64, device: device);

            // Model
            var model = new TransformerModel(
                name:        "ttt",
                vocabSize:   tokenizer.VocabSize,
                contextSize: Arch.ContextSize,
                embedDim:    Arch.EmbedDim,
                numHeads:    Arch.NumHeads,
                ffDim:       Arch.FfDim,
                numLayers:   Arch.NumLayers,
                dropout:     0.1);

            model.to(device);
            model.train();

            Log($"Parameters: {model.parameters().Sum(p => p.numel()):N0}", Color.FromArgb(100, 106, 118));
            Log("", Color.White);
            Log("Training started...", Color.FromArgb(0, 180, 100));

            var optimizer = optim.AdamW(model.parameters(), lr: LearningRate, weight_decay: WeightDecay);
            var lossFn    = nn.CrossEntropyLoss();
            var rng       = new Random(1337);
            float smoothedLoss = float.NaN;
            int ctx = Arch.ContextSize;

            // Pre-allocate CPU index buffer (reused every step — zero GC pressure)
            long[] idxBuf = new long[BatchSize * ctx];

            for (int step = 1; step <= TotalSteps; step++)
            {
                if (cancel.IsCancellationRequested) break;

                // LR schedule: warmup → cosine decay to 10%
                float lr;
                if (step <= WarmupSteps)
                    lr = LearningRate * (float)step / WarmupSteps;
                else
                {
                    float t = (float)(step - WarmupSteps) / (TotalSteps - WarmupSteps);
                    lr = LearningRate * (0.1f + 0.9f * 0.5f * (1f + MathF.Cos(MathF.PI * t)));
                }
                foreach (var pg in optimizer.ParamGroups) pg.LearningRate = lr;

                // Build batch indices on CPU (tiny: BatchSize * ctx * 8 bytes)
                for (int b = 0; b < BatchSize; b++)
                {
                    int off = rng.Next(0, tokenCount - ctx - 1);
                    for (int t2 = 0; t2 < ctx; t2++)
                        idxBuf[b * ctx + t2] = off + t2;
                }

                optimizer.zero_grad();

                using (torch.NewDisposeScope())
                {
                    var idx = tensor(idxBuf, dtype: ScalarType.Int64, device: device);
                    var it  = allTokens.index_select(0, idx).view(BatchSize, ctx);
                    var tgt = idx + 1;
                    var tt  = allTokens.index_select(0, tgt).view(BatchSize, ctx);
                    var lg  = model.forward(it);
                    var ls  = model.ComputeLoss(lg, tt, lossFn);

                    bool shouldLog = step % LogEvery == 0 || step == 1;
                    if (shouldLog)
                    {
                        float lossVal = ls.item<float>();
                        smoothedLoss = float.IsNaN(smoothedLoss)
                            ? lossVal
                            : smoothedLoss * 0.95f + lossVal * 0.05f;
                    }
                    ls.backward();
                }

                nn.utils.clip_grad_norm_(model.parameters(), GradientClip);
                optimizer.step();

                if (step % 100 == 0)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }

                if (step % LogEvery == 0)
                {
                    Log($"  step {step,5}/{TotalSteps}  loss {smoothedLoss:F4}  lr {lr:F6}",
                        Color.FromArgb(230, 232, 236));
                    OnProgress?.Invoke(step, TotalSteps, smoothedLoss);
                }

                if (step % CheckpointEvery == 0 || step == TotalSteps)
                {
                    model.save(ModelPath);
                    Log($"  → checkpoint saved (step {step})", Color.FromArgb(100, 180, 255));
                }
            }

            bool cancelled = cancel.IsCancellationRequested;
            if (!cancelled)
            {
                model.save(ModelPath);
                Arch.Save();
            }

            Log(cancelled ? "\nTraining stopped." : "\nTraining complete!",
                cancelled ? Color.FromArgb(180, 140, 40) : Color.FromArgb(0, 180, 100));

            OnComplete?.Invoke();
        }

        void Log(string text, Color color) => OnLog?.Invoke(text, color);
    }
}
