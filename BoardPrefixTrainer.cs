using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace ChessLLM
{
    /// <summary>
    /// Generates board-prefix training data and fine-tunes the model on it.
    /// Takes existing game data and adds [Ke1Qd1...] position prefixes
    /// at regular intervals so the model learns to read board state.
    ///
    /// Format: "[Ke1Qd1Ra1...]e2e4 [Ke1Qd1Ra1...]e7e5 ..."
    /// The board state is inserted every N moves so the model sees
    /// position → move pairs throughout the game.
    /// </summary>
    public class BoardPrefixTrainer
    {
        public const string BPDataFile = "chess_bp_data.txt";

        public int   NumGames     { get; set; } = 5000;   // games to convert from existing data
        public int   PrefixEveryN { get; set; } = 5;       // insert board state every N moves
        public int   TotalSteps   { get; set; } = 3000;
        public int   WarmupSteps  { get; set; } = 300;
        public float LearningRate { get; set; } = 5e-5f;
        public int   BatchSize    { get; set; } = 2;
        public float GradientClip { get; set; } = 1.0f;
        public float WeightDecay  { get; set; } = 0.01f;
        public int   LogEvery     { get; set; } = 100;

        public Action<string, Color>? OnLog        { get; set; }
        public Action<int, int, float>? OnProgress { get; set; }
        public Action? OnComplete                  { get; set; }

        const string Vocab = "| abcdefgh12345678qrbnWLD\n";

        // ── Phase 1: Generate board-prefix data from existing games ──────────

        public void GeneratePrefixData(CancellationToken cancel = default)
        {
            if (!File.Exists(DatasetForm.DataFile))
            {
                Log("No training data found. Download a dataset first.", Color.FromArgb(200, 60, 60));
                OnComplete?.Invoke();
                return;
            }

            Log($"Generating board-prefix data (every {PrefixEveryN} moves)...", Color.FromArgb(100, 106, 118));

            int gamesProcessed = 0;
            using var reader = new StreamReader(DatasetForm.DataFile, Encoding.UTF8, true, 1024 * 1024);
            using var writer = new StreamWriter(BPDataFile, false, Encoding.UTF8, 1024 * 1024);

            string? line;
            while ((line = reader.ReadLine()) != null && gamesProcessed < NumGames)
            {
                if (cancel.IsCancellationRequested) break;
                if (!line.StartsWith('|')) continue;

                string? converted = ConvertGameWithPrefixes(line);
                if (converted != null)
                {
                    writer.WriteLine(converted);
                    gamesProcessed++;
                    if (gamesProcessed % 500 == 0)
                        Log($"  Prefix data: {gamesProcessed:N0}/{NumGames:N0} games",
                            Color.FromArgb(100, 106, 118));
                }
            }

            Log($"Generated {gamesProcessed:N0} board-prefix games → {BPDataFile}",
                Color.FromArgb(0, 180, 100));
        }

        string? ConvertGameWithPrefixes(string gameLine)
        {
            // Parse: "|e2e4 e7e5 ... W"
            if (gameLine.Length < 5) return null;

            char result = gameLine[^1]; // W/L/D
            string movePart = gameLine[1..^1].Trim(); // strip | and result
            string[] moves = movePart.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (moves.Length < 4) return null;

            var game = new ChessGame();
            var sb = new StringBuilder(moves.Length * 12);
            sb.Append('|');

            for (int i = 0; i < moves.Length; i++)
            {
                string uci = moves[i];

                // Insert board prefix every N moves
                if (i % PrefixEveryN == 0)
                    sb.Append(BoardEncoder.Encode(game));

                if (i > 0) sb.Append(' ');
                sb.Append(uci);

                // Apply move
                if (!game.MakeMove(uci)) break;
            }

            sb.Append(result);
            return sb.ToString();
        }

        // ── Phase 2: Fine-tune on board-prefix data ──────────────────────────

        public void RunFineTune(CancellationToken cancel = default)
        {
            if (!File.Exists(BPDataFile))
            {
                Log("No board-prefix data. Generate it first.", Color.FromArgb(200, 60, 60));
                OnComplete?.Invoke();
                return;
            }

            var device = cuda.is_available() ? CUDA : CPU;
            Log($"Device: {(cuda.is_available() ? "CUDA" : "CPU")}", Color.FromArgb(100, 106, 118));

            // Use the SAME vocab the model was trained with, loaded from the saved tokenizer.
            // If we use a different vocab size, model.load() will fail (embedding shape mismatch).
            var tokenizer = new CharTokenizer();
            if (File.Exists(Trainer.TokenizerPath))
                tokenizer.Load(Trainer.TokenizerPath);
            else
                tokenizer.Build(Vocab);

            Log($"Vocab size: {tokenizer.VocabSize}", Color.FromArgb(100, 106, 118));
            Log("Tokenizing board-prefix data...", Color.FromArgb(100, 106, 118));
            string data = File.ReadAllText(BPDataFile);
            int[] enc = tokenizer.Encode(data);
            byte[] tokens = new byte[enc.Length];
            for (int i = 0; i < enc.Length; i++) tokens[i] = (byte)enc[i];
            data = null!;
            int tokenCount = tokens.Length;
            Log($"BP Tokens: {tokenCount:N0} ({tokenCount / 1_000_000.0:F1}M)", Color.FromArgb(100, 106, 118));

            // Load existing model
            var arch = ArchConfig.Load();
            var model = new TransformerModel("chess", tokenizer.VocabSize,
                arch.ContextSize, arch.EmbedDim, arch.NumHeads, arch.FfDim, arch.NumLayers, 0.05);

            if (File.Exists(Trainer.ModelPath))
            {
                Log("Loading model weights...", Color.FromArgb(100, 106, 118));
                model.load(Trainer.ModelPath);
                Log("Model loaded. Moving to GPU...", Color.FromArgb(100, 106, 118));
            }
            else
            {
                Log("No model found — train a base model first.", Color.FromArgb(200, 60, 60));
                OnComplete?.Invoke();
                return;
            }

            GC.Collect(); GC.WaitForPendingFinalizers();
            model.to(device); model.train();
            Log($"Parameters: {model.parameters().Sum(p => p.numel()):N0}", Color.FromArgb(100, 106, 118));
            Log($"Fine-tuning with board prefixes (LR={LearningRate})...", Color.FromArgb(0, 180, 100));

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
                    Log($"  BP step {step,5}/{TotalSteps}  loss {smoothedLoss:F4}  lr {lr:F6}",
                        Color.FromArgb(230, 232, 236));
                    OnProgress?.Invoke(step, TotalSteps, smoothedLoss);
                }
            }

            bool cancelled = cancel.IsCancellationRequested;
            if (!cancelled) model.save(Trainer.ModelPath);

            model.Dispose();
            optimizer.Dispose();
            lossFn.Dispose();
            tokens = null!;
            GC.Collect(); GC.WaitForPendingFinalizers();

            Log(cancelled ? "\nBP training stopped." : "\nBoard-prefix training complete!",
                cancelled ? Color.FromArgb(180, 140, 40) : Color.FromArgb(0, 180, 100));
            OnComplete?.Invoke();
        }

        void Log(string text, Color color) => OnLog?.Invoke(text, color);
    }
}
