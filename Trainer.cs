using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;
using TinyBrainBot;

namespace ChessLLM
{
    // ── Training state persisted between sessions ────────────────────────────
    public class TrainingState
    {
        public int   Step         { get; set; } = 0;
        public int   TotalSteps   { get; set; } = 0;
        public float SmoothedLoss { get; set; } = float.NaN;
        public bool  Completed    { get; set; } = false;

        public static readonly string Path = "chess_training_state.json";
        static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public void Save() => File.WriteAllText(Path, JsonSerializer.Serialize(this, Opts));

        public static TrainingState? Load()
        {
            if (!File.Exists(Path)) return null;
            try { return JsonSerializer.Deserialize<TrainingState>(File.ReadAllText(Path)); }
            catch { return null; }
        }
    }

    public class Trainer
    {
        public const string ModelPath     = "chess_model.pt";
        public const string TokenizerPath = "chess_tokenizer.bin";

        public ArchConfig Arch { get; set; } = new();

        public int   NumTrainingGames { get; set; } = 50_000;
        public int   TotalSteps       { get; set; } = 20_000;
        public int   WarmupSteps      { get; set; } = 1_000;
        public float LearningRate     { get; set; } = 3e-4f;
        public int   BatchSize        { get; set; } = 32;
        // GradAccumSteps removed — was causing overhead. Use smaller BatchSize directly if VRAM is tight.
        public float GradientClip     { get; set; } = 1.0f;
        public float WeightDecay      { get; set; } = 0.01f;
        public int   LogEvery         { get; set; } = 200;

        public Action<string, Color>? OnLog        { get; set; }
        public Action<int, int, float>? OnProgress { get; set; }
        public Action? OnComplete                  { get; set; }

        const string Vocab = "| abcdefgh12345678qrbnWLD\n";

        /// <summary>Check if there's a resumable checkpoint.</summary>
        public static (bool canResume, int step, int totalSteps, bool completed) CheckState()
        {
            var state = TrainingState.Load();
            if (state == null || !File.Exists(ModelPath) || !File.Exists(ArchConfig.Path))
                return (false, 0, 0, false);
            return (true, state.Step, state.TotalSteps, state.Completed);
        }

        public void Run(CancellationToken cancel = default)
        {
            var device = cuda.is_available() ? CUDA : CPU;
            Log($"Device: {(cuda.is_available() ? "CUDA" : "CPU")}", Color.FromArgb(100, 106, 118));

            // Tokenizer
            var tokenizer = new CharTokenizer();
            tokenizer.Build(Vocab);
            tokenizer.Save(TokenizerPath);
            Log($"Vocab size: {tokenizer.VocabSize}", Color.FromArgb(100, 106, 118));

            // Tokenize to a cache file, then memory-map it.
            // A 6GB token array can't be held as a byte[] — .NET chokes on LOH allocations that large.
            // Memory-mapping lets the OS handle paging without loading everything into managed RAM.
            const string tokenCache = "chess_tokens.bin";

            bool cacheValid = File.Exists(tokenCache)
                && File.Exists(DatasetForm.DataFile)
                && File.GetLastWriteTime(tokenCache) >= File.GetLastWriteTime(DatasetForm.DataFile);

            if (!cacheValid && File.Exists(DatasetForm.DataFile))
            {
                Log($"Tokenizing {DatasetForm.DataFile} (first time — will be cached)...", Color.FromArgb(100, 106, 118));
                StreamTokenizeToFile(DatasetForm.DataFile, tokenCache, tokenizer, OnLog);
                Log("Token cache saved.", Color.FromArgb(100, 106, 118));
            }
            else if (!cacheValid)
            {
                Log($"No dataset — generating {NumTrainingGames:N0} self-play games...", Color.FromArgb(100, 106, 118));
                string data = new DataGenerator(seed: 42).GenerateDataset(NumTrainingGames);
                int[] enc = tokenizer.Encode(data);
                File.WriteAllBytes(tokenCache, enc.Select(x => (byte)x).ToArray());
            }

            // Memory-map the token cache — OS handles paging, no 6GB byte[] in managed heap
            long tokenCount = new FileInfo(tokenCache).Length;
            Log($"Tokens: {tokenCount:N0}  ({tokenCount / 1_000_000.0:F1}M)", Color.FromArgb(100, 106, 118));

            using var mmf = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
                tokenCache, FileMode.Open, null, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
            using var accessor = mmf.CreateViewAccessor(0, tokenCount, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);

            GC.Collect(); GC.WaitForPendingFinalizers();

            // ── Check for resumable checkpoint ───────────────────────────────
            int startStep = 1;
            float smoothedLoss = float.NaN;
            var savedState = TrainingState.Load();
            bool resuming = false;

            // Build model
            var model = new TransformerModel(
                name: "chess", vocabSize: tokenizer.VocabSize,
                contextSize: Arch.ContextSize, embedDim: Arch.EmbedDim,
                numHeads: Arch.NumHeads, ffDim: Arch.FfDim,
                numLayers: Arch.NumLayers, dropout: 0.1);

            if (savedState != null && File.Exists(ModelPath))
            {
                // Check if architecture matches by trying to load
                try
                {
                    var savedArch = ArchConfig.Load();
                    bool archMatch = savedArch.EmbedDim == Arch.EmbedDim
                                  && savedArch.NumHeads == Arch.NumHeads
                                  && savedArch.FfDim == Arch.FfDim
                                  && savedArch.NumLayers == Arch.NumLayers
                                  && savedArch.ContextSize == Arch.ContextSize;

                    if (archMatch)
                    {
                        model.load(ModelPath);
                        smoothedLoss = savedState.SmoothedLoss;

                        if (savedState.Completed)
                        {
                            // Continue training a completed model — start from where it left off
                            startStep = savedState.Step + 1;
                            TotalSteps = savedState.Step + TotalSteps; // add more steps
                            Log($"Continuing completed model from step {startStep}...", Color.FromArgb(100, 180, 255));
                        }
                        else
                        {
                            // Resume interrupted training
                            startStep = savedState.Step + 1;
                            TotalSteps = savedState.TotalSteps; // keep original target
                            Log($"Resuming from step {startStep}/{TotalSteps}...", Color.FromArgb(100, 180, 255));
                        }
                        resuming = true;
                    }
                    else
                    {
                        Log("Architecture changed — starting fresh.", Color.FromArgb(180, 140, 40));
                    }
                }
                catch
                {
                    Log("Could not load checkpoint — starting fresh.", Color.FromArgb(180, 140, 40));
                }
            }

            // Clean up before GPU allocation to minimize VRAM fragmentation
            GC.Collect(); GC.WaitForPendingFinalizers();

            model.to(device);
            model.train();
            Log($"Parameters: {model.parameters().Sum(p => p.numel()):N0}", Color.FromArgb(100, 106, 118));

            if (!resuming)
                Log("Training started...", Color.FromArgb(0, 180, 100));

            var optimizer = optim.AdamW(model.parameters(), lr: LearningRate, weight_decay: WeightDecay);
            var lossFn    = nn.CrossEntropyLoss();
            var rng       = new Random(1337 + startStep);
            int ctx       = Arch.ContextSize;

            long[] inpBuf = new long[BatchSize * ctx];
            long[] tgtBuf = new long[BatchSize * ctx];

            int remainingSteps = TotalSteps - startStep + 1;
            int checkpointInterval = Math.Max(100, remainingSteps / 100);

            for (int step = startStep; step <= TotalSteps; step++)
            {
                if (cancel.IsCancellationRequested) break;

                // LR schedule
                float lr;
                if (step <= WarmupSteps)
                    lr = LearningRate * (float)step / WarmupSteps;
                else
                {
                    float t = (float)(step - WarmupSteps) / (TotalSteps - WarmupSteps);
                    lr = LearningRate * (0.1f + 0.9f * 0.5f * (1f + MathF.Cos(MathF.PI * t)));
                }
                foreach (var pg in optimizer.ParamGroups) pg.LearningRate = lr;

                // Build batch
                for (int b = 0; b < BatchSize; b++)
                {
                    long off = rng.NextInt64(0, tokenCount - ctx - 1);
                    for (int t2 = 0; t2 < ctx; t2++)
                    {
                        inpBuf[b * ctx + t2] = accessor.ReadByte(off + t2);
                        tgtBuf[b * ctx + t2] = accessor.ReadByte(off + t2 + 1);
                    }
                }

                // DisposeScope frees ALL tensors from forward/backward (including anonymous
                // intermediates from tensor().view() and model.forward() internals).
                // optimizer.step() stays OUTSIDE to protect its momentum buffers.
                optimizer.zero_grad();

                using (torch.NewDisposeScope())
                {
                    var it = tensor(inpBuf, dtype: ScalarType.Int64, device: device).view(BatchSize, ctx);
                    var tt = tensor(tgtBuf, dtype: ScalarType.Int64, device: device).view(BatchSize, ctx);
                    var lg = model.forward(it);
                    var ls = model.ComputeLoss(lg, tt, lossFn);

                    if (step % LogEvery == 0 || step == startStep)
                    {
                        float lossVal = ls.item<float>();
                        smoothedLoss = float.IsNaN(smoothedLoss) ? lossVal : smoothedLoss * 0.95f + lossVal * 0.05f;
                    }

                    ls.backward();
                }

                nn.utils.clip_grad_norm_(model.parameters(), GradientClip);
                optimizer.step();

                if (step % LogEvery == 0)
                {
                    Log($"  step {step,6}/{TotalSteps}  loss {smoothedLoss:F4}  lr {lr:F6}",
                        Color.FromArgb(230, 232, 236));
                    OnProgress?.Invoke(step, TotalSteps, smoothedLoss);
                }

                // Save checkpoint every ~1%
                if (step % checkpointInterval == 0 || step == TotalSteps)
                {
                    model.save(ModelPath);
                    Arch.Save();
                    new TrainingState
                    {
                        Step = step, TotalSteps = TotalSteps,
                        SmoothedLoss = smoothedLoss, Completed = false
                    }.Save();
                    Log($"  → checkpoint (step {step})", Color.FromArgb(100, 180, 255));
                }
            }

            bool cancelled = cancel.IsCancellationRequested;

            // Final save
            if (!cancelled)
            {
                model.save(ModelPath);
                Arch.Save();
                new TrainingState
                {
                    Step = TotalSteps, TotalSteps = TotalSteps,
                    SmoothedLoss = smoothedLoss, Completed = true
                }.Save();
            }
            else
            {
                // Save on cancel too so we can resume
                model.save(ModelPath);
                Arch.Save();
                int lastStep = Math.Min(TotalSteps,
                    (TrainingState.Load()?.Step ?? 0));  // keep whatever was last checkpointed
            }

            // Free GPU memory
            model.Dispose();
            optimizer.Dispose();
            lossFn.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Log(cancelled ? "\nTraining stopped (progress saved)." : "\nTraining complete!",
                cancelled ? Color.FromArgb(180, 140, 40) : Color.FromArgb(0, 180, 100));

            OnComplete?.Invoke();
        }

        static void StreamTokenizeToFile(string inPath, string outPath, TinyBrainBot.CharTokenizer tokenizer, Action<string, System.Drawing.Color>? log = null)
        {
            // Direct lookup table: char → byte id (255 = not in vocab)
            byte[] lookup = new byte[128];
            Array.Fill(lookup, (byte)255);
            foreach (var kv in tokenizer.CharToId)
                if (kv.Key < 128) lookup[kv.Key] = (byte)kv.Value;

            bool hasNl = tokenizer.CharToId.ContainsKey('\n');
            byte nlId = hasNl ? (byte)tokenizer.CharToId['\n'] : (byte)255;

            long totalWritten = 0;
            long fileSize = new FileInfo(inPath).Length;

            using (var fs = new FileStream(inPath, FileMode.Open, FileAccess.Read,
                FileShare.Read, 4 * 1024 * 1024, FileOptions.SequentialScan))
            using (var outFs = new FileStream(outPath, FileMode.Create, FileAccess.Write,
                FileShare.None, 4 * 1024 * 1024, FileOptions.SequentialScan))
            {
                byte[] readBuf = new byte[4 * 1024 * 1024];
                byte[] writeBuf = new byte[4 * 1024 * 1024];
                int writeIdx = 0;
                long bytesRead = 0;
                int read;

                while ((read = fs.Read(readBuf, 0, readBuf.Length)) > 0)
                {
                    bytesRead += read;
                    for (int i = 0; i < read; i++)
                    {
                        byte c = readBuf[i];
                        byte id;
                        if (c == '\n')
                        {
                            if (!hasNl) continue;
                            id = nlId;
                        }
                        else if (c < 128)
                        {
                            id = lookup[c];
                            if (id == 255) continue;
                        }
                        else continue;

                        writeBuf[writeIdx++] = id;
                        if (writeIdx == writeBuf.Length)
                        {
                            outFs.Write(writeBuf, 0, writeIdx);
                            totalWritten += writeIdx;
                            writeIdx = 0;
                        }
                    }

                    // Log progress every ~100MB
                    if (bytesRead % (100 * 1024 * 1024) < read)
                        log?.Invoke($"  Tokenizing: {bytesRead / 1048576} / {fileSize / 1048576} MB ({totalWritten + writeIdx:N0} tokens)",
                            System.Drawing.Color.FromArgb(100, 106, 118));
                }

                if (writeIdx > 0)
                {
                    outFs.Write(writeBuf, 0, writeIdx);
                    totalWritten += writeIdx;
                }
            }

            log?.Invoke($"  Tokenization complete: {totalWritten:N0} tokens written to cache.",
                System.Drawing.Color.FromArgb(0, 180, 100));

        }

        void Log(string text, Color color) => OnLog?.Invoke(text, color);
    }
}
