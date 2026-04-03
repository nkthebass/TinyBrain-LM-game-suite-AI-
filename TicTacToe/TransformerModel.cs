using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace TinyBrainBot
{
    /// <summary>
    /// Small GPT-style causal transformer for character-level language modeling.
    ///
    /// Architecture (Pre-LN GPT):
    ///   TokenEmbedding + PositionalEmbedding
    ///   → N × TransformerBlock (LN → Attn → +res, LN → FF → +res)
    ///   → Final LayerNorm
    ///   → Linear projection to vocab logits
    ///
    /// Forward:
    ///   Input:  (batch_size, context_length)          — token IDs
    ///   Output: (batch_size, context_length, vocab_size) — logits for next-token prediction
    ///
    /// Config (~237M params with embed=1280, layers=12, heads=20, ff=5120, vocab=97):
    ///   Token embed:  97 × 1280      =    124,160
    ///   Pos embed:    256 × 1280     =    327,680
    ///   Per block:    ~19.7M × 12    = 236,105,280
    ///   Final LN:     1280 × 2       =      2,560
    ///   LM head:      1280 × 97 + 97 =    124,257
    ///   Total:        ~237M parameters
    /// </summary>
    public class TransformerModel : nn.Module<Tensor, Tensor>
    {
        private readonly int vocabSize;
        private readonly int contextSize;
        private readonly int embedDim;
        private readonly int numLayers;

        // Embeddings
        private readonly nn.Module<Tensor, Tensor> tokenEmbed;   // vocab → embedDim
        private readonly nn.Module<Tensor, Tensor> posEmbed;     // position → embedDim

        // Transformer blocks
        private readonly TransformerBlock[] blocks;

        // Final layer norm (Pre-LN architecture needs a final LN after all blocks)
        private readonly nn.Module<Tensor, Tensor> finalLN;

        // Language model head: project embedDim → vocab logits
        private readonly nn.Module<Tensor, Tensor> lmHead;

        // Embedding dropout
        private readonly nn.Module<Tensor, Tensor> embedDrop;

        // Debug flag: print tensor shapes on first forward pass
        private bool debugShapes = true;

        public TransformerModel(
            string name,
            int vocabSize,
            int contextSize = 128,
            int embedDim = 512,
            int numHeads = 8,
            int ffDim = 2048,
            int numLayers = 6,
            double dropout = 0.0) : base(name)
        {
            this.vocabSize = vocabSize;
            this.contextSize = contextSize;
            this.embedDim = embedDim;
            this.numLayers = numLayers;

            // Token embedding: maps token IDs to dense vectors
            tokenEmbed = nn.Embedding(vocabSize, embedDim);

            // Learned positional embedding: maps position index to dense vectors
            posEmbed = nn.Embedding(contextSize, embedDim);

            // Dropout on embeddings
            embedDrop = nn.Dropout(dropout);

            // Stack of transformer blocks
            blocks = new TransformerBlock[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                blocks[i] = new TransformerBlock(
                    $"block_{i}", embedDim, numHeads, ffDim, contextSize, dropout);
            }

            // Final layer norm
            finalLN = nn.LayerNorm(embedDim);

            // Output projection to vocabulary
            lmHead = nn.Linear(embedDim, vocabSize);

            // Register all submodules so parameters() enumerates them
            RegisterComponents();
            for (int i = 0; i < numLayers; i++)
                register_module($"block_{i}", blocks[i]);

            InitWeights();
        }

        /// <summary>
        /// GPT-2 style weight initialization.
        /// </summary>
        private void InitWeights()
        {
            using (no_grad())
            {
                foreach (var (paramName, param) in named_parameters())
                {
                    if (param.dim() >= 2)
                    {
                        // Xavier uniform for weight matrices
                        nn.init.xavier_uniform_(param);
                    }
                    else if (paramName.EndsWith("bias"))
                    {
                        // Zero init for biases
                        nn.init.zeros_(param);
                    }
                }

                // Scale output projections by 1/sqrt(2*numLayers) for residual stability
                foreach (var (paramName, param) in named_parameters())
                {
                    if (paramName.Contains("out_proj.weight") || paramName.Contains("fc_down.weight"))
                    {
                        param.mul_(1.0f / (float)Math.Sqrt(2.0 * numLayers));
                    }
                }
            }
        }

        public override Tensor forward(Tensor x)
        {
            // x: (batch, seq_len) — integer token IDs
            long B = x.shape[0];
            long T = x.shape[1];

            if (debugShapes)
                Console.WriteLine($"[TransformerModel] Input: ({B}, {T})");

            // Create position indices: [0, 1, 2, ..., T-1]
            var pos = arange(T, dtype: ScalarType.Int64, device: x.device)
                        .unsqueeze(0);  // (1, T) — broadcasts across batch

            // Token embedding + positional embedding
            var tokEmb = tokenEmbed.call(x);              // (B, T, embedDim)
            var posEmbedding = posEmbed.call(pos);         // (1, T, embedDim)
            var h = embedDrop.call(tokEmb + posEmbedding);   // (B, T, embedDim) — broadcast add + dropout

            if (debugShapes)
                Console.WriteLine($"[TransformerModel] After embeddings: ({h.shape[0]}, {h.shape[1]}, {h.shape[2]})");

            // Pass through transformer blocks
            for (int i = 0; i < numLayers; i++)
            {
                h = blocks[i].call(h);  // (B, T, embedDim)

                if (debugShapes && i == 0)
                    Console.WriteLine($"[TransformerModel] After block 0: ({h.shape[0]}, {h.shape[1]}, {h.shape[2]})");
            }

            // Final layer norm
            h = finalLN.call(h);  // (B, T, embedDim)

            // Project to vocab logits
            var logits = lmHead.call(h);  // (B, T, vocabSize)

            if (debugShapes)
            {
                Console.WriteLine($"[TransformerModel] Output logits: ({logits.shape[0]}, {logits.shape[1]}, {logits.shape[2]})");
                Console.WriteLine($"[TransformerModel] Total parameters: {parameters().Sum(p => p.numel()):N0}");
                debugShapes = false;  // Only print once
            }

            return logits;
        }

        /// <summary>
        /// Convenience: compute cross-entropy loss for next-token prediction.
        /// Input tokens: (B, T), Target tokens: (B, T)
        /// Reshapes logits to (B*T, V) and targets to (B*T) for CrossEntropyLoss.
        /// </summary>
        public Tensor ComputeLoss(Tensor logits, Tensor targets, nn.Module<Tensor, Tensor, Tensor> lossFn)
        {
            long B = logits.shape[0];
            long T = logits.shape[1];
            long V = logits.shape[2];

            // Flatten: (B, T, V) → (B*T, V) and (B, T) → (B*T)
            var logitsFlat = logits.view(B * T, V);
            var targetsFlat = targets.view(B * T);

            return lossFn.call(logitsFlat, targetsFlat);
        }
    }
}
