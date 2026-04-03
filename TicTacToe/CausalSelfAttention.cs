using TorchSharp;
using static TorchSharp.torch;

namespace TinyBrainBot
{
    /// <summary>
    /// Multi-head causal self-attention.
    /// Computes Q, K, V projections, applies scaled dot-product attention with
    /// a causal mask (tokens can only attend to previous positions), then projects
    /// the concatenated heads back to embedDim.
    ///
    /// Input shape:  (batch, seq_len, embedDim)
    /// Output shape: (batch, seq_len, embedDim)
    /// </summary>
    public class CausalSelfAttention : nn.Module<Tensor, Tensor>
    {
        private readonly int numHeads;
        private readonly int headDim;
        private readonly int embedDim;

        // Combined Q/K/V projection for efficiency: one matmul instead of three
        private readonly nn.Module<Tensor, Tensor> qkv_proj;  // embedDim → 3 * embedDim
        private readonly nn.Module<Tensor, Tensor> out_proj;  // embedDim → embedDim

        // Causal mask buffer (registered as non-parameter buffer)
        // Upper-triangular = True = positions to MASK (future tokens)
        private Tensor causalMask;
        private readonly int maxSeqLen;
        private readonly nn.Module<Tensor, Tensor> attnDrop;
        private readonly nn.Module<Tensor, Tensor> residDrop;

        public CausalSelfAttention(string name, int embedDim, int numHeads, int maxSeqLen, double dropout = 0.0)
            : base(name)
        {
            if (embedDim % numHeads != 0)
                throw new System.ArgumentException(
                    $"embedDim ({embedDim}) must be divisible by numHeads ({numHeads})");

            this.embedDim = embedDim;
            this.numHeads = numHeads;
            this.headDim = embedDim / numHeads;
            this.maxSeqLen = maxSeqLen;

            qkv_proj = nn.Linear(embedDim, 3 * embedDim);
            out_proj = nn.Linear(embedDim, embedDim);
            attnDrop = nn.Dropout(dropout);
            residDrop = nn.Dropout(dropout);

            // Build causal mask: upper triangular = True (masked positions)
            // Shape: (1, 1, maxSeqLen, maxSeqLen) for broadcasting with (B, H, T, T)
            causalMask = ones(maxSeqLen, maxSeqLen).triu(1).to(ScalarType.Bool)
                            .reshape(1, 1, maxSeqLen, maxSeqLen);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // x: (batch, seq_len, embedDim)
            long B = x.shape[0];
            long T = x.shape[1];

            // Combined QKV projection
            var qkv = qkv_proj.call(x);  // (B, T, 3 * embedDim)

            // Split into Q, K, V — each (B, T, embedDim)
            var chunks = qkv.chunk(3, dim: -1);
            var q = chunks[0];
            var k = chunks[1];
            var v = chunks[2];

            // Reshape to multi-head: (B, T, numHeads, headDim) → (B, numHeads, T, headDim)
            q = q.view(B, T, numHeads, headDim).transpose(1, 2);  // (B, H, T, D)
            k = k.view(B, T, numHeads, headDim).transpose(1, 2);  // (B, H, T, D)
            v = v.view(B, T, numHeads, headDim).transpose(1, 2);  // (B, H, T, D)

            // Scaled dot-product attention
            // attn = (Q @ K^T) / sqrt(headDim)
            float scale = 1.0f / (float)System.Math.Sqrt(headDim);
            var attn = torch.matmul(q, k.transpose(-2, -1)) * scale;  // (B, H, T, T)

            // Apply causal mask: set future positions to -inf before softmax
            // Slice mask to actual sequence length and move to same device
            var mask = causalMask.slice(2, 0, T, 1).slice(3, 0, T, 1);  // (1, 1, T, T)
            if (mask.device_type != x.device_type)
            {
                causalMask = causalMask.to(x.device);
                mask = causalMask.slice(2, 0, T, 1).slice(3, 0, T, 1);
            }
            attn = attn.masked_fill(mask, float.NegativeInfinity);

            attn = nn.functional.softmax(attn, dim: -1);  // (B, H, T, T)
            attn = attnDrop.call(attn);                    // dropout on attention weights

            // Weighted sum of values
            var out_ = torch.matmul(attn, v);  // (B, H, T, D)

            // Concatenate heads: (B, H, T, D) → (B, T, H*D) = (B, T, embedDim)
            out_ = out_.transpose(1, 2).contiguous().view(B, T, embedDim);

            // Output projection
            out_ = out_proj.call(out_);  // (B, T, embedDim)
            out_ = residDrop.call(out_); // dropout before residual add

            return out_;
        }
    }
}
