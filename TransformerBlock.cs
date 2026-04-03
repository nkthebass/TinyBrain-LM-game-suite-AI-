using TorchSharp;
using static TorchSharp.torch;

namespace TinyBrainBot
{
    /// <summary>
    /// Single transformer block using Pre-LayerNorm layout (GPT-2 style).
    ///
    ///   x → LN1 → CausalSelfAttention → + residual
    ///     → LN2 → FeedForward           → + residual
    ///
    /// Pre-LN places LayerNorm BEFORE the sublayer (inside the residual branch),
    /// which provides more stable gradients and easier training than Post-LN.
    ///
    /// Input shape:  (batch, seq_len, embedDim)
    /// Output shape: (batch, seq_len, embedDim)
    /// </summary>
    public class TransformerBlock : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> ln1;
        private readonly CausalSelfAttention attn;
        private readonly nn.Module<Tensor, Tensor> ln2;
        private readonly FeedForward ff;

        public TransformerBlock(string name, int embedDim, int numHeads, int ffDim, int maxSeqLen, double dropout = 0.0)
            : base(name)
        {
            ln1 = nn.LayerNorm(embedDim);
            attn = new CausalSelfAttention($"{name}_attn", embedDim, numHeads, maxSeqLen, dropout);
            ln2 = nn.LayerNorm(embedDim);
            ff = new FeedForward($"{name}_ff", embedDim, ffDim, dropout);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // x: (batch, seq_len, embedDim)

            // Self-attention with residual connection (Pre-LN)
            x = x + attn.call(ln1.call(x));   // (batch, seq_len, embedDim)

            // Feed-forward with residual connection (Pre-LN)
            x = x + ff.call(ln2.call(x));     // (batch, seq_len, embedDim)

            return x;
        }
    }
}
