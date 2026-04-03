using TorchSharp;
using static TorchSharp.torch;

namespace TinyBrainBot
{
    /// <summary>
    /// Position-wise Feed-Forward Network.
    /// Two linear projections with GELU activation in between.
    ///   x → Linear(embedDim, ffDim) → GELU → Linear(ffDim, embedDim)
    ///
    /// Input shape:  (batch, seq_len, embedDim)
    /// Output shape: (batch, seq_len, embedDim)
    /// </summary>
    public class FeedForward : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> fc_up;    // embedDim → ffDim
        private readonly nn.Module<Tensor, Tensor> fc_down;  // ffDim → embedDim
        private readonly nn.Module<Tensor, Tensor> drop;

        public FeedForward(string name, int embedDim, int ffDim, double dropout = 0.0) : base(name)
        {
            fc_up = nn.Linear(embedDim, ffDim);
            fc_down = nn.Linear(ffDim, embedDim);
            drop = nn.Dropout(dropout);

            // Scale down the output projection init for residual stability
            // (GPT-2 style: scale by 1/sqrt(2*numLayers), applied externally or here)
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // x: (batch, seq_len, embedDim)
            var h = fc_up.call(x);        // (batch, seq_len, ffDim)
            h = nn.functional.gelu(h);    // (batch, seq_len, ffDim)
            h = drop.call(h);             // dropout after activation
            h = fc_down.call(h);          // (batch, seq_len, embedDim)
            return h;
        }
    }
}
