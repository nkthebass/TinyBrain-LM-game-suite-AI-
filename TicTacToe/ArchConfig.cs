using System;
using System.IO;
using System.Text.Json;

namespace TicTacLLM
{
    /// <summary>
    /// Model architecture parameters. Saved alongside the model so Player
    /// always loads with the exact architecture that was trained.
    /// </summary>
    public class ArchConfig
    {
        public int ContextSize { get; set; } = 32;
        public int EmbedDim    { get; set; } = 64;
        public int NumHeads    { get; set; } = 8;
        public int FfDim       { get; set; } = 256;
        public int NumLayers   { get; set; } = 4;

        public static readonly string Path = "ttt_arch.json";

        static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public void Save() =>
            File.WriteAllText(Path, JsonSerializer.Serialize(this, Opts));

        public static ArchConfig Load()
        {
            if (!File.Exists(Path)) return new ArchConfig();
            try
            {
                return JsonSerializer.Deserialize<ArchConfig>(File.ReadAllText(Path))
                       ?? new ArchConfig();
            }
            catch { return new ArchConfig(); }
        }
    }
}
