using System;
using System.IO;
using System.Text.Json;

namespace ChessLLM
{
    public class ArchConfig
    {
        public int ContextSize { get; set; } = 512;
        public int EmbedDim    { get; set; } = 128;
        public int NumHeads    { get; set; } = 4;
        public int FfDim       { get; set; } = 512;
        public int NumLayers   { get; set; } = 6;

        public static readonly string Path = "chess_arch.json";
        static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public void Save() => File.WriteAllText(Path, JsonSerializer.Serialize(this, Opts));

        public static ArchConfig Load()
        {
            if (!File.Exists(Path)) return new ArchConfig();
            try { return JsonSerializer.Deserialize<ArchConfig>(File.ReadAllText(Path)) ?? new ArchConfig(); }
            catch { return new ArchConfig(); }
        }
    }
}
