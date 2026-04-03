using System;
using System.IO;
using System.Text.Json;

namespace ChessLLM
{
    /// <summary>
    /// Persists all UI settings (model config, training params, inference params)
    /// so they survive app restarts.
    /// </summary>
    public class UISettings
    {
        // Model
        public int EmbedDim  { get; set; } = 128;
        public int NumHeads  { get; set; } = 4;
        public int FfDim     { get; set; } = 512;
        public int NumLayers { get; set; } = 6;
        public int CtxSize   { get; set; } = 512;

        // Training
        public int     Steps     { get; set; } = 20_000;
        public int     Games     { get; set; } = 50_000;
        public int     Batch     { get; set; } = 32;
        public int     Accum     { get; set; } = 2;
        public int     Warmup    { get; set; } = 1_000;
        public decimal LR        { get; set; } = 0.0003m;
        public decimal GradClip  { get; set; } = 1.0m;
        public decimal WDecay    { get; set; } = 0.01m;
        public int     LogEvery  { get; set; } = 200;

        // Inference
        public decimal Temp { get; set; } = 0.5m;

        // RL Training
        public int     StartMove    { get; set; } = 0;
        public int     SfDepth      { get; set; } = 10;
        public int     SfParallel   { get; set; } = 4;
        public int     SelfPlayGames{ get; set; } = 200;
        public int     CpThreshold  { get; set; } = 50;
        public int     RlSteps      { get; set; } = 5000;
        public decimal RlLR         { get; set; } = 0.0001m;

        // Board Prefix
        public int BpGames   { get; set; } = 5000;
        public int BpEveryN  { get; set; } = 5;

        // Thinking
        public bool ThinkEnabled { get; set; } = false;
        public int  Candidates   { get; set; } = 5;
        public int  Responses    { get; set; } = 10;

        static readonly string Path = "chess_ui_settings.json";
        static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public void Save()
        {
            try { File.WriteAllText(Path, JsonSerializer.Serialize(this, Opts)); } catch { }
        }

        public static UISettings Load()
        {
            if (!File.Exists(Path)) return new UISettings();
            try { return JsonSerializer.Deserialize<UISettings>(File.ReadAllText(Path)) ?? new UISettings(); }
            catch { return new UISettings(); }
        }
    }
}
