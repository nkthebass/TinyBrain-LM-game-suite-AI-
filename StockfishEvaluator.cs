using System;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;

namespace ChessLLM
{
    /// <summary>
    /// Thread-safe UCI wrapper for a single Stockfish process.
    /// For parallel evaluation, create multiple instances.
    ///
    /// Usage:
    ///   using var sf = new StockfishEvaluator("stockfish.exe", depth: 10);
    ///   var (bestMove, score) = sf.Evaluate("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    /// </summary>
    public class StockfishEvaluator : IDisposable
    {
        readonly Process _process;
        readonly int _depth;
        readonly object _lock = new();
        bool _disposed;

        public StockfishEvaluator(string stockfishPath, int depth = 10)
        {
            _depth = depth;

            if (!File.Exists(stockfishPath))
                throw new FileNotFoundException($"Stockfish not found at: {stockfishPath}");

            _process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = stockfishPath,
                    UseShellExecute = false,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            _process.Start();

            // Initialize UCI
            Send("uci");
            WaitFor("uciok");
            Send("isready");
            WaitFor("readyok");
        }

        /// <summary>
        /// Evaluate a position. Returns (bestMove in UCI, score in centipawns from side-to-move's perspective).
        /// Mate scores are returned as ±30000.
        /// </summary>
        public (string bestMove, int scoreCp) Evaluate(string fen)
        {
            lock (_lock)
            {
                Send($"position fen {fen}");
                Send($"go depth {_depth}");

                string bestMove = "";
                int score = 0;

                while (true)
                {
                    string? line = _process.StandardOutput.ReadLine();
                    if (line == null) break;

                    // Parse score from "info depth X ... score cp Y" or "score mate Y"
                    if (line.StartsWith("info") && line.Contains($"depth {_depth}"))
                    {
                        var cpMatch = Regex.Match(line, @"score cp (-?\d+)");
                        if (cpMatch.Success)
                            score = int.Parse(cpMatch.Groups[1].Value);

                        var mateMatch = Regex.Match(line, @"score mate (-?\d+)");
                        if (mateMatch.Success)
                        {
                            int mateIn = int.Parse(mateMatch.Groups[1].Value);
                            score = mateIn > 0 ? 30000 : -30000;
                        }
                    }

                    // "bestmove e2e4" terminates the search
                    if (line.StartsWith("bestmove"))
                    {
                        var parts = line.Split(' ');
                        if (parts.Length >= 2)
                            bestMove = parts[1];
                        break;
                    }
                }

                return (bestMove, score);
            }
        }

        /// <summary>
        /// Get centipawn loss for a specific move.
        /// Returns how much worse (in centipawns) the given move is vs the best move.
        /// 0 = best move, positive = blunder, negative = somehow better (shouldn't happen).
        /// </summary>
        public int CentipawnLoss(string fen, string move)
        {
            lock (_lock)
            {
                // Evaluate best move score
                var (bestMove, bestScore) = Evaluate(fen);

                if (move == bestMove) return 0;

                // Evaluate position after the given move
                Send($"position fen {fen} moves {move}");
                Send($"go depth {_depth}");

                int afterScore = 0;
                while (true)
                {
                    string? line = _process.StandardOutput.ReadLine();
                    if (line == null) break;

                    if (line.StartsWith("info") && line.Contains($"depth {_depth}"))
                    {
                        var cpMatch = Regex.Match(line, @"score cp (-?\d+)");
                        if (cpMatch.Success) afterScore = int.Parse(cpMatch.Groups[1].Value);
                        var mateMatch = Regex.Match(line, @"score mate (-?\d+)");
                        if (mateMatch.Success) afterScore = int.Parse(mateMatch.Groups[1].Value) > 0 ? 30000 : -30000;
                    }
                    if (line.StartsWith("bestmove")) break;
                }

                // After our move, the score is from opponent's perspective, so negate
                return bestScore - (-afterScore);
            }
        }

        void Send(string command)
        {
            _process.StandardInput.WriteLine(command);
            _process.StandardInput.Flush();
        }

        void WaitFor(string expected)
        {
            while (true)
            {
                string? line = _process.StandardOutput.ReadLine();
                if (line == null || line.StartsWith(expected)) break;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try
            {
                Send("quit");
                _process.WaitForExit(3000);
                if (!_process.HasExited) _process.Kill();
                _process.Dispose();
            }
            catch { }
        }
    }
}
