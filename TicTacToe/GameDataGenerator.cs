using System;
using System.Text;
using System.Threading.Tasks;

namespace TicTacLLM
{
    /// <summary>
    /// Generates tic-tac-toe game sequences for training.
    ///
    /// Format: "|" + move digits + outcome
    ///   "|41028W"  = X:4, O:1, X:0, O:2, X:8 → X wins
    ///   "|70234L"  = X:7, O:0, X:2, O:3, X:4 ... → O wins
    ///   "|012345678D" = draw (9 moves)
    ///
    /// Games are generated in parallel across all CPU cores.
    /// </summary>
    public class GameDataGenerator
    {
        private readonly int _seed;

        public GameDataGenerator(int seed = 42)
        {
            _seed = seed;
        }

        public string GenerateDataset(int numGames)
        {
            var games = new string[numGames];

            // Each game gets its own deterministic RNG — fully parallel, no locks
            Parallel.For(0, numGames, i =>
            {
                var rng = new Random(_seed ^ (i * 6364136223846793005L).GetHashCode());
                games[i] = GenerateGame(i, rng);
            });

            return string.Join('\n', games);
        }

        private static string GenerateGame(int index, Random rng)
        {
            int type = index % 5;
            bool minimaxX = type < 2;
            bool minimaxO = type >= 2 && type < 4;

            var game = new TicTacToeGame();
            var seq  = new StringBuilder("|", 14);

            while (!game.IsOver())
            {
                bool useAI = (game.CurrentPlayer == 'X') ? minimaxX : minimaxO;
                int move = useAI ? MinimaxMove(game) : RandomMove(game, rng);
                game.MakeMove(move);
                seq.Append((char)('0' + move));
            }

            var winner = game.CheckWinner();
            seq.Append(winner == 'X' ? 'W' : winner == 'O' ? 'L' : 'D');
            return seq.ToString();
        }

        private static int RandomMove(TicTacToeGame game, Random rng)
        {
            var moves = game.GetValidMoves();
            return moves[rng.Next(moves.Length)];
        }

        private static int MinimaxMove(TicTacToeGame game)
        {
            char player    = game.CurrentPlayer;
            int  bestScore = int.MinValue;
            int  bestMove  = game.GetValidMoves()[0];
            int  alpha     = int.MinValue;

            foreach (int move in game.GetValidMoves())
            {
                var clone = game.Clone();
                clone.MakeMove(move);
                int score = Minimax(clone, player, alpha, int.MaxValue);
                if (score > bestScore) { bestScore = score; bestMove = move; }
                alpha = Math.Max(alpha, bestScore);
            }
            return bestMove;
        }

        private static int Minimax(TicTacToeGame game, char maxPlayer, int alpha, int beta)
        {
            var winner = game.CheckWinner();
            if (winner != null) return winner == maxPlayer ? 1 : -1;
            if (game.IsOver()) return 0;

            if (game.CurrentPlayer == maxPlayer)
            {
                int best = int.MinValue;
                foreach (int move in game.GetValidMoves())
                {
                    var clone = game.Clone();
                    clone.MakeMove(move);
                    best  = Math.Max(best, Minimax(clone, maxPlayer, alpha, beta));
                    alpha = Math.Max(alpha, best);
                    if (beta <= alpha) break;
                }
                return best;
            }
            else
            {
                int best = int.MaxValue;
                foreach (int move in game.GetValidMoves())
                {
                    var clone = game.Clone();
                    clone.MakeMove(move);
                    best = Math.Min(best, Minimax(clone, maxPlayer, alpha, beta));
                    beta = Math.Min(beta, best);
                    if (beta <= alpha) break;
                }
                return best;
            }
        }
    }
}
