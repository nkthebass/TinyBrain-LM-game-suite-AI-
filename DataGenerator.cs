using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ChessLLM
{
    /// <summary>
    /// Generates chess games via self-play at mixed skill levels.
    ///
    /// Skill mix for diverse training data:
    ///   20% — random vs random         (teaches legal move patterns)
    ///   25% — weak greedy vs random     (basic captures)
    ///   25% — greedy vs greedy          (positional play)
    ///   20% — strong greedy vs greedy   (tactical play)
    ///   10% — strong greedy vs strong   (high-level games)
    ///
    /// "Greedy" = 1-ply material eval. "Strong" = 90% best move, 10% random.
    /// "Weak" = 60% best move, 40% random.
    ///
    /// Format: "|e2e4 e7e5 g1f3 ... W\n"
    /// </summary>
    public class DataGenerator
    {
        private readonly int _seed;

        public DataGenerator(int seed = 42) { _seed = seed; }

        public string GenerateDataset(int numGames)
        {
            var games = new string[numGames];
            Parallel.For(0, numGames, i =>
            {
                var rng = new Random(_seed ^ (i * 6364136223846793005L).GetHashCode());
                games[i] = PlayGame(i, numGames, rng);
            });
            return string.Join('\n', games);
        }

        static string PlayGame(int index, int total, Random rng)
        {
            // Determine skill level based on index distribution
            float frac = (float)(index % 20) / 20f;
            float whiteSkill, blackSkill;

            if (frac < 0.20f)      { whiteSkill = 0f;   blackSkill = 0f;   } // random vs random
            else if (frac < 0.45f) { whiteSkill = 0.6f;  blackSkill = 0f;   } // weak greedy vs random
            else if (frac < 0.70f) { whiteSkill = 0.7f;  blackSkill = 0.7f; } // greedy vs greedy
            else if (frac < 0.90f) { whiteSkill = 0.85f; blackSkill = 0.7f; } // strong vs greedy
            else                   { whiteSkill = 0.9f;  blackSkill = 0.9f; } // strong vs strong

            var game = new ChessGame();
            var seq = new StringBuilder("|", 600);
            int moveNum = 0;

            while (!game.IsGameOver() && moveNum < 150)
            {
                var moves = game.GetLegalMoves();
                if (moves.Count == 0) break;

                float skill = game.WhiteToMove ? whiteSkill : blackSkill;
                string move = PickMove(game, moves, skill, rng);

                if (moveNum > 0) seq.Append(' ');
                seq.Append(move);
                game.MakeMove(move);
                moveNum++;
            }

            string result = game.GetResult();
            if (result == "") result = "D";
            seq.Append(result);
            return seq.ToString();
        }

        /// <summary>
        /// Pick a move. skill=0 → pure random, skill=1 → always best by material eval.
        /// </summary>
        static string PickMove(ChessGame game, List<string> moves, float skill, Random rng)
        {
            if (rng.NextDouble() >= skill)
                return moves[rng.Next(moves.Count)];

            return BestByMaterial(game, moves, rng);
        }

        static string BestByMaterial(ChessGame game, List<string> moves, Random rng)
        {
            string best = moves[0];
            int bestScore = int.MinValue;

            foreach (var move in moves)
            {
                var clone = game.Clone();
                clone.MakeMove(move);
                int score = -EvalMaterial(clone);
                score += rng.Next(-3, 4); // tiny tiebreaker for variety
                if (score > bestScore) { bestScore = score; best = move; }
            }
            return best;
        }

        static int EvalMaterial(ChessGame game)
        {
            int score = 0;
            foreach (char c in game.Board)
            {
                score += c switch
                {
                    'P' =>  100, 'N' =>  320, 'B' =>  330, 'R' =>  500, 'Q' =>  900,
                    'p' => -100, 'n' => -320, 'b' => -330, 'r' => -500, 'q' => -900,
                    _ => 0
                };
            }
            return game.WhiteToMove ? score : -score;
        }
    }
}
