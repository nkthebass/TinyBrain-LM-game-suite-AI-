using System.Text;

namespace ChessLLM
{
    /// <summary>
    /// Encodes the current board position using ONLY characters from the base vocab.
    /// No new tokens needed — works with any existing trained model.
    ///
    /// Format: "ke1qd1ra1rh1be1bf1nb1ng1a2b2c2d2e2f2g2h2|ke8qd8ra8rh8bc8bf8nb8ng8a7b7c7d7e7f7g7h7"
    ///
    /// Pieces: k=king, q=queen, r=rook, b=bishop, n=knight (lowercase, already in vocab).
    /// Pawns: just the square name (no prefix — saves space).
    /// White pieces then | then black pieces.
    /// </summary>
    public static class BoardEncoder
    {
        public static string Encode(ChessGame game)
        {
            var sb = new StringBuilder(120);
            AppendPieces(sb, game, true);
            sb.Append('|');
            AppendPieces(sb, game, false);
            return sb.ToString();
        }

        static void AppendPieces(StringBuilder sb, ChessGame game, bool white)
        {
            // Named pieces with lowercase prefix (k, q, r, b, n — all in base vocab)
            char[] named = white
                ? new[] { 'K', 'Q', 'R', 'B', 'N' }
                : new[] { 'k', 'q', 'r', 'b', 'n' };
            char pawn = white ? 'P' : 'p';

            foreach (char piece in named)
            {
                for (int sq = 0; sq < 64; sq++)
                {
                    if (game.Board[sq] == piece)
                    {
                        sb.Append(char.ToLower(piece));
                        sb.Append(ChessGame.SqName(sq));
                    }
                }
            }

            // Pawns — just square name, no prefix
            for (int sq = 0; sq < 64; sq++)
            {
                if (game.Board[sq] == pawn)
                    sb.Append(ChessGame.SqName(sq));
            }
        }
    }
}
