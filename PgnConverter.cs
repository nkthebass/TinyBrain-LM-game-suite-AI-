using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace ChessLLM
{
    /// <summary>
    /// Converts PGN game files (Standard Algebraic Notation) into the
    /// UCI training format the model expects: "|e2e4 e7e5 g1f3 ... W\n"
    /// </summary>
    public static class PgnConverter
    {
        /// <summary>
        /// Read PGN from a TextReader, convert each game to UCI training format.
        /// Calls onGame(gameString) for each successfully converted game.
        /// Returns total games converted.
        /// </summary>
        public static int Convert(TextReader reader, Action<string> onGame, int minElo = 0)
        {
            int count = 0;
            string? line;
            var headers = new Dictionary<string, string>();
            var moveText = new StringBuilder();
            bool inMoves = false;

            while ((line = reader.ReadLine()) != null)
            {
                line = line.Trim();

                if (line.StartsWith('['))
                {
                    // Header line: [Key "Value"]
                    if (inMoves && moveText.Length > 0)
                    {
                        // Process previous game
                        if (TryConvertGame(headers, moveText.ToString(), minElo, out string? result))
                        {
                            onGame(result!);
                            count++;
                        }
                        moveText.Clear();
                        headers.Clear();
                        inMoves = false;
                    }

                    var match = Regex.Match(line, @"\[(\w+)\s+""([^""]*)""\]");
                    if (match.Success)
                        headers[match.Groups[1].Value] = match.Groups[2].Value;
                }
                else if (line.Length > 0)
                {
                    inMoves = true;
                    moveText.Append(' ').Append(line);
                }
            }

            // Last game
            if (moveText.Length > 0 && TryConvertGame(headers, moveText.ToString(), minElo, out string? last))
            {
                onGame(last!);
                count++;
            }

            return count;
        }

        static bool TryConvertGame(Dictionary<string, string> headers, string moveText, int minElo, out string? result)
        {
            result = null;

            // Filter by Elo
            if (minElo > 0)
            {
                int wElo = headers.TryGetValue("WhiteElo", out var w) && int.TryParse(w, out var we) ? we : 0;
                int bElo = headers.TryGetValue("BlackElo", out var b) && int.TryParse(b, out var be) ? be : 0;
                if (wElo < minElo || bElo < minElo) return false;
            }

            // Parse result
            string gameResult;
            if (headers.TryGetValue("Result", out var res))
            {
                gameResult = res switch { "1-0" => "W", "0-1" => "L", _ => "D" };
            }
            else gameResult = "D";

            // Extract SAN moves from move text (strip move numbers, comments, variations)
            var sanMoves = ExtractSanMoves(moveText);
            if (sanMoves.Count < 4) return false; // skip very short games

            // Convert SAN to UCI
            var game = new ChessGame();
            var uciMoves = new List<string>();

            foreach (string san in sanMoves)
            {
                string? uci = SanToUci(game, san);
                if (uci == null) break;
                uciMoves.Add(uci);
                // Apply directly — skip MakeMove's GetLegalMoves validation since
                // we trust the PGN source. This is the other half of the speed fix.
                int from = ChessGame.Sq(uci[..2]);
                int to = ChessGame.Sq(uci[2..4]);
                char promo = uci.Length > 4 ? uci[4] : '\0';
                game.ApplyMoveUnchecked(from, to, promo);
            }

            if (uciMoves.Count < 4) return false;

            result = "|" + string.Join(' ', uciMoves) + gameResult;
            return true;
        }

        static List<string> ExtractSanMoves(string moveText)
        {
            var moves = new List<string>();
            // Remove comments {}, variations (), result at end
            moveText = Regex.Replace(moveText, @"\{[^}]*\}", " ");
            moveText = Regex.Replace(moveText, @"\([^)]*\)", " ");
            moveText = Regex.Replace(moveText, @"(1-0|0-1|1/2-1/2|\*)\s*$", "");

            var tokens = moveText.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var tok in tokens)
            {
                // Skip move numbers like "1." or "12..."
                if (Regex.IsMatch(tok, @"^\d+\.")) continue;
                // Skip NAGs like $1, $2
                if (tok.StartsWith('$')) continue;
                // Skip empty
                if (tok.Length < 2) continue;
                // Keep castling and regular moves
                if (tok.StartsWith("O-O") || (tok[0] >= 'A' && tok[0] <= 'Z') || (tok[0] >= 'a' && tok[0] <= 'h'))
                    moves.Add(tok);
            }
            return moves;
        }

        /// <summary>
        /// Fast SAN→UCI conversion by scanning the board directly instead of
        /// calling GetLegalMoves() (which clones the game ~30× per call).
        /// Trusts that the PGN source produced legal moves.
        /// </summary>
        public static string? SanToUci(ChessGame game, string san)
        {
            san = san.TrimEnd('+', '#', '!', '?');

            if (san is "O-O" or "0-0")
                return game.WhiteToMove ? "e1g1" : "e8g8";
            if (san is "O-O-O" or "0-0-0")
                return game.WhiteToMove ? "e1c1" : "e8c8";

            // Parse SAN components
            char pieceType = 'P';
            int destFile = -1, destRank = -1;
            int disambFile = -1, disambRank = -1;
            char promo = '\0';

            string s = san.Replace("x", "");

            int eq = s.IndexOf('=');
            if (eq >= 0) { promo = char.ToLower(s[eq + 1]); s = s[..eq]; }
            else if (s.Length >= 2 && "QRBN".Contains(s[^1]) && s[^2] >= '1' && s[^2] <= '8')
            { promo = char.ToLower(s[^1]); s = s[..^1]; }

            if (s.Length > 0 && "KQRBN".Contains(s[0])) { pieceType = s[0]; s = s[1..]; }

            if (s.Length >= 2) { destFile = s[^2] - 'a'; destRank = s[^1] - '1'; s = s[..^2]; }

            foreach (char c in s)
            {
                if (c >= 'a' && c <= 'h') disambFile = c - 'a';
                else if (c >= '1' && c <= '8') disambRank = c - '1';
            }

            if (destFile < 0 || destRank < 0) return null;
            int destSq = destFile + destRank * 8;

            // Scan the board for the matching piece (no GetLegalMoves needed)
            char piece = game.WhiteToMove ? pieceType : char.ToLower(pieceType);

            for (int sq = 0; sq < 64; sq++)
            {
                if (game.Board[sq] != piece) continue;
                if (disambFile >= 0 && sq % 8 != disambFile) continue;
                if (disambRank >= 0 && sq / 8 != disambRank) continue;
                if (!CanReach(game.Board, sq, destSq, pieceType, game.WhiteToMove, game.EpSquare))
                    continue;

                string uci = ChessGame.SqName(sq) + ChessGame.SqName(destSq);
                if (promo != '\0') uci += promo;
                return uci;
            }

            return null;
        }

        /// <summary>Can this piece type reach from→to on the current board?</summary>
        static bool CanReach(char[] board, int from, int to, char pieceType, bool white, int epSq)
        {
            int ff = from % 8, fr = from / 8;
            int tf = to % 8, tr = to / 8;
            int df = tf - ff, dr = tr - fr;

            switch (pieceType)
            {
                case 'P':
                    int dir = white ? 1 : -1;
                    if (df == 0 && dr == dir && board[to] == '.') return true;
                    if (df == 0 && dr == 2 * dir && fr == (white ? 1 : 6)
                        && board[to] == '.' && board[from + dir * 8] == '.') return true;
                    if (Math.Abs(df) == 1 && dr == dir
                        && (board[to] != '.' || to == epSq)) return true;
                    return false;

                case 'N':
                    return (Math.Abs(df) == 1 && Math.Abs(dr) == 2)
                        || (Math.Abs(df) == 2 && Math.Abs(dr) == 1);

                case 'K':
                    return Math.Abs(df) <= 1 && Math.Abs(dr) <= 1;

                case 'B':
                    return Math.Abs(df) == Math.Abs(dr) && df != 0
                        && PathClear(board, from, to, Math.Sign(df), Math.Sign(dr));

                case 'R':
                    return (df == 0 || dr == 0) && (df != 0 || dr != 0)
                        && PathClear(board, from, to, Math.Sign(df), Math.Sign(dr));

                case 'Q':
                    if (df == 0 && dr == 0) return false;
                    if (df != 0 && dr != 0 && Math.Abs(df) != Math.Abs(dr)) return false;
                    return PathClear(board, from, to, Math.Sign(df), Math.Sign(dr));

                default: return false;
            }
        }

        static bool PathClear(char[] board, int from, int to, int dFile, int dRank)
        {
            int f = from % 8 + dFile, r = from / 8 + dRank;
            int target = to % 8 + to / 8 * 8;
            while (f + r * 8 != target)
            {
                if (f < 0 || f > 7 || r < 0 || r > 7) return false;
                if (board[f + r * 8] != '.') return false;
                f += dFile; r += dRank;
            }
            return true;
        }
    }
}
