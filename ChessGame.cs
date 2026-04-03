using System;
using System.Collections.Generic;
using System.Linq;

namespace ChessLLM
{
    /// <summary>
    /// Complete chess rules engine.
    /// Board: char[64], file-major: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
    /// Pieces: PNBRQK (white), pnbrqk (black), '.' = empty.
    /// Moves in UCI format: "e2e4", "e7e8q" (promotion).
    /// </summary>
    public class ChessGame
    {
        public char[] Board = new char[64];
        public bool WhiteToMove = true;
        public byte Castling = 0b1111;  // bits: WK=8, WQ=4, BK=2, BQ=1
        public int EpSquare = -1;       // en passant target square, or -1
        public int HalfMoves = 0;       // for 50-move rule
        public int FullMoves = 1;

        const byte WK = 8, WQ = 4, BK = 2, BQ = 1;

        static readonly int[][] KnightJumps = {
            new[]{-2,-1}, new[]{-2,1}, new[]{-1,-2}, new[]{-1,2},
            new[]{1,-2},  new[]{1,2},  new[]{2,-1},  new[]{2,1}
        };
        static readonly int[][] BishopDirs = { new[]{-1,-1}, new[]{-1,1}, new[]{1,-1}, new[]{1,1} };
        static readonly int[][] RookDirs   = { new[]{0,-1}, new[]{0,1}, new[]{-1,0}, new[]{1,0} };
        static readonly int[][] AllDirs    = {
            new[]{-1,-1}, new[]{-1,1}, new[]{1,-1}, new[]{1,1},
            new[]{0,-1},  new[]{0,1},  new[]{-1,0}, new[]{1,0}
        };

        // ── Construction ─────────────────────────────────────────────────────

        public ChessGame() { SetStart(); }

        /// <summary>Create a game from a FEN string.</summary>
        public static ChessGame FromFen(string fen)
        {
            var g = new ChessGame();
            Array.Fill(g.Board, '.');
            var parts = fen.Split(' ');
            if (parts.Length < 4) return g;

            // Board
            int sq = 56; // start at a8
            foreach (char c in parts[0])
            {
                if (c == '/') { sq -= 16; } // next rank down: go back 8, then -8 for the row
                else if (c >= '1' && c <= '8') { sq += c - '0'; }
                else { if (sq >= 0 && sq < 64) g.Board[sq] = c; sq++; }
            }

            // Active color
            g.WhiteToMove = parts[1] != "b";

            // Castling
            g.Castling = 0;
            if (parts.Length > 2)
            {
                foreach (char c in parts[2])
                {
                    if (c == 'K') g.Castling |= 8;
                    if (c == 'Q') g.Castling |= 4;
                    if (c == 'k') g.Castling |= 2;
                    if (c == 'q') g.Castling |= 1;
                }
            }

            // En passant
            g.EpSquare = parts.Length > 3 && parts[3] != "-" ? Sq(parts[3]) : -1;

            // Halfmove + fullmove
            g.HalfMoves = parts.Length > 4 && int.TryParse(parts[4], out int hm) ? hm : 0;
            g.FullMoves = parts.Length > 5 && int.TryParse(parts[5], out int fm) ? fm : 1;

            return g;
        }

        void SetStart()
        {
            Array.Fill(Board, '.');
            "RNBQKBNR".Select((c, i) => Board[i] = c).ToList();
            for (int i = 8; i < 16; i++) Board[i] = 'P';
            for (int i = 48; i < 56; i++) Board[i] = 'p';
            "rnbqkbnr".Select((c, i) => Board[56 + i] = c).ToList();
        }

        public ChessGame Clone()
        {
            var g = new ChessGame();
            Array.Copy(Board, g.Board, 64);
            g.WhiteToMove = WhiteToMove;
            g.Castling = Castling;
            g.EpSquare = EpSquare;
            g.HalfMoves = HalfMoves;
            g.FullMoves = FullMoves;
            return g;
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        public static int Sq(string s) => (s[0] - 'a') + (s[1] - '1') * 8;
        public static string SqName(int i) => $"{(char)('a' + i % 8)}{(char)('1' + i / 8)}";
        static int File(int sq) => sq % 8;
        static int Rank(int sq) => sq / 8;
        bool IsWhite(char c) => c >= 'A' && c <= 'Z';
        bool IsBlack(char c) => c >= 'a' && c <= 'z';
        public bool IsFriendly(char c) => WhiteToMove ? IsWhite(c) : IsBlack(c);
        bool IsEnemy(char c) => WhiteToMove ? IsBlack(c) : IsWhite(c);

        int FindKing(bool white)
        {
            char k = white ? 'K' : 'k';
            for (int i = 0; i < 64; i++)
                if (Board[i] == k) return i;
            return -1;
        }

        // ── Attack detection ─────────────────────────────────────────────────

        public bool IsSquareAttacked(int sq, bool byWhite)
        {
            int f = File(sq), r = Rank(sq);
            char p = byWhite ? 'P' : 'p';
            char n = byWhite ? 'N' : 'n';
            char b = byWhite ? 'B' : 'b';
            char rv = byWhite ? 'R' : 'r';
            char q = byWhite ? 'Q' : 'q';
            char k = byWhite ? 'K' : 'k';

            // Pawn attacks
            int pr = byWhite ? r - 1 : r + 1;
            if (pr >= 0 && pr <= 7)
            {
                if (f > 0 && Board[pr * 8 + f - 1] == p) return true;
                if (f < 7 && Board[pr * 8 + f + 1] == p) return true;
            }

            // Knight
            foreach (var j in KnightJumps)
            {
                int nf = f + j[0], nr = r + j[1];
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7 && Board[nf + nr * 8] == n) return true;
            }

            // King
            foreach (var d in AllDirs)
            {
                int nf = f + d[0], nr = r + d[1];
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7 && Board[nf + nr * 8] == k) return true;
            }

            // Bishop / Queen diagonals
            foreach (var d in BishopDirs)
            {
                int cf = f + d[0], cr = r + d[1];
                while (cf >= 0 && cf <= 7 && cr >= 0 && cr <= 7)
                {
                    char pc = Board[cf + cr * 8];
                    if (pc != '.') { if (pc == b || pc == q) return true; break; }
                    cf += d[0]; cr += d[1];
                }
            }

            // Rook / Queen lines
            foreach (var d in RookDirs)
            {
                int cf = f + d[0], cr = r + d[1];
                while (cf >= 0 && cf <= 7 && cr >= 0 && cr <= 7)
                {
                    char pc = Board[cf + cr * 8];
                    if (pc != '.') { if (pc == rv || pc == q) return true; break; }
                    cf += d[0]; cr += d[1];
                }
            }

            return false;
        }

        // ── Move generation ──────────────────────────────────────────────────

        List<(int from, int to, char promo)> GenPseudoLegal()
        {
            var moves = new List<(int, int, char)>(64);
            for (int sq = 0; sq < 64; sq++)
            {
                char pc = Board[sq];
                if (!IsFriendly(pc)) continue;
                switch (char.ToUpper(pc))
                {
                    case 'P': GenPawn(sq, moves); break;
                    case 'N': GenKnight(sq, moves); break;
                    case 'B': GenSliding(sq, BishopDirs, moves); break;
                    case 'R': GenSliding(sq, RookDirs, moves); break;
                    case 'Q': GenSliding(sq, AllDirs, moves); break;
                    case 'K': GenKing(sq, moves); break;
                }
            }
            return moves;
        }

        void GenPawn(int sq, List<(int, int, char)> moves)
        {
            int f = File(sq), r = Rank(sq);
            int dir = WhiteToMove ? 1 : -1;
            int startR = WhiteToMove ? 1 : 6;
            int promoR = WhiteToMove ? 7 : 0;

            // Forward
            int fwd = sq + dir * 8;
            if (fwd >= 0 && fwd < 64 && Board[fwd] == '.')
            {
                if (r + dir == promoR)
                    foreach (char pr in "qrbn") moves.Add((sq, fwd, pr));
                else
                {
                    moves.Add((sq, fwd, '\0'));
                    int fwd2 = sq + dir * 16;
                    if (r == startR && Board[fwd2] == '.')
                        moves.Add((sq, fwd2, '\0'));
                }
            }

            // Captures
            foreach (int df in new[] { -1, 1 })
            {
                int tf = f + df;
                if (tf < 0 || tf > 7) continue;
                int tr = r + dir;
                if (tr < 0 || tr > 7) continue;
                int to = tf + tr * 8;
                if (IsEnemy(Board[to]) || to == EpSquare)
                {
                    if (tr == promoR)
                        foreach (char pr in "qrbn") moves.Add((sq, to, pr));
                    else
                        moves.Add((sq, to, '\0'));
                }
            }
        }

        void GenKnight(int sq, List<(int, int, char)> moves)
        {
            int f = File(sq), r = Rank(sq);
            foreach (var j in KnightJumps)
            {
                int nf = f + j[0], nr = r + j[1];
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                {
                    int to = nf + nr * 8;
                    if (!IsFriendly(Board[to])) moves.Add((sq, to, '\0'));
                }
            }
        }

        void GenSliding(int sq, int[][] dirs, List<(int, int, char)> moves)
        {
            int f = File(sq), r = Rank(sq);
            foreach (var d in dirs)
            {
                int cf = f + d[0], cr = r + d[1];
                while (cf >= 0 && cf <= 7 && cr >= 0 && cr <= 7)
                {
                    int to = cf + cr * 8;
                    if (IsFriendly(Board[to])) break;
                    moves.Add((sq, to, '\0'));
                    if (IsEnemy(Board[to])) break;
                    cf += d[0]; cr += d[1];
                }
            }
        }

        void GenKing(int sq, List<(int, int, char)> moves)
        {
            int f = File(sq), r = Rank(sq);
            foreach (var d in AllDirs)
            {
                int nf = f + d[0], nr = r + d[1];
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                {
                    int to = nf + nr * 8;
                    if (!IsFriendly(Board[to])) moves.Add((sq, to, '\0'));
                }
            }

            // Castling
            bool enemy = !WhiteToMove;
            if (WhiteToMove)
            {
                if ((Castling & WK) != 0 && sq == 4 &&
                    Board[5] == '.' && Board[6] == '.' && Board[7] == 'R' &&
                    !IsSquareAttacked(4, enemy) && !IsSquareAttacked(5, enemy) && !IsSquareAttacked(6, enemy))
                    moves.Add((4, 6, '\0'));

                if ((Castling & WQ) != 0 && sq == 4 &&
                    Board[3] == '.' && Board[2] == '.' && Board[1] == '.' && Board[0] == 'R' &&
                    !IsSquareAttacked(4, enemy) && !IsSquareAttacked(3, enemy) && !IsSquareAttacked(2, enemy))
                    moves.Add((4, 2, '\0'));
            }
            else
            {
                if ((Castling & BK) != 0 && sq == 60 &&
                    Board[61] == '.' && Board[62] == '.' && Board[63] == 'r' &&
                    !IsSquareAttacked(60, enemy) && !IsSquareAttacked(61, enemy) && !IsSquareAttacked(62, enemy))
                    moves.Add((60, 62, '\0'));

                if ((Castling & BQ) != 0 && sq == 60 &&
                    Board[59] == '.' && Board[58] == '.' && Board[57] == '.' && Board[56] == 'r' &&
                    !IsSquareAttacked(60, enemy) && !IsSquareAttacked(59, enemy) && !IsSquareAttacked(58, enemy))
                    moves.Add((60, 58, '\0'));
            }
        }

        // ── Legal moves (pseudo-legal filtered by check) ─────────────────────

        public List<string> GetLegalMoves()
        {
            var pseudo = GenPseudoLegal();
            var legal = new List<string>(pseudo.Count);

            foreach (var (from, to, promo) in pseudo)
            {
                var clone = Clone();
                clone.ApplyMoveUnchecked(from, to, promo);

                // Was the move safe for the player who just moved?
                bool movedWhite = !clone.WhiteToMove;
                int king = clone.FindKing(movedWhite);
                if (king >= 0 && !clone.IsSquareAttacked(king, !movedWhite))
                {
                    string uci = SqName(from) + SqName(to);
                    if (promo != '\0') uci += promo;
                    legal.Add(uci);
                }
            }
            return legal;
        }

        // ── Make move (public, validates) ────────────────────────────────────

        public bool MakeMove(string uci)
        {
            if (uci.Length < 4) return false;
            int from = Sq(uci[..2]);
            int to = Sq(uci[2..4]);
            char promo = uci.Length > 4 ? uci[4] : '\0';

            // Validate
            string full = SqName(from) + SqName(to);
            if (promo != '\0') full += promo;
            if (!GetLegalMoves().Contains(full)) return false;

            ApplyMoveUnchecked(from, to, promo);
            return true;
        }

        /// <summary>Apply a move without validation. Used by GetLegalMoves and PgnConverter.</summary>
        public void ApplyMoveUnchecked(int from, int to, char promo)
        {
            char piece = Board[from];
            char captured = Board[to];
            char upper = char.ToUpper(piece);

            // Move piece
            Board[to] = piece;
            Board[from] = '.';

            // En passant capture
            if (upper == 'P' && to == EpSquare)
            {
                int capSq = WhiteToMove ? to - 8 : to + 8;
                Board[capSq] = '.';
            }

            // Promotion
            if (promo != '\0')
                Board[to] = WhiteToMove ? char.ToUpper(promo) : char.ToLower(promo);

            // Castling rook movement
            if (upper == 'K')
            {
                if (from == 4 && to == 6) { Board[5] = Board[7]; Board[7] = '.'; }
                if (from == 4 && to == 2) { Board[3] = Board[0]; Board[0] = '.'; }
                if (from == 60 && to == 62) { Board[61] = Board[63]; Board[63] = '.'; }
                if (from == 60 && to == 58) { Board[59] = Board[56]; Board[56] = '.'; }
            }

            // Update castling rights
            if (from == 4 || to == 4) Castling &= unchecked((byte)~(WK | WQ));
            if (from == 60 || to == 60) Castling &= unchecked((byte)~(BK | BQ));
            if (from == 0 || to == 0) Castling &= unchecked((byte)~WQ);
            if (from == 7 || to == 7) Castling &= unchecked((byte)~WK);
            if (from == 56 || to == 56) Castling &= unchecked((byte)~BQ);
            if (from == 63 || to == 63) Castling &= unchecked((byte)~BK);

            // Update en passant square
            EpSquare = -1;
            if (upper == 'P' && Math.Abs(Rank(to) - Rank(from)) == 2)
                EpSquare = (from + to) / 2;

            // Half-move clock
            HalfMoves = (upper == 'P' || captured != '.') ? 0 : HalfMoves + 1;

            // Full moves
            if (!WhiteToMove) FullMoves++;

            // Switch turn
            WhiteToMove = !WhiteToMove;
        }

        // ── Game state queries ───────────────────────────────────────────────

        public bool IsInCheck()
        {
            int king = FindKing(WhiteToMove);
            return king >= 0 && IsSquareAttacked(king, !WhiteToMove);
        }

        public bool IsCheckmate() => IsInCheck() && GetLegalMoves().Count == 0;
        public bool IsStalemate() => !IsInCheck() && GetLegalMoves().Count == 0;
        public bool Is50MoveRule() => HalfMoves >= 100;
        public bool IsGameOver() => GetLegalMoves().Count == 0 || HalfMoves >= 100;

        /// <summary>Returns "W" (white wins), "L" (black wins), "D" (draw), or "" (game ongoing).</summary>
        public string GetResult()
        {
            if (IsCheckmate()) return WhiteToMove ? "L" : "W";
            if (IsStalemate() || Is50MoveRule()) return "D";
            if (GetLegalMoves().Count == 0) return "D";
            return "";
        }

        // ── FEN generation (for Stockfish communication) ─────────────────────

        /// <summary>
        /// Generate FEN string for the current position.
        /// e.g. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        /// </summary>
        public string ToFen()
        {
            var sb = new System.Text.StringBuilder(80);

            // Board (rank 8 down to rank 1)
            for (int rank = 7; rank >= 0; rank--)
            {
                int empty = 0;
                for (int file = 0; file < 8; file++)
                {
                    char piece = Board[file + rank * 8];
                    if (piece == '.')
                    {
                        empty++;
                    }
                    else
                    {
                        if (empty > 0) { sb.Append(empty); empty = 0; }
                        sb.Append(piece);
                    }
                }
                if (empty > 0) sb.Append(empty);
                if (rank > 0) sb.Append('/');
            }

            // Active color
            sb.Append(WhiteToMove ? " w " : " b ");

            // Castling
            if (Castling == 0)
                sb.Append('-');
            else
            {
                if ((Castling & 8) != 0) sb.Append('K');
                if ((Castling & 4) != 0) sb.Append('Q');
                if ((Castling & 2) != 0) sb.Append('k');
                if ((Castling & 1) != 0) sb.Append('q');
            }

            // En passant
            sb.Append(EpSquare >= 0 ? $" {SqName(EpSquare)} " : " - ");

            // Halfmove clock + fullmove number
            sb.Append($"{HalfMoves} {FullMoves}");

            return sb.ToString();
        }
    }
}
