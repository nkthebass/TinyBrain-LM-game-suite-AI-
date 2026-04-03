using System;
using System.Linq;
using System.Text;

namespace TicTacLLM
{
    /// <summary>
    /// Tic-tac-toe board. Positions are numbered 0-8, row-major:
    ///   0 | 1 | 2
    ///  ---+---+---
    ///   3 | 4 | 5
    ///  ---+---+---
    ///   6 | 7 | 8
    /// </summary>
    public class TicTacToeGame
    {
        public char[] Board { get; private set; } = new char[9];

        public int MoveCount { get; private set; } = 0;

        // X always goes first
        public char CurrentPlayer => MoveCount % 2 == 0 ? 'X' : 'O';

        private static readonly int[][] WinLines =
        {
            new[] { 0, 1, 2 }, new[] { 3, 4, 5 }, new[] { 6, 7, 8 }, // rows
            new[] { 0, 3, 6 }, new[] { 1, 4, 7 }, new[] { 2, 5, 8 }, // cols
            new[] { 0, 4, 8 }, new[] { 2, 4, 6 }                      // diagonals
        };

        public TicTacToeGame()
        {
            Array.Fill(Board, '.');
        }

        public bool MakeMove(int pos)
        {
            if (!IsValidMove(pos)) return false;
            Board[pos] = CurrentPlayer;
            MoveCount++;
            return true;
        }

        public bool IsValidMove(int pos) =>
            pos >= 0 && pos <= 8 && Board[pos] == '.';

        public int[] GetValidMoves() =>
            Enumerable.Range(0, 9).Where(IsValidMove).ToArray();

        /// <summary>Returns the winner ('X' or 'O') or null if no winner yet.</summary>
        public char? CheckWinner()
        {
            foreach (var line in WinLines)
            {
                char a = Board[line[0]];
                if (a != '.' && a == Board[line[1]] && a == Board[line[2]])
                    return a;
            }
            return null;
        }

        public int[]? GetWinLine()
        {
            foreach (var line in WinLines)
            {
                char a = Board[line[0]];
                if (a != '.' && a == Board[line[1]] && a == Board[line[2]])
                    return line;
            }
            return null;
        }

        public bool IsDraw() => MoveCount == 9 && CheckWinner() == null;

        public bool IsOver() => CheckWinner() != null || MoveCount == 9;

        public TicTacToeGame Clone()
        {
            var g = new TicTacToeGame();
            Array.Copy(Board, g.Board, 9);
            g.MoveCount = MoveCount;
            return g;
        }

        /// <summary>
        /// Prints the board. Empty squares show their position number.
        ///   X | 1 | 2
        ///  ---+---+---
        ///   3 | O | 5
        ///  ---+---+---
        ///   6 | 7 | 8
        /// </summary>
        public string DisplayBoard()
        {
            var sb = new StringBuilder();
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    int pos = row * 3 + col;
                    char cell = Board[pos] == '.' ? (char)('0' + pos) : Board[pos];
                    sb.Append($" {cell} ");
                    if (col < 2) sb.Append('|');
                }
                if (row < 2) sb.AppendLine("\n---+---+---");
            }
            return sb.ToString();
        }
    }
}
