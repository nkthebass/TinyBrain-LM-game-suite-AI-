using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace ChessLLM
{
    public class ChessBoardPanel : Panel
    {
        static readonly Color LightSq  = Color.FromArgb(209, 175, 134);
        static readonly Color DarkSq   = Color.FromArgb(130, 89, 58);
        static readonly Color SelectSq = Color.FromArgb(180, 180, 60);
        static readonly Color LastFrom  = Color.FromArgb(170, 162, 80);
        static readonly Color LastTo    = Color.FromArgb(190, 182, 70);
        static readonly Color LegalDot = Color.FromArgb(100, 0, 140, 0);

        public char[] Board { get; set; } = new char[64];
        public int SelectedSquare { get; set; } = -1;
        public HashSet<int> LegalTargets { get; set; } = new();
        public int LastMoveFrom { get; set; } = -1;
        public int LastMoveTo { get; set; } = -1;
        public bool FlipBoard { get; set; } = false;
        public bool Clickable { get; set; } = false;

        public event EventHandler<int>? SquareClicked;
        public event EventHandler<int>? SquareRightClicked;

        public ChessBoardPanel()
        {
            DoubleBuffered = true;
            BackColor = Color.FromArgb(30, 31, 36);
        }

        int CellSize => Math.Min(Width, Height) / 8;

        // Convert board square (0-63) to pixel position, accounting for flip
        Point SquareToPixel(int sq)
        {
            int f = sq % 8, r = sq / 8;
            if (FlipBoard) { f = 7 - f; r = 7 - r; } else { r = 7 - r; }
            int cs = CellSize;
            int ox = (Width - cs * 8) / 2, oy = (Height - cs * 8) / 2;
            return new Point(ox + f * cs, oy + r * cs);
        }

        int PixelToSquare(int px, int py)
        {
            int cs = CellSize;
            int ox = (Width - cs * 8) / 2, oy = (Height - cs * 8) / 2;
            int fc = (px - ox) / cs, rc = (py - oy) / cs;
            if (fc < 0 || fc > 7 || rc < 0 || rc > 7) return -1;
            if (FlipBoard) { fc = 7 - fc; rc = 7 - rc; } else { rc = 7 - rc; }
            return fc + rc * 8;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
            int cs = CellSize;
            int ox = (Width - cs * 8) / 2, oy = (Height - cs * 8) / 2;

            // Draw squares
            for (int sq = 0; sq < 64; sq++)
            {
                var pt = SquareToPixel(sq);
                int f = sq % 8, r = sq / 8;
                bool light = (f + r) % 2 != 0; // a1 is dark, b1 is light, etc.

                Color color;
                if (sq == SelectedSquare) color = SelectSq;
                else if (sq == LastMoveFrom) color = LastFrom;
                else if (sq == LastMoveTo) color = LastTo;
                else color = light ? LightSq : DarkSq;

                using var brush = new SolidBrush(color);
                g.FillRectangle(brush, pt.X, pt.Y, cs, cs);

                // Legal move dots
                if (LegalTargets.Contains(sq))
                {
                    using var dotBrush = new SolidBrush(LegalDot);
                    int dotR = cs / 6;
                    g.FillEllipse(dotBrush, pt.X + cs / 2 - dotR, pt.Y + cs / 2 - dotR, dotR * 2, dotR * 2);
                }

                // Pieces
                char piece = Board[sq];
                if (piece != '.')
                {
                    string symbol = PieceUnicode(piece);
                    float fontSize = cs * 0.65f;
                    using var font = new Font("Segoe UI Symbol", fontSize, GraphicsUnit.Pixel);
                    using var pBrush = new SolidBrush(char.IsUpper(piece)
                        ? Color.FromArgb(255, 255, 245)
                        : Color.FromArgb(30, 30, 30));
                    var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };
                    g.DrawString(symbol, font, pBrush, pt.X + cs / 2f, pt.Y + cs / 2f, sf);
                }
            }

            // File/rank labels
            using var labelFont = new Font("Segoe UI", Math.Max(8, cs * 0.16f), GraphicsUnit.Pixel);
            using var labelBrush = new SolidBrush(Color.FromArgb(120, 200, 200, 200));
            for (int i = 0; i < 8; i++)
            {
                // Files (a-h) along bottom
                int displayFile = FlipBoard ? 7 - i : i;
                g.DrawString(((char)('a' + displayFile)).ToString(), labelFont, labelBrush,
                    ox + i * cs + 3, oy + cs * 8 - labelFont.Height - 1);
                // Ranks (1-8) along left
                int displayRank = FlipBoard ? i : 7 - i;
                g.DrawString((displayRank + 1).ToString(), labelFont, labelBrush,
                    ox + 2, oy + i * cs + 2);
            }
        }

        static string PieceUnicode(char c) => c switch
        {
            'K' => "\u2654", 'Q' => "\u2655", 'R' => "\u2656",
            'B' => "\u2657", 'N' => "\u2658", 'P' => "\u2659",
            'k' => "\u265A", 'q' => "\u265B", 'r' => "\u265C",
            'b' => "\u265D", 'n' => "\u265E", 'p' => "\u265F",
            _ => ""
        };

        protected override void OnMouseClick(MouseEventArgs e)
        {
            base.OnMouseClick(e);
            if (!Clickable) return;
            int sq = PixelToSquare(e.X, e.Y);
            if (sq < 0 || sq >= 64) return;

            if (e.Button == MouseButtons.Right)
                SquareRightClicked?.Invoke(this, sq);
            else
                SquareClicked?.Invoke(this, sq);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            Cursor = Clickable ? Cursors.Hand : Cursors.Default;
        }
    }
}
