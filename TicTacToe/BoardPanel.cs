using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace TicTacLLM
{
    /// <summary>
    /// Custom-drawn tic-tac-toe board.
    /// Set Board[] and call Invalidate() to redraw.
    /// Set Clickable = true during the human's turn.
    /// </summary>
    public class BoardPanel : Panel
    {
        // Colors from the dark theme palette
        static readonly Color BgColor      = Color.FromArgb(30, 31, 36);
        static readonly Color GridColor    = Color.FromArgb(60, 65, 75);
        static readonly Color XColor       = Color.FromArgb(100, 180, 255);  // blue
        static readonly Color OColor       = Color.FromArgb(0, 180, 100);    // green
        static readonly Color WinLineColor = Color.FromArgb(230, 200, 60);   // yellow highlight
        static readonly Color HintColor    = Color.FromArgb(60, 65, 75);     // dim position numbers

        public char[] Board   { get; set; } = new char[9];
        public int[]? WinLine { get; set; }
        public bool Clickable { get; set; } = false;

        public event EventHandler<int>? CellClicked;

        public BoardPanel()
        {
            DoubleBuffered = true;
            BackColor      = BgColor;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;

            int w        = Width;
            int h        = Height;
            int cellW    = w / 3;
            int cellH    = h / 3;
            int padding  = Math.Min(cellW, cellH) / 5;

            // Grid lines
            using var gridPen = new Pen(GridColor, 3f);
            for (int i = 1; i < 3; i++)
            {
                g.DrawLine(gridPen, i * cellW, padding, i * cellW, h - padding);
                g.DrawLine(gridPen, padding, i * cellH, w - padding, i * cellH);
            }

            // Highlight winning cells
            if (WinLine != null)
            {
                using var winBrush = new SolidBrush(Color.FromArgb(30, 230, 200, 60));
                foreach (int idx in WinLine)
                {
                    int col = idx % 3, row = idx / 3;
                    g.FillRectangle(winBrush,
                        col * cellW + 2, row * cellH + 2,
                        cellW - 4, cellH - 4);
                }
            }

            // Pieces and position hints
            for (int i = 0; i < 9; i++)
            {
                int col = i % 3, row = i / 3;
                var rect = new Rectangle(
                    col * cellW + padding,
                    row * cellH + padding,
                    cellW - padding * 2,
                    cellH - padding * 2);

                if (Board[i] == 'X')
                {
                    DrawX(g, rect);
                }
                else if (Board[i] == 'O')
                {
                    DrawO(g, rect);
                }
                else
                {
                    // Dim position number as a hint
                    using var hintFont  = new Font("Segoe UI", Math.Max(8f, cellW / 5f));
                    using var hintBrush = new SolidBrush(HintColor);
                    var sf = new StringFormat
                    {
                        Alignment     = StringAlignment.Center,
                        LineAlignment = StringAlignment.Center
                    };
                    g.DrawString(i.ToString(), hintFont, hintBrush, rect, sf);
                }
            }
        }

        private void DrawX(Graphics g, Rectangle rect)
        {
            bool isWin = WinLine != null;
            var color  = isWin ? WinLineColor : XColor;
            using var pen = new Pen(color, 5f) { StartCap = LineCap.Round, EndCap = LineCap.Round };
            int m = rect.Width / 6;  // inset margin
            g.DrawLine(pen, rect.Left + m, rect.Top + m, rect.Right - m, rect.Bottom - m);
            g.DrawLine(pen, rect.Right - m, rect.Top + m, rect.Left + m, rect.Bottom - m);
        }

        private void DrawO(Graphics g, Rectangle rect)
        {
            bool isWin = WinLine != null;
            var color  = isWin ? WinLineColor : OColor;
            using var pen = new Pen(color, 5f);
            int m = rect.Width / 6;
            g.DrawEllipse(pen,
                rect.Left + m, rect.Top + m,
                rect.Width - m * 2, rect.Height - m * 2);
        }

        protected override void OnMouseClick(MouseEventArgs e)
        {
            base.OnMouseClick(e);
            if (!Clickable) return;

            int col = e.X / (Width  / 3);
            int row = e.Y / (Height / 3);
            int pos = row * 3 + col;

            if (pos >= 0 && pos < 9)
                CellClicked?.Invoke(this, pos);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            Cursor = Clickable ? Cursors.Hand : Cursors.Default;
        }
    }
}
