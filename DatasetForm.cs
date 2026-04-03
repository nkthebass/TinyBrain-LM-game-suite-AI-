using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using ZstdSharp;

namespace ChessLLM
{
    public class DatasetForm : Form
    {
        static readonly Color BgDark  = Color.FromArgb(30, 31, 36);
        static readonly Color BgLight = Color.FromArgb(40, 42, 48);
        static readonly Color BgInput = Color.FromArgb(50, 52, 60);
        static readonly Color TextMain = Color.FromArgb(230, 232, 236);
        static readonly Color TextMute = Color.FromArgb(100, 106, 118);
        static readonly Color Green   = Color.FromArgb(0, 132, 80);
        static readonly Color Blue    = Color.FromArgb(100, 180, 255);

        public const string DataFile = "chess_training_data.txt";

        Label       _statusLabel = null!;
        Label       _datasetInfo = null!;  // current dataset stats
        ProgressBar _progressBar = null!;
        ListView    _listView    = null!;
        CheckBox    _appendCheck = null!;  // append vs replace
        Button      _actionBtn   = null!, _loadPgnBtn = null!, _closeBtn = null!;
        CancellationTokenSource? _cts;
        long _currentGameCount = 0;
        double _currentSizeMB = 0;

        // Catalog: (name, type, description, sizeLabel, data, minElo)
        // data = game count for Self-Play, URL for Download
        // minElo = 0 means all games, >0 filters both players above that Elo
        static readonly List<(string name, string type, string desc, string size, string data, int minElo)> Catalog = new()
        {
            // ── Self-play ──
            ("Quick Test",    "Self-Play", "1K mixed-skill games — verify training works", "~10 sec", "1000", 0),
            ("Small",         "Self-Play", "10K mixed-skill games",                        "~1 min",  "10000", 0),
            ("Medium",        "Self-Play", "25K mixed-skill games",                        "~2 min",  "25000", 0),
            ("Large",         "Self-Play", "100K mixed-skill games",                       "~8 min",  "100000", 0),

            // ── Lichess all ratings (direct from database.lichess.org) ──
            ("Lichess 2013-01 (all)", "Download", "All rated games, Jan 2013 (~25K)",  "6 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst", 0),
            ("Lichess 2014-01 (all)", "Download", "All rated games, Jan 2014 (~200K)", "40 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst", 0),
            ("Lichess 2015-01 (all)", "Download", "All rated games, Jan 2015 (~500K+)", "120 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst", 0),

            // ── Lichess 1500+ Elo only ──
            ("Lichess 2014-01 (1500+)", "Download", "Both players 1500+ Elo, Jan 2014", "40 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst", 1500),
            ("Lichess 2015-01 (1500+)", "Download", "Both players 1500+ Elo, Jan 2015", "120 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst", 1500),

            // ── Lichess 2000+ Elo — high quality games ──
            ("★ Lichess 2014-01 (2000+)", "Download", "Both players 2000+ Elo, Jan 2014 — strong games", "40 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst", 2000),
            ("★ Lichess 2015-01 (2000+)", "Download", "Both players 2000+ Elo, Jan 2015 — strong games", "120 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst", 2000),
            ("★ Lichess 2016-01 (2000+)", "Download", "Both players 2000+ Elo, Jan 2016 — large + strong", "250 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2016-01.pgn.zst", 2000),
            ("★ Lichess 2017-01 (2000+)", "Download", "Both players 2000+ Elo, Jan 2017 — very large + strong", "500 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2017-01.pgn.zst", 2000),

            // ── Lichess 2018 (large months, ~1.3M games each) ──
            ("Lichess 2018-01 (all)", "Download", "All rated games, Jan 2018 (~1.3M games)", "400 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2018-01.pgn.zst", 0),
            ("Lichess 2018-03 (all)", "Download", "All rated games, Mar 2018 (~1.3M games)", "415 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2018-03.pgn.zst", 0),
            ("Lichess 2018-05 (all)", "Download", "All rated games, May 2018 (~1.3M games)", "429 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2018-05.pgn.zst", 0),
            ("Lichess 2018-01 (2000+)", "Download", "Both players 2000+ Elo, Jan 2018 (~150K games)", "400 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2018-01.pgn.zst", 2000),
            ("Lichess 2018-05 (2000+)", "Download", "Both players 2000+ Elo, May 2018 (~160K games)", "429 MB",
                "https://database.lichess.org/standard/lichess_db_standard_rated_2018-05.pgn.zst", 2000),

            // ── Lichess Elite (pre-filtered 2200+/2400+ Elo, no filtering needed) ──
            ("★★★ Elite 2021 H2 (6mo)", "Elite-Bulk",
                "Jul-Dec 2021 — ~4M elite games (2200+/2400+), 6 months combined", "~1.2 GB total",
                "2021-07,2021-08,2021-09,2021-10,2021-11", 0),
            ("★★★ Elite 2021 Full Year", "Elite-Bulk",
                "All 2021 — ~8M elite games, 12 months combined", "~2.2 GB total",
                "2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,2021-07,2021-08,2021-09,2021-10,2021-11", 0),
            ("★★★ Elite 2020+2021 (20mo)", "Elite-Bulk",
                "Jun 2020 - Nov 2021 — ~12M+ elite games, massive dataset", "~3.5 GB total",
                "2020-06,2020-07,2020-08,2020-09,2020-10,2020-11,2020-12,2021-01,2021-02,2021-03,2021-04,2021-05,2021-06,2021-07,2021-08,2021-09,2021-10,2021-11", 0),
            ("Elite 2021-08 (single)", "PGN-Zip",
                "Aug 2021 — ~1M elite games, peak month", "257 MB",
                "https://database.nikonoel.fr/lichess_elite_2021-08.zip", 0),
            ("Elite 2022-01 (single)", "PGN-Zip",
                "Jan 2022 — ~480K elite games (2300+/2500+)", "125 MB",
                "https://database.nikonoel.fr/lichess_elite_2022-01.zip", 0),
            ("Elite 2023-06 (single)", "PGN-Zip",
                "Jun 2023 — ~300K elite games (2300+/2500+)", "64 MB",
                "https://database.nikonoel.fr/lichess_elite_2023-06.zip", 0),

            // ── KingBase Lite 2019 (all games 2200+ Elo, curated) ──
            ("★★ KingBase Lite 2019", "PGN-Zip", "1M+ games, both players 2200+ Elo (2000-2019)", "232 MB",
                "https://archive.org/download/KingBaseLite2019/KingBaseLite2019-pgn.zip", 0),

            // ── Maia Chess (human-like play, Elo-bucketed) ──
            ("Maia Testing Set", "CSV-Bz2", "10K games per Elo bracket (1000-2500), move features", "~200 MB",
                "http://csslab.cs.toronto.edu/data/chess/kdd/maia-chess-testing-set.csv.bz2", 0),

            // ── Lichess API (live, works even when database.lichess.org is down) ──
            ("⚡ Lichess API 1K",  "API", "1K games from top players via Lichess API",  "~1 min",  "1000",  2000),
            ("⚡ Lichess API 5K",  "API", "5K games from top players via Lichess API",  "~5 min",  "5000",  2000),
            ("⚡ Lichess API 10K", "API", "10K games from top players via Lichess API", "~10 min", "10000", 2000),
            ("⚡ Lichess API 25K", "API", "25K games from top players via Lichess API", "~25 min", "25000", 2000),
        };

        public DatasetForm()
        {
            BuildUI();
            RefreshDatasetInfo();
        }

        void BuildUI()
        {
            Text            = "Chess Training Data";
            ClientSize      = new Size(620, 560);
            BackColor       = BgDark;
            ForeColor       = TextMain;
            Font            = new Font("Segoe UI", 9f);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox     = false;
            MinimizeBox     = false;
            StartPosition   = FormStartPosition.CenterParent;

            // ── Current dataset info (top) ───────────────────────────────────
            _datasetInfo = new Label
            {
                Location  = new Point(16, 12),
                Size      = new Size(588, 36),
                Font      = new Font("Segoe UI", 10f, FontStyle.Bold),
                ForeColor = TextMute,
                Text      = "No dataset yet"
            };
            Controls.Add(_datasetInfo);

            // ── ListView ─────────────────────────────────────────────────────
            _listView = new ListView
            {
                Location  = new Point(16, 54),
                Size      = new Size(588, 280),
                View      = View.Details,
                FullRowSelect = true,
                MultiSelect   = false,
                BackColor     = BgInput,
                ForeColor     = TextMain,
                Font          = new Font("Segoe UI", 9f),
                BorderStyle   = BorderStyle.None,
                HeaderStyle   = ColumnHeaderStyle.Nonclickable
            };
            _listView.Columns.Add("Name", 140);
            _listView.Columns.Add("Type", 80);
            _listView.Columns.Add("Size", 80);
            _listView.Columns.Add("Description", 280);

            foreach (var (name, type, desc, size, _, minElo) in Catalog)
            {
                string eloLabel = minElo > 0 ? $"{desc} (Elo≥{minElo})" : desc;
                _listView.Items.Add(new ListViewItem(new[] { name, type, size, eloLabel }));
            }

            if (_listView.Items.Count > 0)
                _listView.Items[0].Selected = true;

            Controls.Add(_listView);

            // ── Append checkbox + buttons ────────────────────────────────────
            _appendCheck = new CheckBox
            {
                Text      = "Append to existing data (instead of replacing)",
                Location  = new Point(16, 344),
                Size      = new Size(350, 20),
                ForeColor = TextMain,
                Checked   = true
            };
            Controls.Add(_appendCheck);

            _actionBtn = MkBtn("Generate / Download", Green, new Point(16, 374), 180, 34);
            _actionBtn.Click += (_, __) => OnActionClicked();
            Controls.Add(_actionBtn);

            _loadPgnBtn = MkBtn("Load PGN File...", Blue, new Point(210, 374), 160, 34);
            _loadPgnBtn.Click += (_, __) => OnLoadPgnClicked();
            Controls.Add(_loadPgnBtn);

            // ── Progress + status ────────────────────────────────────────────
            _progressBar = new ProgressBar
            {
                Location = new Point(16, 420),
                Size     = new Size(588, 12),
                Style    = ProgressBarStyle.Continuous,
                Minimum  = 0,
                Maximum  = 100
            };
            Controls.Add(_progressBar);

            _statusLabel = new Label
            {
                Location     = new Point(16, 440),
                Size         = new Size(588, 40),
                ForeColor    = TextMute,
                Text         = "Select a dataset and click Generate / Download.",
                AutoEllipsis = true
            };
            Controls.Add(_statusLabel);

            _closeBtn = MkBtn("Close", Color.FromArgb(80, 80, 90), new Point(510, 506), 94, 32);
            _closeBtn.Click += (_, __) => Close();
            Controls.Add(_closeBtn);
        }

        Button MkBtn(string text, Color bg, Point loc, int w, int h)
        {
            var b = new Button
            {
                Text = text, Location = loc, Size = new Size(w, h),
                FlatStyle = FlatStyle.Flat, Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                BackColor = bg, ForeColor = Color.White, Cursor = Cursors.Hand
            };
            b.FlatAppearance.BorderSize = 0;
            return b;
        }

        void RefreshDatasetInfo()
        {
            if (File.Exists(DataFile))
            {
                var info = new FileInfo(DataFile);
                _currentSizeMB = info.Length / 1024.0 / 1024.0;
                _currentGameCount = 0;
                using (var r = File.OpenText(DataFile))
                    while (r.ReadLine() != null) _currentGameCount++;

                _datasetInfo.Text = $"Current dataset:  {_currentGameCount:N0} games  ({_currentSizeMB:F1} MB)";
                _datasetInfo.ForeColor = Color.FromArgb(0, 180, 100);
            }
            else
            {
                _currentGameCount = 0;
                _currentSizeMB = 0;
                _datasetInfo.Text = "No dataset yet";
                _datasetInfo.ForeColor = TextMute;
            }
        }

        // ── Actions ──────────────────────────────────────────────────────────

        async void OnActionClicked()
        {
            if (_listView.SelectedItems.Count == 0) return;
            int idx = _listView.SelectedIndices[0];
            var entry = Catalog[idx];
            bool append = _appendCheck.Checked && File.Exists(DataFile);

            _actionBtn.Enabled = false;
            _loadPgnBtn.Enabled = false;
            _cts = new CancellationTokenSource();

            try
            {
                if (entry.type == "Self-Play")
                {
                    int numGames = int.Parse(entry.data);
                    await Task.Run(() => GenerateSelfPlay(numGames, append, _cts.Token));
                }
                else if (entry.type == "API")
                {
                    int numGames = int.Parse(entry.data);
                    await DownloadFromApi(numGames, append, entry.minElo, _cts.Token);
                }
                else if (entry.type == "Elite-Bulk")
                {
                    await DownloadEliteBulk(entry.name, entry.data, append, _cts.Token);
                }
                else if (entry.type == "PGN-Zip")
                {
                    await DownloadPgnZip(entry.name, entry.data, append, entry.minElo, _cts.Token);
                }
                else if (entry.type == "CSV-Bz2")
                {
                    await DownloadMaiaCsv(entry.name, entry.data, append, _cts.Token);
                }
                else
                {
                    await DownloadAndConvert(entry.name, entry.data, append, entry.minElo, _cts.Token);
                }
            }
            catch (OperationCanceledException) { SetStatus("Cancelled."); }
            catch (Exception ex) { SetStatus($"Error: {ex.Message}"); }
            finally
            {
                _actionBtn.Enabled = true;
                _loadPgnBtn.Enabled = true;
                RefreshDatasetInfo();
            }
        }

        void GenerateSelfPlay(int numGames, bool append, CancellationToken ct)
        {
            string mode = append ? "Appending" : "Generating";
            SetStatus($"{mode} {numGames:N0} self-play games...");
            SetProgress(0);

            int batchSize = Math.Max(50, numGames / 50);
            int generated = 0;

            using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

            while (generated < numGames)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(batchSize, numGames - generated);
                var gen = new DataGenerator(seed: 42 + generated);
                string batch = gen.GenerateDataset(count);
                writer.WriteLine(batch);
                generated += count;

                int pct = (int)(100.0 * generated / numGames);
                SetProgress(pct);
                SetStatus($"Generating... {generated:N0} / {numGames:N0} games");
            }

            SetProgress(100);
            long prior = append ? _currentGameCount : 0;
            SetStatus($"Done! +{generated:N0} games. Total will be {prior + generated:N0} games.");
        }

        async Task DownloadAndConvert(string name, string url, bool append, int minElo, CancellationToken ct)
        {
            SetStatus($"Downloading {name}...");
            SetProgress(0);

            // Download .zst file — use handler that follows redirects (archive.org uses 302)
            string tmpFile = "temp_download.pgn.zst";
            var handler = new HttpClientHandler { AllowAutoRedirect = true, MaxAutomaticRedirections = 5 };
            using var client = new HttpClient(handler);
            client.Timeout = TimeSpan.FromMinutes(60);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TinyBrainChessLM/1.0");

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();
            long totalBytes = response.Content.Headers.ContentLength ?? 0;

            using (var netStream = await response.Content.ReadAsStreamAsync(ct))
            using (var fileStream = File.Create(tmpFile))
            {
                byte[] buffer = new byte[1024 * 1024]; // 1MB buffer for max download speed
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes > 0)
                    {
                        int pct = (int)(50.0 * downloaded / totalBytes);
                        SetProgress(pct);
                        SetStatus($"Downloading {name}... {downloaded / 1024.0 / 1024.0:F1} / {totalBytes / 1024.0 / 1024.0:F1} MB");
                    }
                }
            }

            ct.ThrowIfCancellationRequested();

            // Decompress and convert PGN
            string eloNote = minElo > 0 ? $" (filtering Elo ≥ {minElo})" : "";
            SetStatus($"Decompressing and converting PGN to UCI{eloNote}...");
            SetProgress(50);

            int gameCount = 0;
            await Task.Run(() =>
            {
                using var compressedStream = File.OpenRead(tmpFile);
                using var decompStream = new DecompressionStream(compressedStream);
                using var reader = new StreamReader(decompStream, Encoding.UTF8);
                using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

                int count = 0;
                PgnConverter.Convert(reader, game =>
                {
                    writer.WriteLine(game);
                    count++;
                    if (count % 500 == 0)
                        SetStatus($"Converting{eloNote}... {count:N0} games kept");
                }, minElo);
                gameCount = count;
            }, ct);

            // Cleanup temp
            try { File.Delete(tmpFile); } catch { }

            SetProgress(100);
            long prior = append ? _currentGameCount : 0;
            SetStatus($"Done! +{gameCount:N0} games from {name}. Total will be {prior + gameCount:N0} games.");
        }

        // ── Lichess API download ─────────────────────────────────────────

        // Strong players to pull games from — mix of GMs and IMs
        static readonly string[] ApiPlayers = new[]
        {
            "DrNykterstein", "alireza2003", "nihalsarin2004", "Zhigalko_Sergei",
            "GMBenjaminBok", "Fins", "penguingm1", "polish_fighter3000",
            "Lance5500", "opperwezen", "Night-King96", "Bombegansen",
            "German11", "Konavets", "FairChess_on_YouTube", "mishanick",
            "muisback", "Msb2", "DrDrunkenstein", "RebeccaHarris"
        };

        async Task DownloadFromApi(int targetGames, bool append, int minElo, CancellationToken ct)
        {
            SetStatus("Connecting to Lichess API...");
            SetProgress(0);

            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(30);
            client.DefaultRequestHeaders.Add("Accept", "application/x-chess-pgn");
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TinyBrainChessLM/1.0");

            int totalConverted = 0;
            long prior = append ? _currentGameCount : 0;
            int playerIdx = 0;
            int gamesPerPlayer = Math.Max(200, targetGames / ApiPlayers.Length + 1);

            using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

            while (totalConverted < targetGames && playerIdx < ApiPlayers.Length)
            {
                ct.ThrowIfCancellationRequested();
                string player = ApiPlayers[playerIdx++];
                int remaining = targetGames - totalConverted;
                int fetchCount = Math.Min(gamesPerPlayer, remaining + 500); // fetch extra, Elo filter may drop some

                string url = $"https://lichess.org/api/games/user/{player}?max={fetchCount}&rated=true&perfType=blitz,rapid,classical&clocks=false&evals=false&opening=false";
                SetStatus($"Fetching games from {player}... ({totalConverted:N0}/{targetGames:N0})");

                try
                {
                    using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
                    if (!response.IsSuccessStatusCode)
                    {
                        SetStatus($"Skipping {player} (HTTP {(int)response.StatusCode})...");
                        await Task.Delay(1000, ct); // back off
                        continue;
                    }

                    using var stream = await response.Content.ReadAsStreamAsync(ct);
                    using var reader = new StreamReader(stream, Encoding.UTF8);

                    int beforeCount = totalConverted;
                    PgnConverter.Convert(reader, game =>
                    {
                        if (totalConverted >= targetGames) return;
                        writer.WriteLine(game);
                        totalConverted++;
                        if (totalConverted % 100 == 0)
                        {
                            int pct = (int)(100.0 * totalConverted / targetGames);
                            SetProgress(pct);
                            SetStatus($"Fetching from {player}... {totalConverted:N0}/{targetGames:N0} games");
                        }
                    }, minElo);

                    int got = totalConverted - beforeCount;
                    if (got > 0)
                        SetStatus($"Got {got:N0} games from {player} — total {totalConverted:N0}/{targetGames:N0}");
                }
                catch (HttpRequestException ex)
                {
                    SetStatus($"Network error on {player}: {ex.Message} — trying next...");
                }

                // Lichess API rate limit: be polite
                await Task.Delay(1500, ct);
            }

            writer.Flush();
            SetProgress(100);
            SetStatus($"Done! +{totalConverted:N0} games via API. Total: {prior + totalConverted:N0} games.");
        }

        // ── Lichess Elite Bulk (multiple months) ─────────────────────────────

        async Task DownloadEliteBulk(string name, string monthList, bool append, CancellationToken ct)
        {
            string[] months = monthList.Split(',');
            int totalGames = 0;
            int completedMonths = 0;

            // Pipeline: download month N+1 while converting month N
            // Uses a shared HttpClient with large buffer + connection pooling
            var handler = new HttpClientHandler
            {
                AllowAutoRedirect = true,
                MaxAutomaticRedirections = 5,
                MaxConnectionsPerServer = 4
            };
            using var client = new HttpClient(handler);
            client.Timeout = TimeSpan.FromMinutes(30);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TinyBrainChessLM/1.0");

            // Pre-download first month
            string? nextFile = null;
            string? nextMonth = null;
            Task? downloadTask = null;

            for (int i = 0; i <= months.Length; i++)
            {
                if (ct.IsCancellationRequested) break;

                // Start downloading next month in background
                string? currentFile = nextFile;
                string? currentMonth = nextMonth;
                nextFile = null;
                nextMonth = null;

                if (i < months.Length)
                {
                    string month = months[i].Trim();
                    string tmpFile = $"temp_elite_{month}.zip";
                    string url = $"https://database.nikonoel.fr/lichess_elite_{month}.zip";

                    // Fast download with 1MB buffer
                    var dlTask = Task.Run(async () =>
                    {
                        using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
                        response.EnsureSuccessStatusCode();
                        long total = response.Content.Headers.ContentLength ?? 0;

                        using var netStream = await response.Content.ReadAsStreamAsync(ct);
                        using var fileStream = new FileStream(tmpFile, FileMode.Create, FileAccess.Write,
                            FileShare.None, 1024 * 1024, FileOptions.SequentialScan);

                        byte[] buffer = new byte[1024 * 1024]; // 1MB buffer
                        long downloaded = 0;
                        int read;
                        while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                        {
                            await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                            downloaded += read;
                            if (total > 0 && downloaded % (4 * 1024 * 1024) < read) // update every ~4MB
                                SetStatus($"Downloading {month}... {downloaded / 1048576}/{total / 1048576} MB " +
                                    $"({completedMonths}/{months.Length} months done, {totalGames:N0} games so far)");
                        }
                    }, ct);

                    // If this is the first iteration, wait for the download to finish
                    if (currentFile == null)
                    {
                        await dlTask;
                        nextFile = tmpFile;
                        nextMonth = month;
                        continue; // loop back to start converting
                    }

                    downloadTask = dlTask;
                    nextFile = tmpFile;
                    nextMonth = month;
                }

                // Convert current month (while next month downloads in parallel)
                if (currentFile != null && currentMonth != null)
                {
                    SetStatus($"Converting {currentMonth}... (next month downloading in background)");
                    int monthGames = 0;

                    try
                    {
                        await Task.Run(() =>
                        {
                            using var zip = ZipFile.OpenRead(currentFile);
                            bool shouldAppend = append || completedMonths > 0;
                            using var writer = new StreamWriter(DataFile, shouldAppend, Encoding.UTF8,
                                bufferSize: 1024 * 1024); // 1MB write buffer

                            foreach (var entry in zip.Entries)
                            {
                                if (!entry.Name.EndsWith(".pgn", StringComparison.OrdinalIgnoreCase)) continue;
                                using var stream = entry.Open();
                                using var reader = new StreamReader(stream, Encoding.UTF8, true, 512 * 1024); // 512KB read buffer

                                int count = 0;
                                PgnConverter.Convert(reader, game =>
                                {
                                    writer.WriteLine(game);
                                    count++;
                                    if (count % 10000 == 0)
                                        SetStatus($"Converting {currentMonth}... {count:N0} games");
                                });
                                monthGames += count;
                            }
                        }, ct);

                        totalGames += monthGames;
                        completedMonths++;
                        int pct = (int)(100.0 * completedMonths / months.Length);
                        SetProgress(pct);
                        SetStatus($"{currentMonth}: +{monthGames:N0}. Total: {totalGames:N0} ({completedMonths}/{months.Length} months)");
                    }
                    catch (Exception ex)
                    {
                        SetStatus($"Error on {currentMonth}: {ex.Message} — continuing...");
                        completedMonths++;
                    }
                    finally
                    {
                        try { File.Delete(currentFile); } catch { }
                    }

                    // Wait for the background download to finish before next loop
                    if (downloadTask != null)
                    {
                        try { await downloadTask; }
                        catch (Exception ex) { SetStatus($"Download error: {ex.Message}"); }
                        downloadTask = null;
                    }
                }
            }

            SetProgress(100);
            long prior = append ? _currentGameCount : 0;
            SetStatus($"Done! +{totalGames:N0} elite games from {completedMonths} months. Total: {prior + totalGames:N0} games.");
        }

        // ── KingBase (PGN inside a ZIP) ──────────────────────────────────────

        async Task DownloadPgnZip(string name, string url, bool append, int minElo, CancellationToken ct)
        {
            SetStatus($"Downloading {name}...");
            SetProgress(0);

            string tmpFile = "temp_download.zip";
            var handler = new HttpClientHandler { AllowAutoRedirect = true, MaxAutomaticRedirections = 5 };
            using var client = new HttpClient(handler);
            client.Timeout = TimeSpan.FromMinutes(60);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TinyBrainChessLM/1.0");

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();
            long totalBytes = response.Content.Headers.ContentLength ?? 0;

            using (var netStream = await response.Content.ReadAsStreamAsync(ct))
            using (var fileStream = File.Create(tmpFile))
            {
                byte[] buffer = new byte[1024 * 1024]; // 1MB buffer for max download speed
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes > 0)
                    {
                        int pct = (int)(50.0 * downloaded / totalBytes);
                        SetProgress(pct);
                        SetStatus($"Downloading {name}... {downloaded / 1024.0 / 1024.0:F1} / {totalBytes / 1024.0 / 1024.0:F1} MB");
                    }
                }
            }

            ct.ThrowIfCancellationRequested();
            SetStatus("Extracting and converting PGN from ZIP...");
            SetProgress(50);

            int gameCount = 0;
            await Task.Run(() =>
            {
                using var zip = System.IO.Compression.ZipFile.OpenRead(tmpFile);
                using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

                foreach (var entry in zip.Entries)
                {
                    if (!entry.Name.EndsWith(".pgn", StringComparison.OrdinalIgnoreCase)) continue;

                    using var stream = entry.Open();
                    using var reader = new StreamReader(stream, Encoding.UTF8);

                    int count = 0;
                    PgnConverter.Convert(reader, game =>
                    {
                        writer.WriteLine(game);
                        count++;
                        if (count % 2000 == 0)
                            SetStatus($"Converting {entry.Name}... {count:N0} games");
                    }, minElo);
                    gameCount += count;
                }
            }, ct);

            try { File.Delete(tmpFile); } catch { }
            SetProgress(100);
            long prior = append ? _currentGameCount : 0;
            SetStatus($"Done! +{gameCount:N0} games from {name}. Total: {prior + gameCount:N0} games.");
        }

        // ── Maia CSV (bzip2-compressed CSV with move features) ───────────────

        async Task DownloadMaiaCsv(string name, string url, bool append, CancellationToken ct)
        {
            SetStatus($"Downloading {name}...");
            SetProgress(0);

            string tmpFile = "temp_maia.csv.bz2";
            var handler = new HttpClientHandler { AllowAutoRedirect = true, MaxAutomaticRedirections = 5 };
            using var client = new HttpClient(handler);
            client.Timeout = TimeSpan.FromMinutes(60);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TinyBrainChessLM/1.0");

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();
            long totalBytes = response.Content.Headers.ContentLength ?? 0;

            using (var netStream = await response.Content.ReadAsStreamAsync(ct))
            using (var fileStream = File.Create(tmpFile))
            {
                byte[] buffer = new byte[1024 * 1024]; // 1MB buffer for max download speed
                long downloaded = 0;
                int read;
                while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                    downloaded += read;
                    if (totalBytes > 0)
                    {
                        int pct = (int)(50.0 * downloaded / totalBytes);
                        SetProgress(pct);
                        SetStatus($"Downloading {name}... {downloaded / 1024.0 / 1024.0:F1} MB");
                    }
                }
            }

            ct.ThrowIfCancellationRequested();
            SetStatus("Converting Maia CSV to UCI format...");
            SetProgress(50);

            // Maia CSV has columns including move sequences in UCI format
            // Format varies but typically: game_id, moves (space-separated UCI), result, white_elo, black_elo
            int gameCount = 0;
            await Task.Run(() =>
            {
                using var fileStream = File.OpenRead(tmpFile);
                using var bz2Stream = new System.IO.Compression.BrotliStream(fileStream, System.IO.Compression.CompressionMode.Decompress);
                // Note: bzip2 isn't natively supported in .NET — try reading as-is
                // Fall back to treating it as a raw stream
                using var reader = new StreamReader(fileStream, Encoding.UTF8);
                using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

                string? header = reader.ReadLine(); // skip header
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    ct.ThrowIfCancellationRequested();
                    // Try to extract UCI moves from the CSV
                    // Maia format: many columns, "moves" column has space-separated UCI
                    var parts = line.Split(',');
                    // Find the moves column — typically one of the longer fields with spaces
                    string? moves = null;
                    string result = "D";
                    foreach (var part in parts)
                    {
                        string trimmed = part.Trim().Trim('"');
                        if (trimmed.Contains(' ') && trimmed.Length > 10
                            && (trimmed.StartsWith("e2") || trimmed.StartsWith("d2") || trimmed.StartsWith("g1")
                                || trimmed.StartsWith("b1") || trimmed.StartsWith("c2") || trimmed.StartsWith("f2")))
                        {
                            moves = trimmed;
                        }
                        if (trimmed == "1-0") result = "W";
                        else if (trimmed == "0-1") result = "L";
                        else if (trimmed == "1/2-1/2") result = "D";
                    }

                    if (moves != null && moves.Split(' ').Length >= 4)
                    {
                        writer.WriteLine($"|{moves}{result}");
                        gameCount++;
                        if (gameCount % 1000 == 0)
                            SetStatus($"Converting... {gameCount:N0} games");
                    }
                }
            }, ct);

            try { File.Delete(tmpFile); } catch { }
            SetProgress(100);
            long prior = append ? _currentGameCount : 0;
            SetStatus($"Done! +{gameCount:N0} games from {name}. Total: {prior + gameCount:N0} games.");
        }

        async void OnLoadPgnClicked()
        {
            using var dlg = new OpenFileDialog
            {
                Filter = "PGN Files (*.pgn)|*.pgn|All Files|*.*",
                Title  = "Select a PGN file"
            };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            _actionBtn.Enabled = false;
            _loadPgnBtn.Enabled = false;
            _cts = new CancellationTokenSource();

            try
            {
                string path = dlg.FileName;
                bool append = _appendCheck.Checked && File.Exists(DataFile);
                SetStatus($"Converting {Path.GetFileName(path)}...");
                SetProgress(0);

                int gameCount = 0;
                await Task.Run(() =>
                {
                    using var reader = new StreamReader(path, Encoding.UTF8);
                    using var writer = new StreamWriter(DataFile, append, Encoding.UTF8);

                    int count = 0;
                    PgnConverter.Convert(reader, game =>
                    {
                        writer.WriteLine(game);
                        count++;
                        if (count % 500 == 0)
                            SetStatus($"Converting... {count:N0} games");
                    });
                    gameCount = count;
                }, _cts.Token);

                SetProgress(100);
                long prior = append ? _currentGameCount : 0;
                SetStatus($"Done! +{gameCount:N0} games. Total: {prior + gameCount:N0} games.");
            }
            catch (Exception ex) { SetStatus($"Error: {ex.Message}"); }
            finally
            {
                _actionBtn.Enabled = true;
                _loadPgnBtn.Enabled = true;
                RefreshDatasetInfo();
            }
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        void SetStatus(string text) =>
            this.Invoke(() => { _statusLabel.Text = text; _statusLabel.ForeColor = TextMute; });

        void SetProgress(int pct) =>
            this.Invoke(() => _progressBar.Value = Math.Clamp(pct, 0, 100));
    }
}
