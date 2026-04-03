using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TinyBrainBot
{
    /// <summary>
    /// Very simple character-level tokenizer.
    /// Turns text into character IDs and back again.
    /// </summary>
    public class CharTokenizer
    {
        public Dictionary<char, int> CharToId { get; private set; } = new();
        public Dictionary<int, char> IdToChar { get; private set; } = new();

        public int VocabSize => CharToId.Count;

        public void Build(string text)
        {
            var chars = text.Distinct().OrderBy(c => c).ToList();

            CharToId.Clear();
            IdToChar.Clear();

            for (int i = 0; i < chars.Count; i++)
            {
                CharToId[chars[i]] = i;
                IdToChar[i] = chars[i];
            }
        }

        /// <summary>
        /// Build tokenizer from printable ASCII range — fixed vocab, no need to scan data.
        /// Covers all chars the CleanText pipeline can produce.
        /// </summary>
        public void BuildAscii()
        {
            CharToId.Clear();
            IdToChar.Clear();

            int id = 0;
            // Tab, newline, then printable ASCII 0x20-0x7E
            foreach (char c in new[] { '\t', '\n' })
            {
                CharToId[c] = id;
                IdToChar[id] = c;
                id++;
            }
            for (char c = (char)0x20; c <= (char)0x7E; c++)
            {
                CharToId[c] = id;
                IdToChar[id] = c;
                id++;
            }
        }

        public int[] Encode(string text)
        {
            var ids = new List<int>();

            foreach (char c in text)
            {
                if (CharToId.TryGetValue(c, out int id))
                    ids.Add(id);
            }

            return ids.ToArray();
        }

        public string Decode(IEnumerable<int> ids)
        {
            return new string(ids
                .Where(id => IdToChar.ContainsKey(id))
                .Select(id => IdToChar[id])
                .ToArray());
        }

        public void Save(string path)
        {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create));
            writer.Write(CharToId.Count);

            foreach (var kv in CharToId.OrderBy(k => k.Value))
            {
                writer.Write(kv.Key);
                writer.Write(kv.Value);
            }
        }

        public bool Load(string path)
        {
            if (!File.Exists(path))
                return false;

            using var reader = new BinaryReader(File.OpenRead(path));

            int count = reader.ReadInt32();
            CharToId.Clear();
            IdToChar.Clear();

            for (int i = 0; i < count; i++)
            {
                char ch = reader.ReadChar();
                int id = reader.ReadInt32();
                CharToId[ch] = id;
                IdToChar[id] = ch;
            }

            return true;
        }
    }
}