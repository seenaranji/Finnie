"""Step 4 demo: fixed-size chunking at sizes 200 / 500 / 1000, with and without overlap.

Run:  .venv/bin/python -m scripts.step4_chunk_demo
Prints the table, then EVERY chunk of one article at each size so you can read them.
"""
import json
import re
from datetime import datetime
from pathlib import Path

from src.rag.chunker import chunk_fixed
from src.rag.loader import load_documents

SIZES = [200, 500, 1000]
OVERLAP = 50   # from config.yaml; a first guess, compared properly in Step 8
docs = load_documents()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def stats(size: int, overlap: int) -> dict:
    all_chunks, cuts, whole, total_sent = [], 0, 0, 0
    for d in docs:
        ch = chunk_fixed(d, size, overlap)
        all_chunks += ch
        for c in ch[:-1]:   # does the chunk END in the middle of a sentence?
            nxt = d.text[c.end:c.end + 1]
            if not (c.text.rstrip()[-1:] in ".!?" or nxt in ("\n", "")):
                cuts += 1
        for s in sentences(d.text):   # does each sentence appear WHOLE in some chunk?
            total_sent += 1
            whole += any(s in c.text for c in ch)
    lens = [len(c.text) for c in all_chunks]
    return {"size": size, "overlap": overlap, "chunks": len(all_chunks),
            "avg_len": round(sum(lens) / len(lens), 1), "min_len": min(lens), "max_len": max(lens),
            "chunks_ending_mid_sentence": cuts,
            "pct_ending_mid_sentence": round(100 * cuts / len(all_chunks), 1),
            "pct_sentences_whole": round(100 * whole / total_sent, 1)}


rows = [stats(s, o) for s in SIZES for o in (OVERLAP, 0)]
print(f"total characters: {sum(len(d.text) for d in docs)} across {len(docs)} articles\n")
print(f"{'size':>5} {'ovlp':>4} {'chunks':>6} {'avg':>6} {'min':>4} {'max':>5} "
      f"{'end mid-sentence':>17} {'sentences kept whole':>21}")
for r in rows:
    print(f"{r['size']:>5} {r['overlap']:>4} {r['chunks']:>6} {r['avg_len']:>6} {r['min_len']:>4} "
          f"{r['max_len']:>5} {r['chunks_ending_mid_sentence']:>8} ({r['pct_ending_mid_sentence']:>4}%) "
          f"{r['pct_sentences_whole']:>19}%")

Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
out = Path(f"benchmarks/results/step4_chunking_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(rows, indent=2))
print(f"\nsaved: {out}")

doc = next(d for d in docs if d.id == "stocks-basics")
for size in SIZES:
    chunks = chunk_fixed(doc, size, OVERLAP)
    print(f"\n{'#' * 8} stocks-basics at size={size}, overlap={OVERLAP}: {len(chunks)} chunks {'#' * 8}")
    for c in chunks:
        print(f"\n--- {c.chunk_id}  chars {c.start}-{c.end} ({len(c.text)} long) ---\n{c.text}")
