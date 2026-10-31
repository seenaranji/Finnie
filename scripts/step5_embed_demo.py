"""Step 5 demo: embed 6 sentences, print the cosine-similarity matrix; then time 100 real chunks.

Run:  .venv/bin/python -m scripts.step5_embed_demo
"""
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.rag.chunker import chunk_structured
from src.rag.embedder import OpenAIEmbedder, cosine_matrix
from src.rag.loader import load_documents

setup_logging("WARNING")
SENTENCES = [
    "An ETF is a fund that holds many stocks and trades on an exchange all day.",
    "Exchange-traded funds let you buy a whole basket of investments in a single trade.",
    "A mutual fund pools money from many investors and is priced once a day.",
    "A bond is a loan to a company or government that pays you interest.",
    "Margherita pizza needs fresh mozzarella, tomatoes, and basil.",
    "A basket of stocks I can trade.",
]
emb = OpenAIEmbedder()

print("##### PART 1: six sentences #####")
res = emb.embed(SENTENCES)
V = res.vectors
print(f"model={res.model} | vectors shape={V.shape} (texts x dimensions) | tokens={res.input_tokens} "
      f"| cost=${res.cost_usd:.7f} | latency={res.latency_s:.2f}s")
print(f"vector length (norm) of sentence 1: {np.linalg.norm(V[0]):.4f}")
print("first 8 numbers of sentence 1's vector:", np.round(V[0][:8], 4).tolist())

M = cosine_matrix(V)
print("\ncosine similarity matrix (rows/cols = sentence 1..6):")
print("      " + "".join(f"{i + 1:>7}" for i in range(6)))
for i in range(6):
    print(f"  {i + 1}   " + "".join(f"{M[i, j]:>7.3f}" for j in range(6)))

print("\nall pairs, most similar first:")
pairs = sorted(combinations(range(6), 2), key=lambda p: -M[p])
for i, j in pairs:
    print(f"  {M[i, j]:.3f}  ({i + 1},{j + 1})  {SENTENCES[i][:42]!r} <-> {SENTENCES[j][:42]!r}")

manual = float(V[0] @ V[1] / (np.linalg.norm(V[0]) * np.linalg.norm(V[1])))
print(f"\nhand-check of pair (1,2): formula gives {manual:.4f}, matrix says {M[0, 1]:.4f}")
print("pizza (5) scores against others:", [f"{M[4, j]:.3f}" for j in range(6) if j != 4])

print("\n##### PART 2: time and cost for 100 real chunks (structured, size 200) #####")
chunks = [c.text for d in load_documents() for c in chunk_structured(d, 200)][:100]
runs = [emb.embed(chunks) for _ in range(3)]
for k, r in enumerate(runs, 1):
    print(f"run {k}: {len(chunks)} chunks | batches={r.batches} | tokens={r.input_tokens} "
          f"| cost=${r.cost_usd:.6f} | latency={r.latency_s:.2f}s")
print("latency summary over 3 runs:", {k: round(v, 3) for k, v in summarize([r.latency_s for r in runs]).items()})

out = Path(f"benchmarks/results/step5_embeddings_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps({
    "model": res.model, "dimensions": int(V.shape[1]), "sentences": SENTENCES,
    "cosine_matrix": np.round(M, 4).tolist(),
    "chunks_100": {"tokens": runs[0].input_tokens, "cost_usd": runs[0].cost_usd,
                   "latency_s": [r.latency_s for r in runs]}}, indent=2))
print(f"\nsaved: {out}")
