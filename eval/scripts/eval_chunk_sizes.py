"""Step 8: does chunk size change retrieval quality? Evaluate 200 / 500 / 1000 on the DEV split.

Each size gets its own throwaway Chroma + BM25 index in a temp folder, so the real chroma_db/ and
keyword_index/ are never touched. Costs a few thousandths of a cent in embeddings.

Caveats printed with the results: (1) the 0.465 threshold and the BM25 cutoff were tuned at size 500, and
similarity/BM25 scores shift with chunk length, so the GATE columns are not a fair comparison across sizes;
RANKING (gate off) is. (2) top-5 chunks cover very different amounts of text at 200 vs 1000 characters.

Run:  .venv/bin/python -m eval.scripts.eval_chunk_sizes
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

from eval.scripts.eval_retrieval import CachingEmbedder, DATASET, RESULTS_DIR, run_config
from src.core.config import load_settings
from src.core.logging import setup_logging
from src.rag.chunker import chunk_structured
from src.rag.embedder import OpenAIEmbedder
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.keyword_index import KeywordIndex
from src.rag.loader import load_documents
from src.rag.vector_store import ChromaStore

SIZES = [200, 500, 1000]
setup_logging("WARNING")
settings = load_settings()
items = [r for r in map(json.loads, DATASET.read_text().splitlines()) if r["split"] == "dev"]
documents = load_documents()
embedder = CachingEmbedder(OpenAIEmbedder(settings))
temp_dir = tempfile.mkdtemp(prefix="finnie_chunk_sweep_")
print(f"dev questions: {len(items)}; throwaway indexes in {temp_dir}\n")

rows = {}
for size in SIZES:
    chunks = [c for d in documents for c in chunk_structured(d, size)]
    vectors = OpenAIEmbedder(settings).embed([c.text for c in chunks]).vectors
    store = ChromaStore(settings.rag.embedding_model, Path(temp_dir) / f"chroma_{size}", f"sweep_{size}")
    store.add(chunks, vectors)
    retriever = HybridRetriever(store=store, embedder=embedder, keyword_index=KeywordIndex.build(chunks), settings=settings)
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    rows[size] = {"chunks": len(chunks), "avg_chars": round(avg_len)}
    for name, mode, threshold, bm25_min in [("dense", "dense", 0.0, None), ("bm25", "bm25", 0.0, 0.0),
                                            ("hybrid", "hybrid", 0.0, 0.0),
                                            ("hybrid+gate", "hybrid", settings.rag.similarity_threshold, 3.3)]:
        result = run_config(retriever, items, mode, threshold, bm25_min, 60)
        rows[size][name] = {k: result[k] for k in ("hit3", "hit5", "mrr", "wrongly_refused", "no_answer_accuracy")}
        rows[size][name]["items"] = result["items"]

f = lambda v: f"{v:5.2f}"
print("RANKING (gate off), 16 answerable dev questions")
print(f"  {'size':>5} {'chunks':>6} {'avg chars':>9} | {'dense Hit@3':>11} {'MRR':>5} | {'bm25 Hit@3':>10} {'MRR':>5} | {'hybrid Hit@3':>12} {'Hit@5':>5} {'MRR':>5}")
for size, r in rows.items():
    print(f"  {size:>5} {r['chunks']:>6} {r['avg_chars']:>9} | {f(r['dense']['hit3']):>11} {f(r['dense']['mrr']):>5} | "
          f"{f(r['bm25']['hit3']):>10} {f(r['bm25']['mrr']):>5} | {f(r['hybrid']['hit3']):>12} {f(r['hybrid']['hit5']):>5} {f(r['hybrid']['mrr']):>5}")
print("\nGATE ON (threshold 0.465 and bm25 >= 3.3, both tuned at size 500: NOT a fair comparison across sizes)")
print(f"  {'size':>5} | {'Hit@3':>6} {'MRR':>5} {'wrongly refused':>16} {'no-answer':>10}")
for size, r in rows.items():
    g = r["hybrid+gate"]
    print(f"  {size:>5} | {f(g['hit3']):>6} {f(g['mrr']):>5} {str(g['wrongly_refused']) + '/16':>16} {round(g['no_answer_accuracy'] * 5):>8}/5")

print("\nHybrid questions NOT found in the top 3, per size:")
for size, r in rows.items():
    misses = [f"{i['query']!r} (rank {i['rank']})" for i in r["hybrid"]["items"] if i["answerable"] and not (i["rank"] and i["rank"] <= 3)]
    print(f"  {size}: {misses or 'none'}")

output_file = RESULTS_DIR / f"chunk_sweep_dev_{datetime.now():%Y%m%d_%H%M%S}.json"
output_file.write_text(json.dumps({"split": "dev", "sizes": rows}, indent=2))
print(f"\nembedding API calls for questions: {embedder.calls}\nsaved: {output_file}")
