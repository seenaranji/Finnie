"""Step 7c experiment: dense vs BM25 vs hybrid, side by side, plus a 15-query rank table and latency.

Everything here is UNGATED (raw ranks from each retriever, top 20) so we can see what each retriever
finds before any threshold is applied. The gate is tested separately.

Run:  .venv/bin/python -m scripts.step7c_side_by_side
"""
import json
import time
from datetime import datetime
from pathlib import Path

from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.hybrid_retriever import HybridRetriever

setup_logging("WARNING")
retriever = HybridRetriever()
RRF_K = retriever.settings.rag.rrf_k

# (query, kind, article ids that would count as a correct answer; empty = nothing in the KB should answer it)
# The expected articles are MY labels (guesses), same caveat as Step 7.
QUERIES = [
    ("VOO expense ratio", "exact (VOO not in KB)", {"etf-basics", "etf-vs-mutual-fund"}),
    ("401(k) match", "exact", {"retirement-accounts-overview"}),
    ("what is HHI", "exact (HHI not in KB)", set()),
    ("a basket of stocks I can trade", "paraphrase", {"etf-basics"}),
    ("spreading my money around", "paraphrase", {"diversification"}),
    ("best pizza in Rome", "off-topic", set()),
    ("what is a Roth IRA", "exact", {"retirement-accounts-overview", "glossary"}),
    ("expense ratio", "exact", {"etf-basics", "etf-vs-mutual-fund", "mutual-funds-basics", "glossary"}),
    ("what does NAV stand for", "exact", {"mutual-funds-basics", "glossary", "etf-vs-mutual-fund"}),
    ("IRA vs 401(k)", "exact", {"retirement-accounts-overview"}),
    ("why does my savings grow faster the longer I leave it", "paraphrase", {"compound-interest"}),
    ("loans I make to a company", "paraphrase", {"bonds-basics"}),
    ("how much could I lose if the market drops", "paraphrase", {"risk-and-return"}),
    ("what do I own when I buy a company's shares", "paraphrase", {"stocks-basics"}),
    ("S&P 500 index fund", "exact (S&P not in KB)", {"etf-basics", "etf-vs-mutual-fund"}),
]
SIDE_BY_SIDE = QUERIES[:6]   # the six queries named in the spec


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#")[0]


def first_rank(chunk_ids: list[str], expected: set[str]) -> int | None:
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if doc_of(chunk_id) in expected:
            return rank
    return None


def lists_for(query: str):
    found = retriever.candidates(query)
    dense_ids = [p.chunk_id for p in found.dense]
    keyword_ids = [h.chunk_id for h in found.keyword]
    fused = reciprocal_rank_fusion([dense_ids, keyword_ids], RRF_K)
    return found, dense_ids, keyword_ids, fused


# ------------------------------------------------------------------ A. side by side
print("=" * 100, "\nA. SIDE BY SIDE (top 5 per retriever, no threshold)\n" + "=" * 100)
for query, kind, expected in SIDE_BY_SIDE:
    found, dense_ids, keyword_ids, fused = lists_for(query)
    dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
    keyword_rank = {cid: r for r, cid in enumerate(keyword_ids, 1)}
    print(f"\nQUERY: {query!r}  [{kind}]  expected: {sorted(expected) or 'nothing'}")
    print(f"  BM25 returned {len(keyword_ids)} chunks with a score > 0")
    print(f"  {'rank':4} | {'DENSE (cosine)':38} | {'BM25 (score)':38} | {'HYBRID (RRF) [dense rank / bm25 rank]':50}")
    for i in range(5):
        dense = f"{found.dense[i].chunk_id} ({found.dense[i].similarity:.3f})" if i < len(found.dense) else "-"
        keyword = f"{found.keyword[i].chunk_id} ({found.keyword[i].score:.2f})" if i < len(found.keyword) else "-"
        if i < len(fused):
            cid, score = fused[i]
            hybrid = f"{cid} {score:.5f} [{dense_rank.get(cid, '-')}/{keyword_rank.get(cid, '-')}]"
        else:
            hybrid = "-"
        print(f"  {i + 1:4} | {dense:38} | {keyword:38} | {hybrid:50}")

# ------------------------------------------------------------------ B. RRF arithmetic by hand
query = "401(k) match"
found, dense_ids, keyword_ids, fused = lists_for(query)
print("\n" + "=" * 100, f"\nB. RRF BY HAND for {query!r}  (k={RRF_K})\n" + "=" * 100)
for cid, score in fused[:5]:
    parts = []
    if cid in dense_ids:
        parts.append(f"1/({RRF_K}+{dense_ids.index(cid) + 1}) = {1 / (RRF_K + dense_ids.index(cid) + 1):.5f}  [dense]")
    if cid in keyword_ids:
        parts.append(f"1/({RRF_K}+{keyword_ids.index(cid) + 1}) = {1 / (RRF_K + keyword_ids.index(cid) + 1):.5f}  [bm25]")
    print(f"  {cid:34} {'  +  '.join(parts)}  =>  {score:.5f}")

# ------------------------------------------------------------------ C. rank of the expected article
print("\n" + "=" * 100, "\nC. RANK OF THE FIRST CORRECT CHUNK per mode (top 20 each; '-' = not found; no threshold)\n" + "=" * 100)
print(f"  {'query':56} {'kind':22} {'dense':>6} {'bm25':>6} {'hybrid':>7}")
table = []
for query, kind, expected in QUERIES:
    found, dense_ids, keyword_ids, fused = lists_for(query)
    if not expected:
        print(f"  {query:56} {kind:22} {'(nothing should match)':>21}   top dense {found.dense[0].similarity:.3f}, "
              f"bm25 hits {len(keyword_ids)}")
        table.append({"query": query, "kind": kind, "expected": [], "dense": None, "bm25": None, "hybrid": None,
                      "top_dense_similarity": found.dense[0].similarity, "bm25_hits": len(keyword_ids)})
        continue
    ranks = {"dense": first_rank(dense_ids, expected), "bm25": first_rank(keyword_ids, expected),
             "hybrid": first_rank([cid for cid, _ in fused], expected)}
    show = {name: (rank if rank else "-") for name, rank in ranks.items()}
    print(f"  {query:56} {kind:22} {show['dense']:>6} {show['bm25']:>6} {show['hybrid']:>7}")
    table.append({"query": query, "kind": kind, "expected": sorted(expected), **ranks})

# ------------------------------------------------------------------ D. latency
print("\n" + "=" * 100, "\nD. LATENCY (ms)\n" + "=" * 100)
timings = {"embed (API)": [], "dense search": [], "bm25 search": [], "RRF fusion": []}
end_to_end = {"dense": [], "bm25": [], "hybrid": []}
for query, _, _ in QUERIES:
    start = time.perf_counter()
    vector = retriever.dense.embedder.embed([query]).vectors[0]
    timings["embed (API)"].append((time.perf_counter() - start) * 1000)
    for _ in range(20):   # tiny operations: repeat so the clock can see them
        start = time.perf_counter()
        retriever.dense.store.search(vector, 20, None)
        timings["dense search"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        keyword_hits = retriever.keyword_index.search(query, 20)
        timings["bm25 search"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        reciprocal_rank_fusion([[f"d{i}" for i in range(20)], [h.chunk_id for h in keyword_hits]], RRF_K)
        timings["RRF fusion"].append((time.perf_counter() - start) * 1000)
for mode in end_to_end:   # whole retrieve() call, includes the embedding API call for dense/hybrid (noisy)
    for query, _, _ in QUERIES:
        start = time.perf_counter()
        retriever.retrieve(query, mode=mode)
        end_to_end[mode].append((time.perf_counter() - start) * 1000)
print(f"  {'step':22} {'n':>4} {'p50':>9} {'p95':>9} {'max':>9}")
for name, values in {**timings, **{f'retrieve() mode={m}': v for m, v in end_to_end.items()}}.items():
    s = summarize(values)
    print(f"  {name:22} {s['count']:>4} {s['p50']:>9.2f} {s['p95']:>9.2f} {s['max']:>9.2f}")

# ------------------------------------------------------------------ E. the filters switched on
rag = retriever.settings.rag
print("\n" + "=" * 100, f"\nE. WITH FILTERING ON: threshold={rag.similarity_threshold}, bm25_min_score={rag.bm25_min_score}, "
      f"dedupe, category filter, top_k={rag.top_k}\n" + "=" * 100)
for query, kind, expected in QUERIES:
    result = retriever.retrieve(query, mode="hybrid")
    shown = [f"{p.chunk_id}(d{p.dense_rank or '-'}/b{p.bm25_rank or '-'})" for p in result.passages]
    print(f"  {query:56} answerable={str(result.answerable):5} dropped_by_gate={result.dropped_by_threshold:3} -> {shown}")
print("\n  Category filter 'etfs' applied to BOTH retrievers, query 'expense ratio':")
result = retriever.retrieve("expense ratio", mode="hybrid", category="etfs", threshold=0.0)
print("   ", [p.chunk_id for p in result.passages])
found = retriever.candidates("expense ratio", category="etfs")
print("    dense categories :", {p.citation.doc_id for p in found.dense})
print("    bm25 categories  :", {h.metadata.get("doc_id") for h in found.keyword})

# ------------------------------------------------------------------ F. choosing bm25_min_score from data
print("\n" + "=" * 100, "\nF. BM25 TOP SCORE on the 24 tuning questions (to choose bm25_min_score; hold-out untouched)\n" + "=" * 100)
tuning = [json.loads(line) for line in Path("eval/datasets/retrieval_threshold.jsonl").read_text().splitlines() if line]
scores = {"answerable": [], "unanswerable": []}
for item in tuning:
    hits = retriever.keyword_index.search(item["question"], 1)
    scores[item["label"]].append(hits[0].score if hits else 0.0)
for label, values in scores.items():
    print(f"  {label:13} n={len(values)} sorted={[round(v, 2) for v in sorted(values)]}")

output = {"rrf_k": RRF_K, "bm25_top_scores_tuning": scores, "rank_table": table,
          "latency_ms": {name: summarize(v) for name, v in {**timings, **{f'retrieve_{m}': v for m, v in end_to_end.items()}}.items()}}
output_file = Path(f"benchmarks/results/step7c_side_by_side_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(output, indent=2))
print(f"\nsaved: {output_file}")
