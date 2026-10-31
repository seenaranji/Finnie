"""Step 8: evaluate retrieval on the golden dataset (eval/datasets/retrieval.jsonl).

Two separate questions, measured separately:
  RANKING (gate off): is the right article found, and how high?   Hit@3, Hit@5, MRR, per query type.
                      Answerable questions only. threshold=0 and bm25_min_score=0 so nothing is filtered out.
  GATE (gate on):     does the system refuse what it should, and answer what it should?
                      Wrongly-refused rate (answerable) and no-answer accuracy (unanswerable).

Run:  .venv/bin/python -m eval.scripts.eval_retrieval --split dev
      .venv/bin/python -m eval.scripts.eval_retrieval --split test     (look at test only at milestones!)
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.core.logging import setup_logging
from src.rag.embedder import Embedder, EmbeddingResult
from src.rag.hybrid_retriever import HybridRetriever

DATASET = Path("eval/datasets/retrieval.jsonl")
RESULTS_DIR = Path("eval/results")


class CachingEmbedder(Embedder):
    """Embeds each distinct question once, so a grid of 15 configurations costs 21 API calls, not 300+."""

    def __init__(self, inner: Embedder) -> None:
        self.inner, self.cache = inner, {}
        self.calls = 0

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if len(texts) != 1:
            return self.inner.embed(texts)
        if texts[0] not in self.cache:
            self.calls += 1
            self.cache[texts[0]] = self.inner.embed(texts)
        return self.cache[texts[0]]


def first_rank(passages, expected: list[str]) -> int | None:
    for rank, passage in enumerate(passages, start=1):
        if passage.citation.doc_id in expected:
            return rank
    return None


def run_config(retriever: HybridRetriever, items: list[dict], mode: str, threshold: float,
               bm25_min_score: float | None, rrf_k: int) -> dict:
    retriever.settings.rag.rrf_k = rrf_k
    per_item = []
    for item in items:
        result = retriever.retrieve(item["query"], top_k=5, threshold=threshold, mode=mode, bm25_min_score=bm25_min_score)
        rank = first_rank(result.passages, item["expected_ids"]) if item["answerable"] else None
        found_docs = {p.citation.doc_id for p in result.passages[:5]} & set(item["expected_ids"])
        per_item.append({"id": item["id"], "qtype": item["qtype"], "query": item["query"], "answerable": item["answerable"],
                         "rank": rank, "returned": len(result.passages), "recall5": len(found_docs) / max(len(item["expected_ids"]), 1),
                         "top": [p.chunk_id for p in result.passages[:3]]})
    answerable = [r for r in per_item if r["answerable"]]
    unanswerable = [r for r in per_item if not r["answerable"]]

    def hit(k, rows):
        return sum(1 for r in rows if r["rank"] and r["rank"] <= k) / len(rows) if rows else None

    def mrr(rows):
        return sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows) if rows else None

    by_type = {t: {"n": len([r for r in answerable if r["qtype"] == t]), "mrr": mrr([r for r in answerable if r["qtype"] == t])}
               for t in sorted({r["qtype"] for r in answerable})}
    return {"hit3": hit(3, answerable), "hit5": hit(5, answerable), "mrr": mrr(answerable),
            "recall5": sum(r["recall5"] for r in answerable) / len(answerable),
            "wrongly_refused": sum(1 for r in answerable if r["returned"] == 0), "n_answerable": len(answerable),
            "no_answer_accuracy": sum(1 for r in unanswerable if r["returned"] == 0) / len(unanswerable),
            "n_unanswerable": len(unanswerable), "mrr_by_type": by_type, "items": per_item}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "expansion"], default="dev")
    args = parser.parse_args()
    setup_logging("WARNING")
    items = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    items = [item for item in items if item["split"] == args.split]
    print(f"split={args.split}: {len(items)} questions ({sum(i['answerable'] for i in items)} answerable, "
          f"{sum(not i['answerable'] for i in items)} unanswerable)")
    if args.split == "test":
        print("*** TEST SPLIT: report these numbers; do not tune on them. ***")

    retriever = HybridRetriever()
    retriever.dense.embedder = CachingEmbedder(retriever.dense.embedder)
    default_threshold = retriever.settings.rag.similarity_threshold

    # (name, kind, mode, threshold, bm25_min_score, rrf_k)
    configs = [
        ("dense", "ranking", "dense", 0.0, None, 60),
        ("bm25", "ranking", "bm25", 0.0, 0.0, 60),
        ("hybrid rrf_k=20", "ranking", "hybrid", 0.0, 0.0, 20),
        ("hybrid rrf_k=60", "ranking", "hybrid", 0.0, 0.0, 60),
        ("hybrid rrf_k=100", "ranking", "hybrid", 0.0, 0.0, 100),
        ("dense thr=0.40", "gate", "dense", 0.40, None, 60),
        ("dense thr=0.465", "gate", "dense", default_threshold, None, 60),
        ("dense thr=0.50", "gate", "dense", 0.50, None, 60),
        ("hybrid thr=.465 bm25=off", "gate", "hybrid", default_threshold, None, 60),
        ("hybrid thr=.465 bm25>=2.5", "gate", "hybrid", default_threshold, 2.5, 60),
        ("hybrid thr=.465 bm25>=3.3", "gate", "hybrid", default_threshold, 3.3, 60),
        ("hybrid thr=.465 bm25>=4.0", "gate", "hybrid", default_threshold, 4.0, 60),
    ]
    results = {}
    for name, kind, mode, threshold, bm25_min, rrf_k in configs:
        results[name] = {"kind": kind, "mode": mode, "threshold": threshold, "bm25_min_score": bm25_min, "rrf_k": rrf_k,
                         **run_config(retriever, items, mode, threshold, bm25_min, rrf_k)}

    def f(value):
        return "  n/a" if value is None else f"{value:5.2f}"

    print("\nA. RANKING (gate off, answerable questions only). Higher is better.")
    print(f"  {'config':28} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'Recall@5':>9}   MRR by query type")
    for name, r in results.items():
        if r["kind"] == "ranking":
            types = "  ".join(f"{t}={f(v['mrr'])} (n={v['n']})" for t, v in r["mrr_by_type"].items())
            print(f"  {name:28} {f(r['hit3']):>6} {f(r['hit5']):>6} {f(r['mrr']):>6} {f(r['recall5']):>9}   {types}")

    print("\nB. GATE ON (the system as it would ship). Answerable questions that got nothing back count as misses.")
    print(f"  {'config':28} {'Hit@3':>6} {'MRR':>6} {'wrongly refused':>16} {'no-answer acc':>14}")
    for name, r in results.items():
        if r["kind"] == "gate":
            refused = f"{r['wrongly_refused']}/{r['n_answerable']}"
            noans = f"{round(r['no_answer_accuracy'] * r['n_unanswerable'])}/{r['n_unanswerable']}"
            print(f"  {name:28} {f(r['hit3']):>6} {f(r['mrr']):>6} {refused:>16} {noans:>14}")

    print("\nC. FAILED QUERIES for hybrid rrf_k=60 (ranking, gate off): not in the top 3")
    for row in results["hybrid rrf_k=60"]["items"]:
        if row["answerable"] and not (row["rank"] and row["rank"] <= 3):
            print(f"  [{row['qtype']}] {row['query']!r}: rank={row['rank']} top3={row['top']}  "
                  f"(dense rank={next(i['rank'] for i in results['dense']['items'] if i['id'] == row['id'])}, "
                  f"bm25 rank={next(i['rank'] for i in results['bm25']['items'] if i['id'] == row['id'])})")
    print("\nD. UNANSWERABLE questions wrongly answered at dense thr=0.465")
    for row in results["dense thr=0.465"]["items"]:
        if not row["answerable"] and row["returned"] > 0:
            print(f"  {row['query']!r} returned {row['top']}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"retrieval_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_file.write_text(json.dumps({"split": args.split, "git_commit": commit, "dataset": str(DATASET),
                                       "embedding_model": retriever.settings.rag.embedding_model,
                                       "chunk_size": retriever.settings.rag.chunk_size, "configs": results}, indent=2))
    print(f"\nembedding API calls: {retriever.dense.embedder.calls}")
    print(f"saved: {output_file}")


if __name__ == "__main__":
    main()
