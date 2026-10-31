"""Step 7 extras: category filter, dedupe, citations, k=2 vs 5 vs 8, and reranking.

Run:  .venv/bin/python -m scripts.step7_extras_demo
"""
import json
from pathlib import Path

from src.core.logging import setup_logging
from src.rag.passages import Citation, Passage, dedupe_passages
from src.rag.reranker import LLMReranker
from src.rag.retriever import Retriever

setup_logging("WARNING")
retriever = Retriever()
tuning_questions = [json.loads(line) for line in Path("eval/datasets/retrieval_threshold.jsonl").read_text().splitlines() if line]
holdout_questions = [json.loads(line) for line in Path("eval/datasets/retrieval_holdout.jsonl").read_text().splitlines() if line]
answerable = [item for item in tuning_questions + holdout_questions if item["label"] == "answerable"]


def show(result, limit=5):
    for rank, passage in enumerate(result.passages[:limit], 1):
        rerank = f" rerank={passage.rerank_score:.0f}" if passage.rerank_score is not None else ""
        print(f"   {rank}. sim={passage.similarity:.3f}{rerank}  {passage.chunk_id:34} [{passage.citation.section}]")


print("=" * 70, "\nA. CATEGORY FILTER (threshold off so we see everything the filter lets through)\n" + "=" * 70)
question = "how do I spread my risk across investments"
for category in (None, "etfs", "bonds", "taxes"):
    result = retriever.retrieve(question, top_k=5, category=category, threshold=0.0)
    print(f"category={category!r}: {len(result.passages)} passages, candidates fetched={result.candidates}")
    show(result)

print("\n" + "=" * 70, "\nB. DEDUPE (fake passages, no API call)\n" + "=" * 70)
citation = Citation("doc", "Doc", "s", "src", "url", "2026-10-30")
fake = [
    Passage("doc#0", "Diversification means spreading your money across different investments so one bad result does not hurt you.", 0.60, citation),
    Passage("doc#1", "Spreading your money across different investments so one bad result does not hurt you is called diversification.", 0.55, citation),
    Passage("doc#2", "Bonds are loans you make to a government or company in return for interest.", 0.50, citation),
    Passage("doc#3", "Bonds are loans.", 0.49, citation),
    Passage("doc#4", "Bonds are loans.", 0.48, citation),
]
kept, dropped = dedupe_passages(fake, overlap=0.8)
print(f"in: {[p.chunk_id for p in fake]}\nkept: {[p.chunk_id for p in kept]}  dropped: {dropped}")
print("Real index: duplicates dropped across all", len(answerable), "answerable questions =",
      sum(retriever.retrieve(item["question"], threshold=0.0).dropped_as_duplicates for item in answerable))

print("\n" + "=" * 70, "\nC. CITATIONS\n" + "=" * 70)
result = retriever.retrieve("what is the difference between an ETF and a mutual fund", top_k=2)
for passage in result.passages:
    print(f"- {passage.citation.label()}\n  url: {passage.citation.source_url}\n  chunk: {passage.chunk_id}")

print("\n" + "=" * 70, "\nD. k = 2 vs 5 vs 8 on", len(answerable), "answerable questions (config threshold)\n" + "=" * 70)
print(f"{'k':>3} {'avg passages':>13} {'avg from expected article':>26} {'avg from other articles (noise)':>32} {'questions with expected article':>34}")
for k in (2, 5, 8):
    total = from_expected = hit_questions = 0
    for item in answerable:
        passages = retriever.retrieve(item["question"], top_k=k).passages
        matching = sum(passage.citation.doc_id == item["expected_article"] for passage in passages)
        total += len(passages)
        from_expected += matching
        hit_questions += matching > 0
    count = len(answerable)
    print(f"{k:>3} {total / count:>13.2f} {from_expected / count:>26.2f} {(total - from_expected) / count:>32.2f} {hit_questions:>30}/{count}")
print("(labels are my own guesses of the best article; a question can be answerable from several)")

print("\n" + "=" * 70, "\nE. RERANKING (threshold 0.3 so there are enough candidates to reorder)\n" + "=" * 70)
total_cost = total_latency = 0.0
for item in answerable:
    if item["id"] not in ("a02", "a11", "a12", "h05", "h03", "a10"):
        continue
    before = retriever.retrieve(item["question"], top_k=5, threshold=0.3, rerank=False)
    after = retriever.retrieve(item["question"], top_k=5, threshold=0.3, rerank=True)
    total_cost += after.rerank_cost_usd
    total_latency += after.rerank_latency_s
    print(f"\n{item['id']} {item['question']!r}   (my expected article: {item['expected_article']})")
    print(f" candidates passing threshold: {before.candidates - before.dropped_by_threshold}, "
          f"rerank cost ${after.rerank_cost_usd:.6f}, rerank latency {after.rerank_latency_s:.2f}s")
    print(" BEFORE (vector search order):")
    show(before, 3)
    print(" AFTER  (reranked):")
    show(after, 3)
print(f"\nrerank total for 6 questions: ${total_cost:.6f}, {total_latency:.2f}s (about {total_latency / 6:.2f}s extra per question)")
