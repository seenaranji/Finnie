"""Step 7: apply the configured similarity threshold to a labeled question set.

Counts wrongly refused (answerable but dropped) and wrongly answered (unanswerable but kept).

Run:  .venv/bin/python -m scripts.step7_check_threshold eval/datasets/retrieval_holdout.jsonl
"""
import json
import sys
from pathlib import Path

from src.core.logging import setup_logging
from src.rag.retriever import Retriever

setup_logging("WARNING")
questions = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
retriever = Retriever()
threshold = retriever.settings.rag.similarity_threshold
print(f"threshold = {threshold}   dataset = {sys.argv[1]}\n")

wrongly_refused = wrongly_answered = 0
print(f"{'id':4} {'label':12} {'best':>6}  {'decision':9} {'verdict':14} question")
for item in questions:
    result = retriever.retrieve(item["question"])
    decision = "ANSWER" if result.answerable else "REFUSE"
    if item["label"] == "answerable":
        correct = result.answerable
        wrongly_refused += not correct
    else:
        correct = not result.answerable
        wrongly_answered += not correct
    top_chunk = result.passages[0].chunk_id if result.passages else "-"
    print(f"{item['id']:4} {item['label']:12} {result.best_similarity:>6.3f}  {decision:9} "
          f"{'ok' if correct else '*** WRONG ***':14} {item['question']}   -> {top_chunk}")

answerable = sum(item["label"] == "answerable" for item in questions)
print(f"\nwrongly refused : {wrongly_refused} / {answerable} answerable")
print(f"wrongly answered: {wrongly_answered} / {len(questions) - answerable} unanswerable")
