"""Step 7: show the top similarity score for answerable vs unanswerable questions.

No threshold is applied here: we want the raw numbers first, to choose a threshold from data.

Run:  .venv/bin/python -m scripts.step7_score_questions
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.rag.embedder import OpenAIEmbedder
from src.rag.vector_store import ChromaStore

settings = load_settings()
setup_logging("WARNING")
questions = [json.loads(line) for line in Path("eval/datasets/retrieval_threshold.jsonl").read_text().splitlines() if line]

store = ChromaStore.from_settings(settings)
embedding = OpenAIEmbedder(settings).embed([item["question"] for item in questions])   # one batch for all

rows = []
for item, vector in zip(questions, embedding.vectors):
    best = store.search(vector, top_k=3)[0]
    rows.append({**item, "top_score": round(best.similarity, 3), "top_chunk": best.chunk_id})

print(f"{'id':4} {'label':12} {'kind':14} {'top':>6}  {'top chunk':32} question")
for row in rows:
    article_ok = ""
    if row["label"] == "answerable":
        article_ok = "OK  " if row["top_chunk"].split("#")[0] == row["expected_article"] else "MISS"
    print(f"{row['id']:4} {row['label']:12} {row.get('kind', ''):14} {row['top_score']:>6.3f}  "
          f"{row['top_chunk']:24} {article_ok:4} {row['question']}")

summary = {}
for label in ("answerable", "unanswerable"):
    scores = [row["top_score"] for row in rows if row["label"] == label]
    summary[label] = {key: round(value, 3) for key, value in summarize(scores).items()}
    print(f"\n{label:12} top-score summary: {summary[label]}")

output_file = Path(f"benchmarks/results/step7_scores_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
print(f"\nsaved: {output_file}")
