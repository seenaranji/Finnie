"""Step 8 with LangSmith: the same retrieval evaluation, run as LangSmith "experiments".

Vocabulary:
  dataset     = the golden questions (inputs) + the right answers (reference outputs), stored in LangSmith
  target      = the function under test: question in, what retrieval returned out
  evaluator   = scores ONE example (did the right article appear? at what rank?)
  summary evaluator = scores the WHOLE run (average Hit@3, MRR, no-answer accuracy)
  experiment  = one full run of the target over the dataset with one configuration; experiments can be compared

Run locally, nothing uploaded (no account needed):
  .venv/bin/python -m eval.scripts.eval_retrieval_langsmith --split dev
Run and upload to LangSmith (needs LANGSMITH_API_KEY in .env):
  .venv/bin/python -m eval.scripts.eval_retrieval_langsmith --split dev --upload
"""
import argparse
import json
import os
import uuid
from pathlib import Path

import pandas as pd
from langsmith import Client, evaluate
from langsmith.schemas import Example

from eval.scripts.eval_retrieval import CachingEmbedder
from src.core.config import load_settings
from src.core.logging import setup_logging
from src.rag.hybrid_retriever import HybridRetriever

DATASET_FILE = Path("eval/datasets/retrieval.jsonl")

# name -> (mode, similarity threshold, bm25_min_score, rrf_k). Same grid as eval_retrieval.py, gate ON (as shipped).
CONFIGS = {
    "dense-0.465": ("dense", 0.465, None, 60),
    "dense-0.40": ("dense", 0.40, None, 60),
    "hybrid-bm25off": ("hybrid", 0.465, None, 60),
    "hybrid-bm25-3.3": ("hybrid", 0.465, 3.3, 60),
    "hybrid-bm25-4.0": ("hybrid", 0.465, 4.0, 60),
}


def load_examples(split: str) -> list[dict]:
    rows = [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line]
    return [{"inputs": {"query": r["query"]},
             "outputs": {"expected_ids": r["expected_ids"], "answerable": r["answerable"]},
             "metadata": {"id": r["id"], "qtype": r["qtype"], "split": r["split"]}}
            for r in rows if r["split"] == split]


def make_target(retriever: HybridRetriever, mode: str, threshold: float, bm25_min: float | None, rrf_k: int):
    def target(inputs: dict) -> dict:
        retriever.settings.rag.rrf_k = rrf_k
        result = retriever.retrieve(inputs["query"], top_k=5, threshold=threshold, mode=mode, bm25_min_score=bm25_min)
        return {"doc_ids": [p.citation.doc_id for p in result.passages],
                "chunk_ids": [p.chunk_id for p in result.passages],
                "refused": not result.answerable}
    return target


def retrieval_quality(outputs: dict, reference_outputs: dict) -> dict:
    """Per-example scores. Answerable questions get Hit@3, Hit@5, reciprocal rank (1/rank, 0 if missing or refused).
    Unanswerable questions get no_answer_correct (1 if retrieval returned nothing)."""
    if not reference_outputs["answerable"]:
        return {"results": [{"key": "no_answer_correct", "score": int(outputs["refused"])}]}
    rank = next((i for i, doc in enumerate(outputs["doc_ids"], 1) if doc in reference_outputs["expected_ids"]), None)
    return {"results": [{"key": "hit@3", "score": int(bool(rank and rank <= 3))},
                        {"key": "hit@5", "score": int(bool(rank and rank <= 5))},
                        {"key": "reciprocal_rank", "score": 1 / rank if rank else 0.0},
                        {"key": "wrongly_refused", "score": int(outputs["refused"])}]}


def summary(runs, examples) -> dict:
    """Averages over the whole run. Answerable-only for Hit/MRR, unanswerable-only for no-answer accuracy."""
    hit3, hit5, rr, noans = [], [], [], []
    for run, example in zip(runs, examples):
        reference = example.outputs
        if not run.outputs:   # target crashed on this example
            continue
        if reference["answerable"]:
            rank = next((i for i, d in enumerate(run.outputs["doc_ids"], 1) if d in reference["expected_ids"]), None)
            hit3.append(int(bool(rank and rank <= 3)))
            hit5.append(int(bool(rank and rank <= 5)))
            rr.append(1 / rank if rank else 0.0)
        else:
            noans.append(int(run.outputs["refused"]))
    mean = lambda values: sum(values) / len(values) if values else None
    return {"results": [{"key": "Hit@3", "score": mean(hit3)}, {"key": "Hit@5", "score": mean(hit5)},
                        {"key": "MRR", "score": mean(rr)}, {"key": "No-answer accuracy", "score": mean(noans)}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--upload", action="store_true", help="send dataset + results to LangSmith (needs LANGSMITH_API_KEY)")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    args = parser.parse_args()

    setup_logging("WARNING")
    load_settings()   # loads .env into the environment, so LANGSMITH_API_KEY is visible below
    examples = load_examples(args.split)
    retriever = HybridRetriever()
    retriever.dense.embedder = CachingEmbedder(retriever.dense.embedder)

    # Local runs need real Example objects (a plain list of dicts is silently read as zero examples);
    # uploaded runs pass the dataset NAME and LangSmith serves the examples.
    data, client = [Example(id=uuid.uuid4(), **example) for example in examples], None
    if args.upload:
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise SystemExit("LANGSMITH_API_KEY is not set. Add it to .env (see .env.example), or run without --upload.")
        client = Client()
        name = f"finnie-retrieval-{args.split}"
        if not client.has_dataset(dataset_name=name):
            client.create_dataset(name, description=f"Finnie retrieval golden questions ({args.split} split)")
            client.create_examples(dataset_name=name, examples=examples)
            print(f"created LangSmith dataset '{name}' with {len(examples)} examples")
        data = name

    for config_name in args.configs:
        mode, threshold, bm25_min, rrf_k = CONFIGS[config_name]
        results = evaluate(
            make_target(retriever, mode, threshold, bm25_min, rrf_k),
            data=data, evaluators=[retrieval_quality], summary_evaluators=[summary],
            experiment_prefix=f"{args.split}-{config_name}",
            metadata={"mode": mode, "threshold": threshold, "bm25_min_score": bm25_min, "rrf_k": rrf_k,
                      "chunk_size": retriever.settings.rag.chunk_size, "split": args.split},
            client=client, upload_results=args.upload, max_concurrency=0)
        df = results.to_pandas()
        if df.empty:
            raise SystemExit("the experiment ran over 0 examples; check the data argument")
        wanted = [c for c in df.columns if c.startswith("feedback.")]
        print(f"\n=== {config_name}: per-question scores ({len(df)} rows) ===")
        print(df[["inputs.query"] + wanted].fillna("").to_string(index=False, max_colwidth=58))
        score = lambda key: pd.to_numeric(df.get(f"feedback.{key}"), errors="coerce")   # blank (not applicable) -> NaN, ignored by mean()
        print(f"SUMMARY {config_name}: Hit@3={score('hit@3').mean():.3f}  Hit@5={score('hit@5').mean():.3f}  "
              f"MRR={score('reciprocal_rank').mean():.3f}  wrongly refused={int(score('wrongly_refused').sum())}/{score('hit@3').count()}  "
              f"no-answer accuracy={int(score('no_answer_correct').sum())}/{score('no_answer_correct').count()}")
        print(f"embedding API calls so far: {retriever.dense.embedder.calls}")


if __name__ == "__main__":
    main()
