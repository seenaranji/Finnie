"""Step 24: regression gate. Re-measure a few headline numbers and FAIL (exit 1) if any fell below the saved baseline.

Measured (the shipped configuration, held-out splits where they exist):
  retrieval_hit3_dev / retrieval_hit3_test   Hit@3 of the dense retriever with the shipped threshold (the gate on), answerable questions only
  router_accuracy_test / router_accuracy_test2   router accuracy with the default prompt on the two held-out router splits

Costs real money (about $0.01: 42 embeddings and 64 router calls) and needs the database, so it is NOT part of `pytest`. Run via `make eval-gate`.

  .venv/bin/python -m eval.scripts.check_baseline            compare with eval/baseline.json, exit 1 on a regression
  .venv/bin/python -m eval.scripts.check_baseline --update   record the current numbers as the new baseline (do this on purpose, never to make a red gate green)
"""
import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

BASELINE = Path("eval/baseline.json")
# How far below the baseline a number may fall before the gate fails. Retrieval is deterministic, so one lost question (0.0625 of 16) fails it.
# The router is a model at temperature 0, which is not perfectly repeatable: one flipped query out of 48 (0.021) is allowed, two (0.042) are not.
DEFAULT_TOLERANCE = {"retrieval_hit3_dev": 0.01, "retrieval_hit3_test": 0.01, "router_accuracy_test": 0.025, "router_accuracy_test2": 0.025}


def compare(baseline: dict, current: dict, tolerance: dict) -> list[dict]:
    """One row per baseline metric: passes unless the current value is lower than baseline - tolerance. A metric that was not measured fails."""
    rows = []
    for name, expected in baseline.items():
        value = current.get(name)
        allowed = tolerance.get(name, 0.0)
        rows.append({"metric": name, "baseline": expected, "current": value, "tolerance": allowed,
                     "ok": value is not None and value >= expected - allowed - 1e-9})
    return rows


def measure() -> dict:
    from eval.scripts.eval_retrieval import CachingEmbedder, run_config
    from eval.scripts.eval_router import evaluate
    from src.core.llm import LLMClient
    from src.core.logging import setup_logging
    from src.rag.hybrid_retriever import HybridRetriever
    from src.workflow.router import Router

    setup_logging("ERROR")
    metrics = {}
    retriever = HybridRetriever()
    retriever.dense.embedder = CachingEmbedder(retriever.dense.embedder)
    retrieval_rows = [json.loads(line) for line in Path("eval/datasets/retrieval.jsonl").read_text().splitlines() if line]
    for split in ("dev", "test"):
        rows = [r for r in retrieval_rows if r["split"] == split]
        result = run_config(retriever, rows, "dense", retriever.settings.rag.similarity_threshold, None, retriever.settings.rag.rrf_k)
        metrics[f"retrieval_hit3_{split}"] = round(result["hit3"], 4)

    router = Router(LLMClient())
    router_rows = [json.loads(line) for line in Path("eval/datasets/router.jsonl").read_text().splitlines() if line]
    for split in ("test", "test2"):
        rows = [r for r in router_rows if r["split"] == split]
        scored = [r for r in evaluate(rows, router) if r["label"] in router.labels]   # out_of_scope rows only count when the prompt has that label
        metrics[f"router_accuracy_{split}"] = round(sum(r["correct"] for r in scored) / len(scored), 4)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="save the current numbers as the baseline")
    args = parser.parse_args()
    current = measure()
    if args.update:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        BASELINE.write_text(json.dumps({"recorded": date.today().isoformat(), "git_commit": commit, "tolerance": DEFAULT_TOLERANCE, "metrics": current}, indent=2) + "\n")
        print(f"baseline saved to {BASELINE}: {current}")
        return 0
    saved = json.loads(BASELINE.read_text())
    rows = compare(saved["metrics"], current, saved.get("tolerance", DEFAULT_TOLERANCE))
    print(f"baseline recorded {saved['recorded']} at commit {saved['git_commit']}\n")
    print(f"{'metric':26} {'baseline':>9} {'now':>8} {'allowed drop':>13}  result")
    for r in rows:
        now = "missing" if r["current"] is None else f"{r['current']:.4f}"
        print(f"{r['metric']:26} {r['baseline']:9.4f} {now:>8} {r['tolerance']:13.3f}  {'ok' if r['ok'] else 'REGRESSION'}")
    failed = [r["metric"] for r in rows if not r["ok"]]
    print("\n" + (f"GATE FAILED: {', '.join(failed)}" if failed else "gate passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
