"""Step 21: evaluate the router on eval/datasets/router.jsonl (72 queries: 8 labels x 8, plus 8 multi-intent).

What is scored
  - A prediction is CORRECT if it equals the label, or is listed in the query's `also_ok` (multi-intent and borderline queries have more than one defensible answer).
  - `out_of_scope` queries cannot be routed correctly yet (the router has no such label until Step 21a). They are NOT counted in accuracy; the report shows where they landed,
    so Step 21a can show the improvement.
  - Per-class precision, recall and F1, and a confusion matrix, over the 7 routable labels.
  - Confidence: accuracy at each confidence level, and what the low-confidence -> clarify policy does.
The dev split is for improving the prompt. The test split is held out: run it once per prompt version, at the end.

Run:  .venv/bin/python -m eval.scripts.eval_router --prompt v1 --split dev
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.workflow.router import PROMPTS, ROUTES, Router

DATASET = Path("eval/datasets/router.jsonl")


def evaluate(rows: list[dict], router: Router) -> list[dict]:
    results = []
    for row in rows:
        d = router.classify(row["query"])
        accepted = {row["label"], *row["also_ok"]}
        results.append({**row, "pred": d.route, "raw_route": d.raw_route, "confidence": d.confidence, "overridden": d.overridden, "fell_back": d.fell_back,
                        "correct": d.route in accepted, "raw_correct": d.raw_route in accepted,
                        "effective": row["label"] if d.route in accepted else d.route, "latency_s": d.latency_s, "cost_usd": d.cost_usd,
                        "tokens": d.input_tokens + d.output_tokens})
    return results


def report(results: list[dict], title: str) -> dict:
    routable = [r for r in results if r["label"] != "out_of_scope"]
    offtopic = [r for r in results if r["label"] == "out_of_scope"]
    correct = sum(r["correct"] for r in routable)
    print("=" * 100 + f"\n{title}\n" + "=" * 100)
    print(f"ACCURACY (7 routable labels, {len(routable)} queries, multi-intent credited by also_ok): {correct}/{len(routable)} = {correct / len(routable):.1%}")
    singles = [r for r in routable if r["kind"] == "single"]
    hards = [r for r in routable if r["kind"] == "hard"]
    multis = [r for r in routable if r["kind"] == "multi"]
    print(f"   single-intent {sum(r['correct'] for r in singles)}/{len(singles)}   multi-intent {sum(r['correct'] for r in multis)}/{len(multis)}   hard {sum(r['correct'] for r in hards)}/{len(hards)}")

    print("\nPER CLASS (precision / recall / F1; 'effective' prediction = the label whenever the prediction was acceptable)")
    print(f"  {'class':12}{'support':>8}{'precision':>11}{'recall':>8}{'F1':>7}")
    per_class = {}
    for label in ROUTES:
        support = sum(r["label"] == label for r in routable)
        tp = sum(r["label"] == label and r["effective"] == label for r in routable)
        predicted = sum(r["effective"] == label for r in routable)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        print(f"  {label:12}{support:>8}{precision:>11.2f}{recall:>8.2f}{f1:>7.2f}")

    print("\nCONFUSION MATRIX (rows = true label, columns = predicted route)")
    print(f"  {'':12}" + "".join(f"{c[:9]:>10}" for c in ROUTES))
    matrix = defaultdict(Counter)
    for r in routable:
        matrix[r["label"]][r["effective"]] += 1
    for label in ROUTES:
        print(f"  {label:12}" + "".join(f"{matrix[label][c]:>10}" for c in ROUTES))

    print("\nWRONG ROUTES")
    for r in routable:
        if not r["correct"]:
            print(f"  [{r['label']:10}->{r['pred']:10}] conf={r['confidence']:6} raw={r['raw_route']:10} {r['query']}")

    print("\nCONFIDENCE (does the model's own confidence carry information?)")
    for level in ("high", "medium", "low"):
        sub = [r for r in routable if r["confidence"] == level]
        if sub:
            print(f"  {level:6}: {len(sub):2d} queries, the model's raw route was acceptable in {sum(r['raw_correct'] for r in sub)}/{len(sub)} = {sum(r['raw_correct'] for r in sub) / len(sub):.0%}")
    overridden = [r for r in routable if r["overridden"]]
    print(f"  policy 'low confidence -> clarify' changed {len(overridden)} routes: {sum(r['raw_correct'] for r in overridden)} were right before (now lost), "
          f"{sum(not r['raw_correct'] for r in overridden)} were wrong before (now sent to clarify)")
    print(f"  routed to clarify: {sum(r['pred'] == 'clarify' for r in routable)}/{len(routable)} routable queries ({sum(r['pred'] == 'clarify' and r['label'] != 'clarify' for r in routable)} of them should not have been)")

    print(f"\nOFF-TOPIC QUERIES ({len(offtopic)}), no out_of_scope label yet: where they landed")
    for r in offtopic:
        print(f"  -> {r['pred']:10} (conf {r['confidence']:6}) {r['query']}")
    print(f"  counts: {dict(Counter(r['pred'] for r in offtopic))}")

    latencies = [r["latency_s"] for r in results]
    print(f"\nLATENCY {summarize(latencies)}")
    print(f"COST ${sum(r['cost_usd'] for r in results) / len(results):.6f} per classification, {sum(r['tokens'] for r in results) / len(results):.0f} tokens per call")
    return {"accuracy": correct / len(routable), "correct": correct, "n": len(routable), "per_class": per_class,
            "confusion": {k: dict(v) for k, v in matrix.items()}, "offtopic_routes": dict(Counter(r["pred"] for r in offtopic)),
            "clarify_rate": sum(r["pred"] == "clarify" for r in routable) / len(routable)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v1")
    parser.add_argument("--split", choices=["dev", "test", "dev2", "test2"], default="dev")
    parser.add_argument("--no-policy", action="store_true", help="do not turn low-confidence routes into clarify")
    args = parser.parse_args()
    setup_logging("ERROR")
    rows = [r for r in (json.loads(line) for line in DATASET.read_text().splitlines() if line) if r["split"] == args.split]
    if args.split in ("test", "test2"):
        print("*** TEST SPLIT: held out. Report these numbers; do not tune the prompt on them. ***\n")
    router = Router(LLMClient(), prompt_version=args.prompt, low_confidence_to_clarify=not args.no_policy)
    results = evaluate(rows, router)
    summary = report(results, f"ROUTER prompt={args.prompt} split={args.split} policy={'off' if args.no_policy else 'on'} ({len(rows)} queries)")
    output = Path(f"eval/results/router_{args.prompt}_{args.split}{'_nopolicy' if args.no_policy else ''}_{datetime.now():%Y%m%d_%H%M%S}.json")
    output.write_text(json.dumps({"prompt": args.prompt, "split": args.split, "summary": summary, "results": results}, indent=2, default=str))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
