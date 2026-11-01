"""Step 21a: the out-of-scope gate. eval/datasets/scope.jsonl: 40 questions = 20 clearly off-topic, 10 borderline-but-in-scope, 10 plain finance (half dev, half test).

Two errors, very different cost:
  MISSED CATCH   an off-topic question gets through            minor: one wasted call, and the retrieval threshold is still a second net
  FALSE BLOCK    a real finance question is told it is off-topic   worse: the user is failed. Target about 0, and EVERY false block is printed to be read.
Reported for the prompt WITHOUT the gate (v2) and WITH it (v3):
  catch rate        off-topic -> out_of_scope / off-topic total (target >= 90%)
  stopped early     off-topic -> out_of_scope OR clarify (neither calls an agent, retrieval, tool or answering model)
  false-block rate  in-scope -> out_of_scope / in-scope total (target about 0%); also shown BEFORE the low-confidence policy (the policy sends a low-confidence block to clarify)
  in-scope accuracy in-scope questions that reach an acceptable agent
  router cost/latency per call (the extra label makes the prompt longer)

Run:  .venv/bin/python -m eval.scripts.eval_scope --split dev
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.workflow.router import Router

DATASET = Path("eval/datasets/scope.jsonl")


def run(rows, variant):
    """variant is a prompt version, optionally with "+veto", e.g. v3, v4, v3+veto, v4+veto."""
    version, _, extra = variant.partition("+")
    router = Router(LLMClient(), prompt_version=version, finance_veto=extra == "veto")
    out = []
    for row in rows:
        d = router.classify(row["query"])
        out.append({**row, "pred": d.route, "raw_route": d.raw_route, "vetoed": d.vetoed, "confidence": d.confidence, "overridden": d.overridden, "latency_s": d.latency_s,
                    "tokens": d.input_tokens + d.output_tokens, "cost_usd": d.cost_usd})
    return out


def report(results, version):
    off = [r for r in results if r["kind"] == "off_topic"]
    inscope = [r for r in results if r["kind"] != "off_topic"]
    caught = sum(r["pred"] == "out_of_scope" for r in off)
    early = sum(r["pred"] in ("out_of_scope", "clarify") for r in off)
    blocked = [r for r in inscope if r["pred"] == "out_of_scope"]
    raw_blocked = [r for r in inscope if r["raw_route"] == "out_of_scope"]
    clarified = [r for r in inscope if r["pred"] == "clarify"]
    right = sum(r["pred"] in r["expect"] for r in inscope)
    print("=" * 100 + f"\nVARIANT {version}: {len(off)} off-topic, {len(inscope)} in-scope ({sum(r['kind'] == 'borderline' for r in inscope)} borderline)\n" + "=" * 100)
    print(f"  out-of-scope CATCH RATE        : {caught}/{len(off)} = {caught / len(off):.0%}      (target >= 90%)")
    print(f"  stopped early (scope or clarify): {early}/{len(off)} = {early / len(off):.0%}")
    print(f"  FALSE-BLOCK RATE (after policy): {len(blocked)}/{len(inscope)} = {len(blocked) / len(inscope):.0%}      before the low-confidence policy: {len(raw_blocked)}/{len(inscope)}")
    print(f"  in-scope sent to clarify       : {len(clarified)}/{len(inscope)}      in-scope reaching an acceptable agent: {right}/{len(inscope)} = {right / len(inscope):.0%}")
    for r in off:
        if r["pred"] != "out_of_scope":
            print(f"    MISSED CATCH  -> {r['pred']:10} (conf {r['confidence']:6}{', VETOED' if r.get('vetoed') else ''}) {r['query']}")
    for r in raw_blocked:
        print(f"    FALSE BLOCK   -> raw {r['raw_route']}, final {r['pred']}, conf {r['confidence']}: {r['query']}")
    for r in inscope:
        if r["pred"] not in r["expect"] and r["pred"] != "out_of_scope":
            print(f"    in-scope went to {r['pred']} (expected {r['expect']}): {r['query']}")
    print(f"  router per call: {summarize([r['latency_s'] for r in results])['p50']:.2f}s median, {sum(r['tokens'] for r in results) / len(results):.0f} tokens, ${sum(r['cost_usd'] for r in results) / len(results):.6f}")
    return {"catch_rate": caught / len(off), "stopped_early": early / len(off), "false_blocks": len(blocked), "raw_false_blocks": len(raw_blocked), "in_scope": len(inscope),
            "in_scope_accuracy": right / len(inscope), "tokens": sum(r["tokens"] for r in results) / len(results), "cost": sum(r["cost_usd"] for r in results) / len(results),
            "p50_latency": summarize([r["latency_s"] for r in results])["p50"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "stress", "stress2", "stress3"], default="dev")
    parser.add_argument("--versions", nargs="*", default=["v2", "v3"], help="variants such as v3 v4 v3+veto v4+veto")
    args = parser.parse_args()
    setup_logging("ERROR")
    rows = [r for r in (json.loads(line) for line in DATASET.read_text().splitlines() if line) if r["split"] == args.split]
    if args.split in ("test", "stress2"):
        print("*** TEST SPLIT: held out. Report; do not tune on it. ***\n")
    summary = {}
    for version in args.versions:
        results = run(rows, version)
        summary[version] = report(results, version) | {"results": results}
    out = Path(f"eval/results/scope_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
