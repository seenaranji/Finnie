"""Step 21b: multi-intent handling on eval/datasets/multi_intent.jsonl (24 queries: 20 multi-intent, 4 single-intent controls).

STAGE router   does the router send multi-intent messages to `multi`, and leave single-intent ones alone?
STAGE planner  given the message, which agents does the plan use? Precision and recall of the agent SET against the expected set (an `optional` agent is neither
               rewarded nor punished), exact-set match, and whether real dependencies were found / false ones invented. The planner is judged on its own: it sees every query.
STAGE e2e      (--e2e) run the whole graph on the multi-intent rows: completeness, how often the synthesizer's text had to be replaced by the code merge, partial answers, cost, latency.
               COMPLETENESS is a CODE check (the LLM judge is paused): every branch that succeeded must contribute something visible to the final answer
               (one of its numbers, citation markers or two or more distinctive words). It cannot tell whether the contribution is correct.
Run:  .venv/bin/python -m eval.scripts.eval_multi_intent --split dev
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.workflow.planner import Planner
from src.workflow.router import Router

DATASET = Path("eval/datasets/multi_intent.jsonl")
CSV = "ticker,shares,avg_cost\nVTI,60,220\nBND,100,74\nVXUS,80,58\nAAPL,40,150\n"
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_MARK = re.compile(r"\[[^\]\s][^\]]*\]")
_WORD = re.compile(r"[a-z]{6,}")


def agent_set_scores(rows, plans):
    tp = fp = fn = exact = 0
    for row, tasks in zip(rows, plans):
        predicted, expected, optional = {t.agent for t in tasks}, set(row["expected"]), set(row["optional"])
        tp += len(predicted & expected); fp += len(predicted - expected - optional); fn += len(expected - predicted)
        exact += expected <= predicted <= expected | optional
    precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, exact / len(rows)


def has_dependency(tasks, later, earlier):
    by_id = {t.id: t for t in tasks}
    return any(t.agent == later and any(by_id[d].agent == earlier for d in t.depends_on) for t in tasks)


def contributes(branch_text: str, final_text: str) -> bool:
    nums = {n.replace(",", "").rstrip(".") for n in _NUM.findall(_MARK.sub(" ", branch_text))}
    if nums & {n.replace(",", "").rstrip(".") for n in _NUM.findall(_MARK.sub(" ", final_text))}:
        return True
    if set(_MARK.findall(branch_text)) & set(_MARK.findall(final_text)):
        return True
    return len(set(_WORD.findall(branch_text.lower())) & set(_WORD.findall(final_text.lower()))) >= 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--planner-models", nargs="*", default=["router_model"])
    parser.add_argument("--e2e", action="store_true")
    args = parser.parse_args()
    setup_logging("ERROR")
    rows = [r for r in (json.loads(l) for l in DATASET.read_text().splitlines() if l) if args.split == "all" or r["split"] == args.split]
    llm = LLMClient()
    report = {}

    print("=" * 100 + f"\nSTAGE router: multi detection ({len(rows)} queries, split={args.split})\n" + "=" * 100)
    router, decisions = Router(llm), []
    for r in rows:
        decisions.append(router.classify(r["query"]))
    multi_rows = [(r, d) for r, d in zip(rows, decisions) if r["kind"] == "multi"]
    single_rows = [(r, d) for r, d in zip(rows, decisions) if r["kind"] == "single"]
    found = sum(d.route == "multi" for _, d in multi_rows)
    false = sum(d.route == "multi" for _, d in single_rows)
    print(f"  multi-intent routed to multi: {found}/{len(multi_rows)} = {found / len(multi_rows):.0%}      single-intent wrongly routed to multi: {false}/{len(single_rows)}")
    for r, d in multi_rows:
        if d.route != "multi":
            print(f"    MISSED multi -> {d.route:10} (conf {d.confidence}) {r['query']}")
    for r, d in single_rows:
        if d.route == "multi":
            print(f"    FALSE multi  -> expected {r['expected']}: {r['query']}")
    report["router"] = {"multi_recall": found / len(multi_rows), "false_multi": false, "n_single": len(single_rows)}

    for model_key in args.planner_models:
        print("\n" + "=" * 100 + f"\nSTAGE planner: model={model_key}\n" + "=" * 100)
        planner, plans, seconds, cost, failures, dropped = Planner(llm, 4, model_key), [], [], 0.0, 0, []
        for r in rows:
            start = time.perf_counter()
            try:
                plan = planner.plan(r["query"])
                plans.append(plan.tasks); cost += plan.cost_usd; dropped += plan.dropped
            except Exception as error:
                plans.append([]); failures += 1
                print(f"    PLAN FAILED ({type(error).__name__}): {r['query']}")
            seconds.append(time.perf_counter() - start)
        precision, recall, exact = agent_set_scores(rows, plans)
        print(f"  agent-set precision {precision:.2f}   recall {recall:.2f}   exact-set match {exact:.0%}   unusable plans {failures}")
        dep_rows = [(r, t) for r, t in zip(rows, plans) if r["depends"]]
        found_deps = sum(all(has_dependency(t, later, earlier) for later, earlier in r["depends"]) for r, t in dep_rows)
        invented = sum(any(task.depends_on for task in t) for r, t in zip(rows, plans) if not r["depends"])
        print(f"  real dependencies found: {found_deps}/{len(dep_rows)}      plans that still carry a dependency where none was expected: {invented}/{len(rows) - len(dep_rows)}      dependencies removed by the code rule: {len(dropped)} {dropped}")
        for r, t in zip(rows, plans):
            predicted, expected, optional = {x.agent for x in t}, set(r["expected"]), set(r["optional"])
            if not (expected <= predicted <= expected | optional):
                print(f"    SET MISMATCH expected {sorted(expected)}+opt{sorted(optional)} got {sorted(predicted)}: {r['query']}")
        print(f"  planner latency {summarize(seconds)}   cost ${cost / len(rows):.6f} per plan")
        report[f"planner_{model_key}"] = {"precision": precision, "recall": recall, "exact": exact, "deps_found": found_deps, "deps_expected": len(dep_rows), "invented_deps": invented,
                                           "failures": failures, "p50_s": summarize(seconds)["p50"], "cost": cost / len(rows)}

    if args.e2e:
        from src.workflow.factory import build_default_graph, config_for
        print("\n" + "=" * 100 + "\nSTAGE e2e: the whole graph on the multi-intent rows (real agents, parallel)\n" + "=" * 100)
        graph = build_default_graph(llm=llm)
        out = []
        for r in (r for r in rows if r["kind"] == "multi"):
            session = f"mi-{r['id']}"
            start = time.perf_counter()
            state = graph.invoke({"message": r["query"], "session_id": session, "portfolio_csv": CSV}, config_for(session))
            wall = time.perf_counter() - start
            meta = state["response"]["metadata"]
            text = state["response"]["text"]
            results = state.get("sub_results") or {}
            ok = [x for x in results.values() if x["status"] == "ok"]
            covered = sum(contributes(x["text"], text) for x in ok) if state["route"] == "multi" and len(results) > 1 else None
            row = {"id": r["id"], "route": state["route"], "tasks": len(results), "ok": len(ok), "contributing": covered, "synthesis": meta.get("synthesis"),
                   "issues": meta.get("synthesis_issues"), "partial": meta.get("partial"), "wall_s": wall, "cost": meta.get("cost_usd") or 0.0, "timing": meta.get("timing")}
            out.append(row)
            print(f"  {r['id']} route={state['route']:9} tasks={row['tasks']} ok={row['ok']} contributing={covered} synthesis={row['synthesis']} {wall:5.1f}s ${row['cost']:.4f}  {r['query'][:60]}")
        merged = [o for o in out if o["contributing"] is not None]
        full = sum(o["contributing"] == o["ok"] for o in merged)
        print(f"\n  ANSWER COMPLETENESS (code check): every successful branch visible in the answer for {full}/{len(merged)} merged answers")
        print(f"  synthesizer text rejected and replaced by the code merge: {sum(o['synthesis'] == 'concat' for o in merged)}/{len(merged)}      partial answers: {sum(bool(o['partial']) for o in out)}")
        print(f"  end-to-end latency {summarize([o['wall_s'] for o in out])}   cost per question ${sum(o['cost'] for o in out) / len(out):.4f}")
        report["e2e"] = out

    path = Path(f"eval/results/multi_intent_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json")
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
