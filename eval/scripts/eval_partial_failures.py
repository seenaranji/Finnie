"""Step 21b: partial results under fault injection. One three-part job, one deliberately broken part per scenario, the full graph with real models.

GRACEFUL means all of:  the graph returned an answer | no technical leakage | a disclaimer | the answer's `missing` list is EXACTLY the parts that were broken |
                        the surviving parts are visible in the answer | the missing parts are named in the text in plain words | it came back within timeout + 12 s.
Score = graceful scenarios / scenarios.

Run:  .venv/bin/python -m eval.scripts.eval_partial_failures
"""
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from eval.scripts.eval_faults import AlwaysDown, BrokenRetriever, FaultyLLM, LEAK
from eval.scripts.eval_multi_intent import contributes
from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.news import NewsService
from src.workflow.factory import build_default_graph, config_for
from src.workflow.synthesizer import TITLES
from src.workflow.timeline import render_timeline

JOB = "How is Apple doing, what's the latest news on it, and what is an expense ratio?"
CHAIN = "Find my largest holding and give me the latest news on it"
CSV = "ticker,shares,avg_cost\nVTI,60,220\nBND,100,74\nVXUS,80,58\nAAPL,40,150\n"


def main():
    setup_logging("CRITICAL")
    settings, real = load_settings(), LLMClient()
    dead_news = NewsService(settings.news, get=lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("injected")))

    def slow_news_get(*a, **k):
        time.sleep(9)
        raise requests.ConnectionError("injected")

    dead_market = MarketDataService(AlwaysDown(), TTLCache(), settings, sleep=lambda s: None)
    results = []

    def scenario(name, message, broken, timeout=None, csv=None, **graph_kwargs):
        s = settings.model_copy(deep=True)
        if timeout:
            s.workflow.branch_timeout_seconds = timeout
        graph = build_default_graph(settings=s, **graph_kwargs)
        session, start = f"pf-{name[:12]}", time.perf_counter()
        raised = None
        try:
            state = graph.invoke({"message": message, "session_id": session, **({"portfolio_csv": csv} if csv else {})}, config_for(session))
        except Exception as error:
            state, raised = None, f"{type(error).__name__}: {error}"
        seconds = time.perf_counter() - start
        r = (state or {}).get("response", {})
        text, meta = r.get("text", ""), r.get("metadata", {})
        sub = (state or {}).get("sub_results") or {}
        ok_branches = [x for x in sub.values() if x["status"] == "ok"]
        missing = sorted(meta.get("missing") or [])
        names_missing = all(TITLES[a].lower() in text.lower() for a in broken) if broken else True
        survivors_visible = True if not ok_branches else all(contributes(x["text"], text) for x in ok_branches)   # nothing survived = nothing to show
        checks = {"returned": raised is None and bool(text), "no_leak": not LEAK.search(text), "disclaimer": bool(r.get("disclaimer")),
                  "missing_is_exact": missing == sorted(broken) or (not meta.get("missing") and not broken), "survivors_visible": survivors_visible,
                  "missing_named": names_missing, "in_time": seconds < (timeout or settings.workflow.branch_timeout_seconds) + 12}
        graceful = all(checks.values())
        results.append({"scenario": name, "graceful": graceful, "checks": checks, "seconds": seconds, "missing": missing, "text": text, "statuses": {k: v["status"] for k, v in sub.items()}})
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  {'GRACEFUL' if graceful else 'FAILED  '} {name:48} {seconds:5.1f}s  statuses={ {k: v['status'] for k, v in sub.items()} }  missing={missing}" + (f"  failed: {failed}" if failed else ""))
        print(f"      answer: {text[:330].replace(chr(10), ' | ')!r}")
        if sub and name.startswith("slow"):
            print(render_timeline(sub))

    print("=" * 100 + "\nPARTIAL FAILURES THROUGH THE FULL GRAPH\n" + "=" * 100)
    scenario("nothing broken (control)", JOB, [])
    scenario("news feeds down", JOB, ["news"], news_service=dead_news)
    scenario("market provider down", JOB, ["market"], market_service=dead_market)
    scenario("knowledge base broken", JOB, ["finance_qa"], retriever=BrokenRetriever())
    scenario("slow news branch (9 s) with a 4 s timeout", JOB, ["news"], timeout=4.0, news_service=NewsService(settings.news, get=slow_news_get))
    scenario("news AND market down", JOB, ["news", "market"], news_service=dead_news, market_service=dead_market)
    scenario("all three parts down", JOB, ["finance_qa", "market", "news"], news_service=dead_news, market_service=dead_market, retriever=BrokenRetriever())
    scenario("dependency: portfolio down, news skipped", CHAIN, ["portfolio", "news"], csv=CSV, market_service=dead_market)
    scenario("merging model down", JOB, [], llm=FaultyLLM(real, lambda kind, kw: str(kw.get("system", "")).startswith("You combine the answers")), router_llm=real)
    scenario("planner model down", JOB, [], llm=FaultyLLM(real, lambda kind, kw: (kw.get("response_format") or {}).get("json_schema", {}).get("name") == "plan"), router_llm=real)
    graceful = sum(r["graceful"] for r in results)
    print("\n" + "=" * 100 + f"\nGRACEFUL PARTIAL-FAILURE RATE: {graceful}/{len(results)} = {graceful / len(results):.0%}")
    out = Path(f"eval/results/partial_failures_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
