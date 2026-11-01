"""Step 21b: the same multi-intent job run forced-sequential and in parallel. Timelines, then 20 interleaved pairs, then a dependent chain where parallel cannot help.

Run:  .venv/bin/python -m scripts.step21b_parallel_demo
"""
import json
import time
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.news import NewsService
from src.data.providers.yfinance_provider import YFinanceProvider
from src.rag.hybrid_retriever import HybridRetriever
from src.workflow.factory import build_default_graph, config_for
from src.workflow.timeline import render_timeline, timing_summary

JOB = "How is Apple doing, what's the latest news on it, and what is an expense ratio?"
CHAIN = "Find my largest holding and give me the latest news on it"
CSV = "ticker,shares,avg_cost\nVTI,60,220\nBND,100,74\nVXUS,80,58\nAAPL,40,150\n"
setup_logging("ERROR")
settings, llm, retriever = load_settings(), LLMClient(), HybridRetriever()
counter = {"n": 0}


def run(message, mode, csv=None):
    """One COLD run: fresh market and news services, so neither mode benefits from the other's cache."""
    counter["n"] += 1
    graph = build_default_graph(settings=settings, llm=llm, retriever=retriever, checkpointer=None,
                                market_service=MarketDataService(YFinanceProvider(10), TTLCache(), settings), news_service=NewsService(settings.news))
    start = time.perf_counter()
    state = graph.invoke({"message": message, "session_id": f"par-{counter['n']}", "execution": mode, **({"portfolio_csv": csv} if csv else {})}, config_for(f"par-{counter['n']}"))
    wall = time.perf_counter() - start
    meta = state["response"]["metadata"]
    router_cost = sum(t.get("cost_usd") or 0 for t in state["trace"] if t["node"] == "router")
    return {"mode": mode, "wall": wall, "branch_wall": meta["timing"]["wall_s"], "busy": meta["timing"]["busy_s"], "cost": (meta.get("cost_usd") or 0) + router_cost,
            "planner_s": state["plan"]["seconds"], "synthesis_s": next((t["seconds"] for t in state["trace"] if t["node"] == "synthesize"), 0.0),
            "sub": state["sub_results"], "tasks": len(state["sub_results"]), "synthesis": meta.get("synthesis")}


print("=" * 100)
print("EXPECTATION (written before running): branches take about 3.0 s, 2.5 s and 1.7 s -> branch phase about 7.2 s sequential vs about 3.0 s parallel (2.4x).")
print("Planning (about 0.9 s) and synthesis (about 2 s) cannot overlap, so the whole request should speed up less: about 1.6x to 1.8x (Amdahl's law).")
print("=" * 100)
print(f"JOB: {JOB}\n")
for mode in ("sequential", "parallel"):
    r = run(JOB, mode)
    print(f"--- {mode.upper()}: end-to-end {r['wall']:.2f}s (planner {r['planner_s']:.2f}s, branches {r['branch_wall']:.2f}s, synthesis {r['synthesis_s']:.2f}s), cost ${r['cost']:.4f}")
    print(render_timeline(r["sub"]) + "\n")

print("=" * 100 + "\n20 INTERLEAVED PAIRS (cold caches), same job\n" + "=" * 100)
pairs = []
for i in range(20):
    seq, par = run(JOB, "sequential"), run(JOB, "parallel")
    pairs.append((seq, par))
    print(f"  pair {i + 1:2d}: sequential {seq['wall']:5.1f}s (branches {seq['branch_wall']:4.1f}s)   parallel {par['wall']:5.1f}s (branches {par['branch_wall']:4.1f}s)   speedup {seq['wall'] / par['wall']:.2f}x")
seq_w, par_w = [p[0]["wall"] for p in pairs], [p[1]["wall"] for p in pairs]
seq_b, par_b = [p[0]["branch_wall"] for p in pairs], [p[1]["branch_wall"] for p in pairs]
S, P, SB, PB = summarize(seq_w), summarize(par_w), summarize(seq_b), summarize(par_b)
print(f"\n  END-TO-END   sequential p50 {S['p50']:.2f}s p95 {S['p95']:.2f}s | parallel p50 {P['p50']:.2f}s p95 {P['p95']:.2f}s | speedup p50 {S['p50'] / P['p50']:.2f}x, p95 {S['p95'] / P['p95']:.2f}x, mean pair {sum(a / b for a, b in zip(seq_w, par_w)) / 20:.2f}x")
print(f"  BRANCH PHASE sequential p50 {SB['p50']:.2f}s | parallel p50 {PB['p50']:.2f}s | speedup {SB['p50'] / PB['p50']:.2f}x   (planner p50 {summarize([p[1]['planner_s'] for p in pairs])['p50']:.2f}s, synthesis p50 {summarize([p[1]['synthesis_s'] for p in pairs])['p50']:.2f}s: not parallelizable)")
sc, pc = sum(p[0]["cost"] for p in pairs) / 20, sum(p[1]["cost"] for p in pairs) / 20
print(f"  COST per request: sequential ${sc:.5f}, parallel ${pc:.5f}  (parallel changes WHEN the work happens, not how much)")
print(f"  merged by the model in {sum(p[0]['synthesis'] == 'llm' for p in pairs)}/20 sequential and {sum(p[1]['synthesis'] == 'llm' for p in pairs)}/20 parallel runs")

print("\n" + "=" * 100 + "\nWHEN PARALLEL DOES NOT HELP: a dependent chain (the news task needs the portfolio result), 6 interleaved pairs\n" + "=" * 100)
chain = []
for i in range(6):
    seq, par = run(CHAIN, "sequential", CSV), run(CHAIN, "parallel", CSV)
    chain.append((seq, par))
    print(f"  pair {i + 1}: sequential {seq['wall']:5.1f}s   parallel {par['wall']:5.1f}s   speedup {seq['wall'] / par['wall']:.2f}x   branch overlap {timing_summary(par['sub'])['overlap_factor']:.2f}x")
print(f"  p50 sequential {summarize([c[0]['wall'] for c in chain])['p50']:.2f}s vs parallel {summarize([c[1]['wall'] for c in chain])['p50']:.2f}s -> the second task cannot start until the first finishes, so there is nothing to overlap")
print("\nThe chain's timeline (parallel mode):")
print(render_timeline(chain[0][1]["sub"]))

out = Path(f"benchmarks/results/step21b_parallel_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps({"pairs": [{"seq": {k: v for k, v in a.items() if k != "sub"}, "par": {k: v for k, v in b.items() if k != "sub"}} for a, b in pairs],
                           "chain": [{"seq": {k: v for k, v in a.items() if k != "sub"}, "par": {k: v for k, v in b.items() if k != "sub"}} for a, b in chain]}, indent=2, default=str))
print(f"\nsaved: {out}")
