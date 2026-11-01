"""Step 25, part 2: where does the time go INSIDE one chat request, per route?

Runs the real graph in this process (real model, real database, real market data) with timers wrapped around the expensive calls. Time is EXCLUSIVE:
a stage's number does not include stages nested inside it (e.g. the market tool call that happens during an agent's model turn is counted as
`market_data`, not as `llm`). `other` is what is left of the wall time: graph plumbing, prompt building, validation, threads.

  router         Router.classify, including its own model call
  embedding      the question embedding call (retrieval)
  vector_search  the pgvector query
  market_data    MarketDataService.get_quote / get_history (cache hit or provider call)
  news_fetch     NewsService feeds
  portfolio_calc the portfolio math
  llm            every other model call (the answering model, tool-calling turns)

  python -m benchmarks.profile_run [--runs 10] [--label baseline] [--cprofile]
"""
import argparse
import cProfile
import functools
import io
import pstats
import threading
import time
import uuid
from collections import defaultdict

from benchmarks.common import ROUTES, save

STAGES = ("router", "embedding", "vector_search", "market_data", "news_fetch", "portfolio_calc", "llm")
_local = threading.local()


class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.seconds, self.calls = defaultdict(float), defaultdict(int)
        self.llm_in = self.llm_out = 0
        self.cost = 0.0


recorder = Recorder()


def stack():
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def timed(name):
    """Wrap a function so its exclusive time is added to `recorder` under `name` (calls made while it runs are subtracted by their own wrappers)."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            frames = stack()
            if name == "llm" and frames and frames[-1][0] == "router":
                return fn(*args, **kwargs)   # the router's model call is part of the router stage
            frames.append([name, 0.0])
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                _, child = frames.pop()
                recorder.seconds[name] += elapsed - child
                recorder.calls[name] += 1
                if frames:
                    frames[-1][1] += elapsed
        return wrapper
    return decorate


def install():
    from src.agents import portfolio_calc
    from src.core.llm import LLMClient
    from src.data.market_data import MarketDataService
    from src.data.news import NewsService
    from src.rag.embedder import OpenAIEmbedder
    from src.rag.pg_vector_store import PgVectorStore
    from src.workflow.router import Router

    Router.classify = timed("router")(Router.classify)
    OpenAIEmbedder.embed = timed("embedding")(OpenAIEmbedder.embed)
    PgVectorStore.search = timed("vector_search")(PgVectorStore.search)
    for method in ("get_quote", "get_history"):
        setattr(MarketDataService, method, timed("market_data")(getattr(MarketDataService, method)))
    for method in ("get_ticker_news", "get_market_news"):
        setattr(NewsService, method, timed("news_fetch")(getattr(NewsService, method)))
    portfolio_calc.analyze = timed("portfolio_calc")(portfolio_calc.analyze)

    original = LLMClient._complete

    def counted(self, *args, **kwargs):
        response, result = original(self, *args, **kwargs)
        recorder.llm_in += result.input_tokens
        recorder.llm_out += result.output_tokens
        recorder.cost += result.cost_usd
        return response, result
    LLMClient._complete = timed("llm")(counted)


def run_route(graph, route, runs):
    from src.workflow.factory import config_for
    rows = []
    requests = (ROUTES[route] * 2)[:runs]
    for message, csv in [ROUTES[route][0]] + requests:   # the first request is a throwaway warm-up
        recorder.reset()
        session = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        state = graph.invoke({"message": message, "session_id": session, **({"portfolio_csv": csv} if csv else {})}, config_for(session, 4))
        wall = time.perf_counter() - start
        rows.append({"wall": wall, "route": state.get("route"), "seconds": dict(recorder.seconds), "calls": dict(recorder.calls),
                     "tokens_in": recorder.llm_in, "tokens_out": recorder.llm_out, "cost_usd": recorder.cost})
    return rows[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--routes", nargs="*", default=list(ROUTES), choices=list(ROUTES))
    parser.add_argument("--cprofile", action="store_true", help="also print the cProfile top 15 for a QA and a portfolio request")
    args = parser.parse_args()
    from src.core.logging import setup_logging
    setup_logging("ERROR")
    install()
    from src.workflow.factory import build_default_graph
    graph = build_default_graph()

    report = {}
    print(f"{'route':10} {'wall':>7} | " + " ".join(f"{s[:9]:>9}" for s in STAGES) + f" | {'other':>7} | llm calls  tokens in/out   cost")
    for route in args.routes:
        rows = run_route(graph, route, args.runs)
        n = len(rows)
        mean = lambda key: sum(r["seconds"].get(key, 0.0) for r in rows) / n
        wall = sum(r["wall"] for r in rows) / n
        other = wall - sum(mean(s) for s in STAGES)
        report[route] = {"runs": n, "wall_mean_s": wall, "stages_mean_s": {s: mean(s) for s in STAGES} | {"other": other},
                         "llm_calls_mean": sum(r["calls"].get("llm", 0) + r["calls"].get("router", 0) for r in rows) / n,
                         "tokens_in_mean": sum(r["tokens_in"] for r in rows) / n, "tokens_out_mean": sum(r["tokens_out"] for r in rows) / n,
                         "cost_usd_mean": sum(r["cost_usd"] for r in rows) / n, "routed_to": sorted({r["route"] for r in rows}), "rows": rows}
        print(f"{route:10} {wall:7.2f} | " + " ".join(f"{mean(s):9.3f}" for s in STAGES) + f" | {other:7.3f} | {report[route]['llm_calls_mean']:5.1f}  "
              f"{report[route]['tokens_in_mean']:6.0f}/{report[route]['tokens_out_mean']:<5.0f} ${report[route]['cost_usd_mean']:.5f}")
    print("\nshare of wall time (%):")
    for route, r in report.items():
        print(f"{route:10} " + "  ".join(f"{s} {100 * v / r['wall_mean_s']:4.1f}" for s, v in r["stages_mean_s"].items() if v / r["wall_mean_s"] >= 0.005))

    if args.cprofile:
        from src.workflow.factory import config_for
        for route in ("qa", "portfolio"):
            message, csv = ROUTES[route][1]
            session = uuid.uuid4().hex[:12]
            profiler = cProfile.Profile()
            profiler.enable()
            graph.invoke({"message": message, "session_id": session, **({"portfolio_csv": csv} if csv else {})}, config_for(session, 4))
            profiler.disable()
            out = io.StringIO()
            pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(14)
            print(f"\ncProfile, one '{route}' request, top 14 by cumulative time:\n" + out.getvalue())
            report[f"cprofile_{route}"] = out.getvalue()
    print(f"\nraw results: {save(f'step25_stages_{args.label}', report)}")


if __name__ == "__main__":
    main()
