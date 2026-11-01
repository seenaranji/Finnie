"""Step 21: fault injection through the WHOLE graph (real router, real resolver, real agents; one component deliberately broken per scenario).

A scenario is GRACEFUL if ALL of these hold:
  - the graph returned an answer (nothing raised)                 - the text has no technical leakage (exception names, tracebacks, file paths)
  - the reply says plainly that something is unavailable / could not be done (unless the scenario expects a normal answer)
  - a disclaimer is attached                                      - it came back within 60 seconds
  - market scenarios show no invented price (no $ figure unless the data was flagged stale)
Score = graceful scenarios / scenarios. This is graceful DEGRADATION, not correctness of normal answers.

Run:  .venv/bin/python -m eval.scripts.eval_faults
"""
import re
import time

import requests
from datetime import datetime
from pathlib import Path
import json

from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable
from src.data.market_data import MarketDataService
from src.data.news import NewsService
from src.data.providers.base import MarketDataProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.workflow.factory import build_default_graph, config_for

LEAK = re.compile(r"Traceback|\b\w+Error\b|\b\w+Exception\b|File \"|NoneType|\bat 0x", re.I)
APOLOGY = re.compile(r"sorry|couldn't|could not|can't|cannot|unable|not (responding|available)|try again|unavailable|don't have|no recent news|not able", re.I)
STALE = re.compile(r"out of date|outdated|stale|may not be current|last (known|saved)|as of", re.I)
PRICE = re.compile(r"\$\s?\d")


class FaultyLLM:
    """Wraps the real client and raises on the calls the scenario wants broken."""
    def __init__(self, real, fail_when):
        self._real, self._fail_when = real, fail_when

    def chat(self, *args, **kwargs):
        if self._fail_when("chat", kwargs):
            raise ConnectionError("injected: model unreachable")
        return self._real.chat(*args, **kwargs)

    def complete(self, *args, **kwargs):
        if self._fail_when("complete", kwargs):
            raise ConnectionError("injected: model unreachable")
        return self._real.complete(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class AlwaysDown(MarketDataProvider):
    name = "down"

    def get_quote(self, symbol):
        raise ProviderUnavailable("The market data service is not responding right now.")

    get_history = get_quote


class Breakable(MarketDataProvider):
    name = "breakable"

    def __init__(self, real):
        self.real, self.broken = real, False

    def get_quote(self, symbol):
        if self.broken:
            raise ProviderUnavailable("The market data service is not responding right now.")
        return self.real.get_quote(symbol)

    def get_history(self, symbol, period):
        if self.broken:
            raise ProviderUnavailable("The market data service is not responding right now.")
        return self.real.get_history(symbol, period)


class BrokenRetriever:
    def retrieve(self, *args, **kwargs):
        raise RuntimeError("injected: vector store corrupted")


def main() -> None:
    setup_logging("CRITICAL")
    settings, real = load_settings(), LLMClient()
    fast = settings.model_copy(deep=True)
    fast.cache.quote_ttl_seconds = fast.cache.history_ttl_seconds = 1   # so a cached value expires within the scenario
    fast.market_data.retry_base_seconds = 0.1
    dead_news = NewsService(settings.news, get=lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("injected: feeds unreachable")))   # the exception a real network failure raises
    results = []

    def scenario(name, turns, expect_normal=False, market=False, allow_stale=False, **graph_kwargs):
        """turns: list of messages sent to one session; the LAST reply is judged. A callable item is run between turns (to break something)."""
        graph = build_default_graph(settings=graph_kwargs.pop("settings", settings), llm=graph_kwargs.pop("llm", real), **graph_kwargs)
        session, state, start = f"fault-{name}", None, time.perf_counter()
        raised = None
        try:
            for turn in turns:
                if callable(turn):
                    turn()
                    continue
                state = graph.invoke({"message": turn, "session_id": session}, config_for(session))
        except Exception as error:   # the graph itself must never raise
            raised = f"{type(error).__name__}: {error}"
        seconds = time.perf_counter() - start
        text = (state or {}).get("response", {}).get("text", "") if state else ""
        disclaimer = bool((state or {}).get("response", {}).get("disclaimer")) if state else False
        checks = {"returned": raised is None and bool(text), "no_leak": not LEAK.search(text), "says_unavailable": expect_normal or bool(APOLOGY.search(text)),
                  "disclaimer": disclaimer, "fast_enough": seconds < 60,
                  "no_invented_price": (not market) or (not PRICE.search(text)) or (allow_stale and bool(STALE.search(text)))}
        graceful = all(checks.values())
        results.append({"scenario": name, "graceful": graceful, "checks": checks, "seconds": seconds, "route": (state or {}).get("route"),
                        "error_recorded": (state or {}).get("error"), "text": text, "raised": raised})
        failed = [k for k, v in checks.items() if not v]
        print(f"  {'GRACEFUL' if graceful else 'FAILED  '} {name:42} route={str((state or {}).get('route')):10} {seconds:5.1f}s" + (f"  failed checks: {failed}" if failed else ""))
        print(f"           error recorded: {(state or {}).get('error') and (state['error']['node'], state['error']['type'])}\n           reply: {text[:230]!r}")

    print("=" * 100 + "\nFAULT INJECTION THROUGH THE FULL GRAPH\n" + "=" * 100)
    scenario("market provider down from the start", ["How is Apple doing today?"], market=True,
             market_service=MarketDataService(AlwaysDown(), TTLCache(), fast, sleep=lambda s: None))
    real_provider = Breakable(YFinanceProvider(10))
    service = MarketDataService(real_provider, TTLCache(), fast)
    scenario("market provider killed MID-RUN (stale served)", ["How is Apple doing today?", lambda: setattr(real_provider, "broken", True), lambda: time.sleep(1.5),
                                                               "How is Apple doing today?"], market=True, allow_stale=True, expect_normal=True, market_service=service)
    scenario("unknown ticker", ["How is FAKEXYZ doing today?"], market=True)
    scenario("news feeds down", ["What's the latest news on Apple?"], news_service=dead_news)
    scenario("vector store broken (finance_qa)", ["What is the difference between an ETF and a mutual fund?"], retriever=BrokenRetriever())
    scenario("vector store broken (tax)", ["What is the 401(k) contribution limit?"], retriever=BrokenRetriever())
    scenario("router model unreachable", ["What is an ETF?"], router_llm=FaultyLLM(real, lambda kind, kw: True))
    scenario("answer model unreachable (router fine)", ["What is the capital gains tax rate?"],
             llm=FaultyLLM(real, lambda kind, kw: kw.get("model_key", "answer_model") != "router_model" or kw.get("temperature") is None), router_llm=real)
    scenario("goal agent model unreachable", ["I want to save $50,000 in 5 years, I have $2,000, medium risk"],
             llm=FaultyLLM(real, lambda kind, kw: True), router_llm=real)
    scenario("resolver model fails on turn 2", ["How is Apple doing today?", "What about Microsoft?"], expect_normal=True,
             llm=FaultyLLM(real, lambda kind, kw: kw.get("response_format", {}).get("json_schema", {}).get("name") == "standalone"), router_llm=real)
    scenario("oversized message (3,000 characters)", ["What is an ETF? " * 190])

    graceful = sum(r["graceful"] for r in results)
    print("\n" + "=" * 100 + f"\nGRACEFUL-DEGRADATION RATE: {graceful}/{len(results)} = {graceful / len(results):.0%}      slowest recovery: {max(r['seconds'] for r in results):.1f}s")
    out = Path(f"eval/results/faults_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
