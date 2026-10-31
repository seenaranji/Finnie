"""Step 11: market data layer. Real calls first, then measurements, then deliberate faults (offline, scripted).

Run:  .venv/bin/python -m scripts.step11_market_demo
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

import src.data.providers.yfinance_provider as yf_module
from src.core.config import load_settings
from src.core.logging import setup_logging
from src.core.metrics import summarize, timer
from src.data.cache import TTLCache
from src.data.errors import MarketDataError, ProviderUnavailable
from src.data.market_data import MarketDataService
from src.data.providers.yfinance_provider import YFinanceProvider

setup_logging("INFO")
settings = load_settings()
report: dict = {}


def show(label, call):
    with timer() as timing:
        try:
            result = call()
            outcome = result.model_dump(include={"symbol", "price", "change_pct", "source", "is_stale", "age_seconds"})
        except MarketDataError as error:
            outcome = f"{type(error).__name__}: {error.user_message}"
    print(f"  {label:34} {timing['seconds']:7.3f}s  {outcome}")
    return timing["seconds"]


print("=" * 100, "\nA. REAL CALLS: cold vs warm, fake ticker, invalid input\n" + "=" * 100)
service = MarketDataService(YFinanceProvider(settings.market_data.timeout_seconds), TTLCache(), settings)
cold = show("AAPL (cold)", lambda: service.get_quote("AAPL"))
warm = show("AAPL (warm, within 60 s)", lambda: service.get_quote("AAPL"))
show("AAPL history 1mo (cold)", lambda: service.get_history("AAPL", "1mo"))
show("FAKEXYZ (unknown ticker)", lambda: service.get_quote("FAKEXYZ"))
show("'DROP TABLE' (invalid input)", lambda: service.get_quote("DROP TABLE"))
report["cold_vs_warm_s"] = {"cold": cold, "warm": warm}


print("\n" + "=" * 100, "\nB. NETWORK FAILURE: provider dies after one good fetch, TTL set to 1 s\n" + "=" * 100)
class Flaky(YFinanceProvider):
    fail = False

    def _bars(self, symbol, period):
        if self.fail:
            raise ProviderUnavailable("The market data service is not responding right now.", "simulated network failure")
        return super()._bars(symbol, period)


settings.cache.quote_ttl_seconds = 1
settings.market_data.retry_base_seconds = 0.2
flaky = Flaky(settings.market_data.timeout_seconds)
service_b = MarketDataService(flaky, TTLCache(), settings)
show("AAPL (good fetch)", lambda: service_b.get_quote("AAPL"))
flaky.fail = True
time.sleep(1.2)
show("AAPL (provider down, TTL expired)", lambda: service_b.get_quote("AAPL"))
show("MSFT (provider down, never cached)", lambda: service_b.get_quote("MSFT"))


print("\n" + "=" * 100, "\nC. CACHE HIT RATE: 50 mixed requests (8 tickers, 1 fake), TTL 60 s\n" + "=" * 100)
settings.cache.quote_ttl_seconds = 60
service_c = MarketDataService(YFinanceProvider(settings.market_data.timeout_seconds), TTLCache(), settings)
random.seed(7)
tickers = ["AAPL", "MSFT", "VOO", "SPY", "BND", "NVDA", "TSLA", "FAKEXYZ"]
weights = [5, 4, 3, 3, 2, 2, 1, 1]   # a few popular tickers get asked about most
hit_times, miss_times = [], []
for symbol in random.choices(tickers, weights, k=50):
    hits_before = service_c.cache.hits
    with timer() as timing:
        try:
            service_c.get_quote(symbol)
        except MarketDataError:
            pass
    (hit_times if service_c.cache.hits > hits_before else miss_times).append(timing["seconds"])
cache = service_c.cache
print(f"  requests=50  hits={cache.hits}  misses={cache.misses}  hit rate={cache.hit_rate:.0%}")
print(f"  latency on a MISS (network): {summarize(miss_times)}")
print(f"  latency on a HIT  (cache):   {summarize(hit_times)}")
report["hit_rate"] = {"hits": cache.hits, "misses": cache.misses, "rate": cache.hit_rate,
                      "miss_latency": summarize(miss_times), "hit_latency": summarize(hit_times)}


print("\n" + "=" * 100, "\nD. FAULT INJECTION (offline, scripted): graceful-degradation rate (spec 6.5)\n" + "=" * 100)
GOOD = pd.DataFrame({"Close": [100.0, 101.5]}, index=pd.to_datetime(["2026-10-29", "2026-10-30"]))


class ScriptedTicker:
    behaviour = "ok"

    def __init__(self, symbol):
        pass

    def history(self, period, timeout):
        kind = ScriptedTicker.behaviour
        if kind == "timeout":
            raise requests.exceptions.ReadTimeout("read timed out")
        if kind == "429":
            raise requests.exceptions.HTTPError("429 Too Many Requests")
        if kind == "offline":
            raise requests.exceptions.ConnectionError("Failed to resolve query1.finance.yahoo.com")
        if kind == "malformed (no Close column)":
            return pd.DataFrame({"Open": [1.0, 2.0]})
        if kind == "malformed (not a table)":
            return None
        if kind == "invalid ticker":
            return pd.DataFrame()
        return GOOD


class Clock:
    now = 0.0

    def __call__(self):
        return Clock.now


# (scenario, transient?)  transient faults must serve a flagged stale value when one exists; permanent ones must say so clearly
SCENARIOS = [("timeout", True), ("429", True), ("malformed (no Close column)", True), ("malformed (not a table)", True),
             ("offline", True), ("invalid ticker", False)]
settings.market_data.retry_base_seconds = 0.05
outcomes = []
with patch.object(yf_module.yf, "Ticker", ScriptedTicker):
    for name, transient in SCENARIOS:
        for warm in (False, True):
            Clock.now = 0.0
            service_d = MarketDataService(YFinanceProvider(1), TTLCache(Clock()), settings)
            if warm:   # prime the cache with a good value, then let the TTL expire
                ScriptedTicker.behaviour = "ok"
                service_d.get_quote("AAPL")
                Clock.now = 999
            ScriptedTicker.behaviour = name
            start = time.perf_counter()
            try:
                quote = service_d.get_quote("AAPL")
                result = f"stale value ({quote.price}), is_stale={quote.is_stale}, age {quote.age_seconds}s"
                good = warm and transient and quote.is_stale
            except MarketDataError as error:
                result = f"clean error: {error.user_message!r}"
                good = (not warm) or (not transient)
            except Exception as error:   # a raw exception reaching the caller = a stack trace for the user = failure
                result = f"RAW EXCEPTION {type(error).__name__}: {error}"
                good = False
            seconds = time.perf_counter() - start
            outcomes.append({"scenario": name, "cache": "warm" if warm else "cold", "good": good, "result": result, "seconds": seconds})
            print(f"  {'OK  ' if good else 'FAIL'} {name:30} cache={'warm' if warm else 'cold'}  {seconds:5.2f}s  {result}")
rate = sum(o["good"] for o in outcomes) / len(outcomes)
print(f"\n  graceful-degradation rate = {sum(o['good'] for o in outcomes)}/{len(outcomes)} = {rate:.0%}")
report["fault_injection"] = {"rate": rate, "outcomes": outcomes}

output_file = Path(f"benchmarks/results/step11_market_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {output_file}")
