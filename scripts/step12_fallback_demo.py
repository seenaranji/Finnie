"""Step 12: Alpha Vantage + FallbackProvider.

  A. normal: yfinance serves (baseline latency)
  B. yfinance disabled on purpose: Alpha Vantage serves (REAL calls; spends about 4 of the 25 daily requests)
  C. injected failures, offline (scripted): does the chain degrade gracefully?
  D. Alpha Vantage calls vs its limits

Run:  .venv/bin/python -m scripts.step12_fallback_demo
"""
import json
import logging
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
from src.data.factory import build_providers
from src.data.market_data import MarketDataService
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.fallback import FallbackProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.rate_limiter import CallBudget

setup_logging("INFO")
settings = load_settings()
report: dict = {}


def fetch(label, call):
    with timer() as timing:
        try:
            quote = call()
            shown = quote.model_dump(include={"symbol", "price", "change_pct", "as_of", "source", "is_stale"})
        except MarketDataError as error:
            quote, shown = None, f"{type(error).__name__}: {error.user_message}"
    print(f"  {label:26} {timing['seconds']:6.2f}s  {shown}")
    return quote, timing["seconds"]


print("=" * 100, "\nA. NORMAL: yfinance only (baseline latency, 3 cold quotes)\n" + "=" * 100)
yf_service = MarketDataService(YFinanceProvider(settings.market_data.timeout_seconds), TTLCache(), settings)
normal = [fetch(symbol, lambda s=symbol: yf_service.get_quote(s)) for symbol in ("MSFT", "VOO", "SPY")]
normal_times = [t for _, t in normal]


print("\n" + "=" * 100, "\nB. YFINANCE DISABLED ON PURPOSE: the chain falls back to Alpha Vantage (REAL calls)\n" + "=" * 100)
class Disabled(YFinanceProvider):
    def _bars(self, symbol, period):
        raise ProviderUnavailable("The market data service is not responding right now.", "yfinance disabled on purpose")


yf_real, alpha = build_providers(settings)
chain = FallbackProvider([Disabled(settings.market_data.timeout_seconds), alpha])
service_b = MarketDataService(chain, TTLCache(), settings)
fallback = [fetch(symbol, lambda s=symbol: service_b.get_quote(s)) for symbol in ("MSFT", "VOO", "SPY")]
history, history_time = None, 0.0
with timer() as timing:
    history = service_b.get_history("AAPL", "1mo")
print(f"  {'AAPL history 1mo':26} {timing['seconds']:6.2f}s  source={history.source} period={history.period} points={len(history.points)} "
      f"first={history.points[0].date} last={history.points[-1].date}")
fallback_times = [t for _, t in fallback]
print(f"\n  served_by={dict(chain.served_by)}  failed={dict(chain.failed)}")

same_shape = set(normal[0][0].model_dump()) == set(fallback[0][0].model_dump())
print(f"  same Quote fields from both providers: {same_shape}   (yfinance as_of={normal[0][0].as_of}, alpha as_of={fallback[0][0].as_of})")
print(f"  latency, yfinance only : {summarize(normal_times)}")
print(f"  latency, via fallback  : {summarize(fallback_times)}   (includes the 1.1 s spacing wait between Alpha calls)")
report["live"] = {"normal_s": normal_times, "fallback_s": fallback_times, "served_by": dict(chain.served_by)}


print("\n" + "=" * 100, "\nC. INJECTED FAILURES, offline: yfinance fault x Alpha Vantage state (graceful = served by Alpha, or a clean error)\n" + "=" * 100)
GOOD_FRAME = pd.DataFrame({"Close": [100.0, 101.5]}, index=pd.to_datetime(["2026-10-29", "2026-10-30"]))
GOOD_ALPHA = {"Global Quote": {"05. price": "344.63", "07. latest trading day": "2026-10-30", "08. previous close": "348.91",
                               "10. change percent": "-1.2267%"}}


class ScriptedTicker:
    fault = "ok"

    def __init__(self, symbol):
        pass

    def history(self, period, timeout):
        if ScriptedTicker.fault == "timeout":
            raise requests.exceptions.ReadTimeout("read timed out")
        if ScriptedTicker.fault == "429":
            raise requests.exceptions.HTTPError("429 Too Many Requests")
        if ScriptedTicker.fault == "offline":
            raise requests.exceptions.ConnectionError("Failed to resolve host")
        if ScriptedTicker.fault == "malformed":
            return pd.DataFrame({"Open": [1.0]})
        if ScriptedTicker.fault == "unknown ticker":
            return pd.DataFrame()
        return GOOD_FRAME


class ScriptedHttp:
    def __init__(self, payload=None, raises=None):
        self.payload, self.raises, self.calls = payload, raises, 0

    def get(self, url, params, timeout):
        self.calls += 1
        if self.raises:
            raise self.raises

        class Response:
            status_code = 200

            def raise_for_status(self_):
                pass

            def json(self_):
                return self.payload
        return Response()


ALPHA_STATES = {"healthy": lambda: ScriptedHttp(GOOD_ALPHA),
                "rate limited": lambda: ScriptedHttp({"Information": "Thank you ... 25 requests per day ..."}),
                "malformed": lambda: ScriptedHttp({"Global Quote": {"05. price": "abc"}}),
                "network down": lambda: ScriptedHttp(raises=requests.exceptions.ConnectionError("offline"))}
settings.market_data.retry_base_seconds = 0.05
logging.getLogger("src.data").setLevel(logging.ERROR)   # 20+ scenarios x several log lines would bury the table
outcomes = []
with patch.object(yf_module.yf, "Ticker", ScriptedTicker):
    for yfinance_fault in ("timeout", "429", "offline", "malformed", "unknown ticker"):
        for alpha_state, make_http in ALPHA_STATES.items():
            http = make_http()
            budget = CallBudget(25, 0, sleep=lambda s: None)
            chain_c = FallbackProvider([YFinanceProvider(1), AlphaVantageProvider("KEY", budget, http=http)])
            service_c = MarketDataService(chain_c, TTLCache(), settings)
            ScriptedTicker.fault = yfinance_fault
            start = time.perf_counter()
            try:
                quote = service_c.get_quote("AAPL")
                result, raw_exception = f"served by {quote.source}", False
            except MarketDataError as error:
                result, raw_exception = f"clean {type(error).__name__}: {error.user_message!r}", False
            except Exception as error:
                result, raw_exception = f"RAW EXCEPTION {type(error).__name__}", True
            seconds = time.perf_counter() - start
            if yfinance_fault == "unknown ticker":   # permanent: must NOT spend an Alpha call, must say so clearly
                good = (not raw_exception) and http.calls == 0 and "TickerNotFound" in result
            elif alpha_state == "healthy":           # primary transient failure + healthy backup: must be served by the backup
                good = result == "served by alpha_vantage"
            else:                                    # both down: a clean error, never a raw exception
                good = (not raw_exception) and result.startswith("clean ProviderUnavailable")
            outcomes.append({"yfinance": yfinance_fault, "alpha": alpha_state, "good": good, "result": result,
                             "alpha_calls": http.calls, "seconds": seconds})
            print(f"  {'OK  ' if good else 'FAIL'} yfinance={yfinance_fault:15} alpha={alpha_state:13} alpha_calls={http.calls}  {seconds:5.2f}s  {result}")
good_count = sum(o["good"] for o in outcomes)
backup_cases = [o for o in outcomes if o["yfinance"] != "unknown ticker" and o["alpha"] == "healthy"]
print(f"\n  graceful-degradation rate = {good_count}/{len(outcomes)} = {good_count / len(outcomes):.0%}")
print(f"  fallback success rate (yfinance transient fault + healthy Alpha) = {sum(o['good'] for o in backup_cases)}/{len(backup_cases)}")
report["fault_injection"] = {"rate": good_count / len(outcomes), "outcomes": outcomes}


print("\n" + "=" * 100, "\nD. ALPHA VANTAGE CALLS vs LIMITS (the real budget used in section B)\n" + "=" * 100)
budget = alpha.budget
print(f"  calls in the last 60 s : {budget.calls_in_last_minute()}   (spacing guard: at most 1 per {settings.alpha_vantage.min_interval_seconds} s = {60 / settings.alpha_vantage.min_interval_seconds:.0f}/min; provider limit 1/s)")
print(f"  calls today (our count): {budget.used_today} of our cap {settings.alpha_vantage.max_calls_per_day} (provider free limit 25/day; manual probes today are NOT in this count)")
report["alpha_budget"] = {"last_minute": budget.calls_in_last_minute(), "used_today": budget.used_today,
                          "cap": settings.alpha_vantage.max_calls_per_day}

output_file = Path(f"benchmarks/results/step12_fallback_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {output_file}")
