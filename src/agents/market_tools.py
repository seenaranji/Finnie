"""The tools the Market Analysis Agent may call: JSON schemas (what the model sees) + code (what actually runs).

The model sees ONLY the name, description and argument schema. It never sees our code. Every tool:
  - validates its arguments (the model can send anything)
  - never raises: a failure becomes {"error": ...} so the model can tell the user, instead of crashing the conversation
  - returns numbers already computed in Python, each result stamped with as_of / source / is_stale
"""
import time

from src.core.logging import get_logger
from src.data import stats
from src.data.errors import MarketDataError, TickerNotFound
from src.data.market_data import MarketDataService

log = get_logger(__name__)
PERIODS = ["5d", "1mo", "3mo", "6mo", "1y"]

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_quote",
        "description": "Latest price, previous close and daily % change for ONE ticker symbol (e.g. AAPL, VOO). Includes the date of the price (as_of), the data source and whether it is stale.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}},
                       "required": ["symbol"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Summary of daily closing prices for ONE ticker over a period: first and last date and close, highest and lowest close, and the last 5 closes.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "period": {"type": "string", "enum": PERIODS}},
                       "required": ["symbol", "period"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Computed statistics for ONE ticker: price vs its 50-day and 200-day moving averages, 52-week low/high (closing prices), and annualized volatility. A value is null when there is not enough data.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False}}},
]


def _error(error: MarketDataError) -> dict:
    return {"error": error.user_message, "kind": "ticker_not_found" if isinstance(error, TickerNotFound) else "unavailable"}


def _freshness(item) -> dict:
    return {"as_of": getattr(item, "as_of", None), "source": item.source, "is_stale": item.is_stale,
            "fetched_at": item.fetched_at.isoformat(timespec="seconds"), "age_seconds": item.age_seconds}


class MarketTools:
    def __init__(self, service: MarketDataService) -> None:
        self.service = service
        self.handlers = {"get_quote": self.get_quote, "get_history": self.get_history, "get_stats": self.get_stats}

    def run(self, name: str, arguments: dict | None) -> tuple[dict, dict]:
        """Returns (result for the model, trace entry for us). Never raises."""
        cache = self.service.cache
        hits_before, misses_before = cache.hits, cache.misses
        start = time.perf_counter()
        handler = self.handlers.get(name)
        if handler is None:
            result = {"error": f"unknown tool {name!r}", "kind": "bad_request"}
        elif not isinstance(arguments, dict) or not isinstance(arguments.get("symbol"), str):
            result = {"error": "arguments must be a JSON object with a string 'symbol'", "kind": "bad_request"}
        else:
            try:
                result = handler(**arguments)
            except MarketDataError as error:
                result = _error(error)
            except TypeError as error:   # the model sent an argument we do not accept
                result = {"error": f"bad arguments: {error}", "kind": "bad_request"}
        trace = {"tool": name, "arguments": arguments, "result": result, "seconds": round(time.perf_counter() - start, 4),
                 "cache_hits": cache.hits - hits_before, "cache_misses": cache.misses - misses_before}
        log.info("tool %s(%s) -> %s in %.3fs (cache hits %d, misses %d)", name, arguments,
                 "error" if "error" in result else "ok", trace["seconds"], trace["cache_hits"], trace["cache_misses"])
        return result, trace

    def get_quote(self, symbol: str) -> dict:
        quote = self.service.get_quote(symbol)
        return {"symbol": quote.symbol, "price": quote.price, "previous_close": quote.previous_close, "change_pct": quote.change_pct,
                "currency": quote.currency, **_freshness(quote)}

    def get_history(self, symbol: str, period: str = "1mo") -> dict:
        if period not in PERIODS:
            return {"error": f"period must be one of {PERIODS}", "kind": "bad_request"}
        history = self.service.get_history(symbol, period)
        closes = [p.close for p in history.points]
        return {"symbol": history.symbol, "period": history.period, "points": len(closes),
                "first_date": history.points[0].date, "first_close": closes[0], "last_date": history.points[-1].date, "last_close": closes[-1],
                "highest_close": max(closes), "lowest_close": min(closes), "last_5_closes": closes[-5:], **_freshness(history)}

    def get_stats(self, symbol: str) -> dict:
        quote = self.service.get_quote(symbol)
        notes, result = [], {"symbol": quote.symbol, "price": quote.price, "change_pct": quote.change_pct, **_freshness(quote)}
        try:
            history = self.service.get_history(symbol, "1y")
        except MarketDataError:
            history = None
            notes.append("price history unavailable, so averages, range and volatility are null")
        closes = [p.close for p in history.points] if history else []
        ma50, ma200 = stats.moving_average(closes, 50), stats.moving_average(closes, 200)
        result.update({"ma50": ma50, "price_vs_ma50_pct": stats.price_vs_average_pct(quote.price, ma50),
                       "ma200": ma200, "price_vs_ma200_pct": stats.price_vs_average_pct(quote.price, ma200),
                       "range_52w": stats.range_52w(closes), "volatility_annual_pct": stats.annualized_volatility_pct(closes),
                       "history_bars": len(closes)})
        if history and ma200 is None:
            notes.append(f"200-day average unavailable: only {len(closes)} daily closes")
        if history and (result["range_52w"] is None or not result["range_52w"]["covers_full_year"]):
            notes.append(f"52-week range uses only {len(closes)} daily closes (less than a year)")
        result["notes"] = notes
        return result
