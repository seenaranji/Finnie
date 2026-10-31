"""Alpha Vantage provider (free key: 25 requests/day, 1/second; free quotes are end-of-day, real-time is premium).

What the real API does (checked 2026-10-31):
  - it answers HTTP 200 for EVERYTHING, so errors must be read from the body
  - over the limit  -> {"Information": "...rate limit..."}   (older docs say "Note")  -> ProviderUnavailable
  - unknown symbol  -> GLOBAL_QUOTE returns {"Global Quote": {}}                     -> TickerNotFound
  - TIME_SERIES_DAILY with a bad symbol is documented to return {"Error Message": ...} (not yet observed here)
  - free history is the latest 100 daily bars ("compact"); "full" is premium
"""
import requests

from src.core.logging import get_logger
from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.models import PriceHistory, PricePoint, Quote
from src.data.providers.base import MarketDataProvider
from src.data.rate_limiter import CallBudget

log = get_logger(__name__)
_UNAVAILABLE = "The backup market data service is not responding right now."
_PERIOD_DAYS = {"5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504}


class AlphaVantageProvider(MarketDataProvider):
    name = "alpha_vantage"

    def __init__(self, api_key: str, budget: CallBudget, base_url: str = "https://www.alphavantage.co/query",
                 timeout_seconds: float = 10, http=None) -> None:
        self._key, self.budget, self.base_url, self.timeout = api_key, budget, base_url, timeout_seconds
        self.http = http or requests

    def _call(self, **params) -> dict:
        self.budget.acquire()   # may wait (spacing) or refuse (daily cap) BEFORE any network call
        try:
            response = self.http.get(self.base_url, params={**params, "apikey": self._key}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as error:   # timeout, DNS, HTTP 5xx, body that is not JSON
            raise ProviderUnavailable(_UNAVAILABLE, f"{type(error).__name__}: {error}".replace(self._key, "***")) from error
        if not isinstance(data, dict):
            raise ProviderUnavailable(_UNAVAILABLE, f"unexpected payload type {type(data).__name__}")
        notice = data.get("Information") or data.get("Note")
        if notice:   # HTTP 200 but a rate-limit / plan message
            log.warning("alpha vantage notice: %s", str(notice)[:120])
            raise ProviderUnavailable(_UNAVAILABLE, f"alpha vantage notice: {str(notice)[:120]}")
        if "Error Message" in data:
            message = str(data["Error Message"])
            if "apikey" in message.lower():   # our key is wrong: not the user's fault and not "ticker not found"
                raise ProviderUnavailable(_UNAVAILABLE, "alpha vantage rejected the API key")
            raise TickerNotFound(f"I couldn't find a ticker called {params.get('symbol')}.", message[:120])
        return data

    def get_quote(self, symbol: str) -> Quote:
        data = self._call(function="GLOBAL_QUOTE", symbol=symbol)
        raw = data.get("Global Quote")
        if raw == {}:
            raise TickerNotFound(f"I couldn't find a ticker called {symbol}.", "empty Global Quote")
        try:
            price, previous = float(raw["05. price"]), float(raw["08. previous close"])
            change = round(float(raw["10. change percent"].rstrip("%")), 2)
            as_of = raw["07. latest trading day"]
        except (TypeError, KeyError, ValueError) as error:
            raise ProviderUnavailable(_UNAVAILABLE, f"malformed GLOBAL_QUOTE payload: {error!r}") from error
        return Quote(symbol=symbol, price=round(price, 4), previous_close=round(previous, 4), change_pct=change,
                     source=self.name, as_of=as_of)

    def get_history(self, symbol: str, period: str) -> PriceHistory:
        data = self._call(function="TIME_SERIES_DAILY", symbol=symbol)
        try:
            series = data["Time Series (Daily)"]
            ordered = sorted((day, float(bar["4. close"])) for day, bar in series.items())
        except (TypeError, KeyError, ValueError, AttributeError) as error:
            raise ProviderUnavailable(_UNAVAILABLE, f"malformed TIME_SERIES_DAILY payload: {error!r}") from error
        wanted = _PERIOD_DAYS.get(period, 21)
        kept = ordered[-wanted:]
        label = period if len(kept) >= wanted else f"{len(kept)}d"   # free tier has only ~100 bars: say what we really cover
        return PriceHistory(symbol=symbol, period=label, points=[PricePoint(date=d, close=round(c, 4)) for d, c in kept],
                            source=self.name)
