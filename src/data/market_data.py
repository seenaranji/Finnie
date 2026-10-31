"""MarketDataService: the one door agents use to get prices. No LLM anywhere in here.

get_quote / get_history flow:
  fresh cache hit -> return it (no network)
  else ask the provider, retrying TRANSIENT failures with exponential backoff + jitter
       success -> cache it, return it
       provider still failing -> last saved value flagged is_stale=True (with its age), or a clear MarketDataError
  unknown ticker -> TickerNotFound straight away (permanent: no retry, no stale value)
"""
import re
import time
from typing import Callable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.config import Settings, load_settings
from src.core.logging import get_logger
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.models import PriceHistory, Quote, utc_now
from src.data.providers.base import MarketDataProvider

log = get_logger(__name__)
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")   # AAPL, BRK-B, BRK.B; rejects "", "DROP TABLE", "A"*50


class MarketDataService:
    def __init__(self, provider: MarketDataProvider, cache: TTLCache | None = None, settings: Settings | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        settings = settings or load_settings()
        self.provider = provider
        self.cache = cache or TTLCache()
        self.quote_ttl = settings.cache.quote_ttl_seconds
        self.history_ttl = settings.cache.history_ttl_seconds
        self.max_attempts = settings.market_data.max_attempts
        self.retry_base = settings.market_data.retry_base_seconds
        self.sleep = sleep   # injectable so tests do not really wait

    def get_quote(self, symbol: str) -> Quote:
        symbol = self._clean(symbol)
        return self._get(f"quote:{symbol}", self.quote_ttl, lambda: self.provider.get_quote(symbol))

    def get_history(self, symbol: str, period: str = "1mo") -> PriceHistory:
        symbol = self._clean(symbol)
        return self._get(f"history:{symbol}:{period}", self.history_ttl, lambda: self.provider.get_history(symbol, period))

    def _clean(self, symbol: str) -> str:
        cleaned = (symbol or "").strip().upper()
        if not _SYMBOL.match(cleaned):
            raise TickerNotFound(f"'{symbol}' doesn't look like a valid ticker symbol.", f"rejected symbol {symbol!r}")
        return cleaned

    def _get(self, key: str, ttl: float, fetch):
        cached = self.cache.get(key)
        if cached is not None:
            log.info("cache HIT %s", key)
            return self._aged(cached)
        log.info("cache MISS %s", key)
        try:
            value = self._fetch_with_retry(fetch)
        except ProviderUnavailable as error:
            saved = self.cache.get_stale(key)
            if saved is None:
                log.error("provider failed and nothing cached for %s: %s", key, error)
                raise
            value, age = saved
            log.warning("provider failed (%s); serving STALE %s, %.0fs old", error, key, age)
            return self._aged(value.model_copy(update={"is_stale": True}))
        self.cache.set(key, value, ttl)
        return self._aged(value)

    def _fetch_with_retry(self, fetch):
        def log_retry(state) -> None:
            log.warning("transient provider error (%s); retry %d in %.1fs",
                        state.outcome.exception(), state.attempt_number, state.next_action.sleep)
        retrying = Retrying(retry=retry_if_exception_type(ProviderUnavailable), stop=stop_after_attempt(self.max_attempts),
                            wait=wait_exponential_jitter(multiplier=self.retry_base, jitter=self.retry_base, max=8), sleep=self.sleep,
                            before_sleep=log_retry, reraise=True)   # TickerNotFound is not in the retry list: it passes straight through
        return retrying(fetch)

    @staticmethod
    def _aged(value):
        return value.model_copy(update={"age_seconds": round((utc_now() - value.fetched_at).total_seconds(), 1)})
