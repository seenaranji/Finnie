"""FallbackProvider: try providers in order. It IS a MarketDataProvider, so the service cannot tell it from a single one.

Policy (the only place it lives):
  ProviderUnavailable (transient)  -> log it and try the next provider
  TickerNotFound (permanent)       -> trust it and stop: asking a second provider would spend its limited quota on a typo
  every provider failed            -> raise ProviderUnavailable
The Quote/PriceHistory `source` field says who actually served the data; `served_by` / `failed` count it for reports.
"""
from collections import Counter

from src.core.logging import get_logger
from src.data.errors import ProviderUnavailable
from src.data.providers.base import MarketDataProvider

log = get_logger(__name__)


class FallbackProvider(MarketDataProvider):
    name = "fallback"

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        self.providers = providers
        self.served_by: Counter = Counter()
        self.failed: Counter = Counter()

    def _first_that_works(self, method: str, *args):
        errors = []
        for provider in self.providers:
            try:
                result = getattr(provider, method)(*args)
            except ProviderUnavailable as error:
                self.failed[provider.name] += 1
                errors.append(f"{provider.name}: {error}")
                log.warning("%s.%s failed (%s); trying the next provider", provider.name, method, error)
                continue
            self.served_by[provider.name] += 1
            if errors:
                log.info("served by %s AFTER %s", provider.name, "; ".join(errors))
            else:
                log.info("served by %s", provider.name)
            return result
        raise ProviderUnavailable("Market data is not available right now.", " | ".join(errors))

    def get_quote(self, symbol: str):
        return self._first_that_works("get_quote", symbol)

    def get_history(self, symbol: str, period: str):
        return self._first_that_works("get_history", symbol, period)
