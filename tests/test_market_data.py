import pytest

from src.core.config import load_settings
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.market_data import MarketDataService
from src.data.models import PriceHistory, Quote
from src.data.providers.base import MarketDataProvider


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, failures=()):
        self.failures = list(failures)   # exceptions to raise on the next calls, in order; then succeed
        self.calls = 0

    def get_quote(self, symbol):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return Quote(symbol=symbol, price=100.0 + self.calls, source=self.name)

    def get_history(self, symbol, period):
        self.calls += 1
        return PriceHistory(symbol=symbol, period=period, points=[], source=self.name)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_service(provider, clock=None):
    clock = clock or Clock()
    sleeps = []
    service = MarketDataService(provider, TTLCache(clock), load_settings(), sleep=sleeps.append)
    return service, clock, sleeps


def down():
    return ProviderUnavailable("down")


def test_second_call_within_ttl_is_a_cache_hit():
    provider = FakeProvider()
    service, clock, _ = make_service(provider)
    first = service.get_quote("AAPL")
    clock.now = 59
    second = service.get_quote("aapl ")   # normalised to the same key
    assert provider.calls == 1 and first.price == second.price
    assert (service.cache.hits, service.cache.misses) == (1, 1)


def test_call_after_ttl_fetches_again():
    provider = FakeProvider()
    service, clock, _ = make_service(provider)
    service.get_quote("AAPL")
    clock.now = 61
    assert service.get_quote("AAPL").price == 102.0 and provider.calls == 2


def test_transient_errors_are_retried_with_growing_waits():
    provider = FakeProvider([down(), down()])
    service, _, sleeps = make_service(provider)
    assert service.get_quote("AAPL").is_stale is False
    assert provider.calls == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9   # backoff grows (jitter allowed)


def test_unknown_ticker_is_not_retried():
    provider = FakeProvider([TickerNotFound("nope")])
    service, _, sleeps = make_service(provider)
    with pytest.raises(TickerNotFound):
        service.get_quote("ZZZZ")
    assert provider.calls == 1 and sleeps == []


def test_provider_down_with_cached_value_returns_it_flagged_stale():
    provider = FakeProvider()
    service, clock, _ = make_service(provider)
    service.get_quote("AAPL")
    clock.now = 500                               # TTL long expired
    provider.failures = [down(), down(), down()]  # all 3 attempts fail
    quote = service.get_quote("AAPL")
    assert quote.is_stale and quote.price == 101.0


def test_provider_down_with_nothing_cached_raises_a_clean_error():
    service, _, _ = make_service(FakeProvider([down()] * 3))
    with pytest.raises(ProviderUnavailable) as error:
        service.get_quote("AAPL")
    assert "not responding" in error.value.user_message or error.value.user_message == "down"


@pytest.mark.parametrize("bad", ["", "   ", "DROP TABLE", "A" * 30, "AA$L"])
def test_invalid_symbols_are_rejected_without_a_network_call(bad):
    provider = FakeProvider()
    service, _, _ = make_service(provider)
    with pytest.raises(TickerNotFound):
        service.get_quote(bad)
    assert provider.calls == 0


def test_history_is_cached_per_period():
    provider = FakeProvider()
    service, _, _ = make_service(provider)
    service.get_history("AAPL", "1mo")
    service.get_history("AAPL", "1mo")
    service.get_history("AAPL", "1y")
    assert provider.calls == 2
