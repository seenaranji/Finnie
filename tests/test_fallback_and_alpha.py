import json

import pytest

from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.models import PriceHistory, Quote
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.base import MarketDataProvider
from src.data.providers.fallback import FallbackProvider
from src.data.rate_limiter import CallBudget


# ---------- fakes ----------
class Scripted(MarketDataProvider):
    def __init__(self, name, error=None):
        self.name, self.error, self.calls = name, error, 0

    def get_quote(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return Quote(symbol=symbol, price=1.0, source=self.name)

    def get_history(self, symbol, period):
        self.calls += 1
        if self.error:
            raise self.error
        return PriceHistory(symbol=symbol, period=period, points=[], source=self.name)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeHttp:
    def __init__(self, payload, status=200, raises=None):
        self.payload, self.status, self.raises, self.calls = payload, status, raises, []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if self.raises:
            raise self.raises
        return FakeResponse(self.payload, self.status)


def make_alpha(http, per_day=25):
    budget = CallBudget(per_day, 0, sleep=lambda s: None)
    return AlphaVantageProvider("SECRETKEY123", budget, http=http)


GOOD_QUOTE = {"Global Quote": {"01. symbol": "AAPL", "05. price": "344.6300", "07. latest trading day": "2026-10-30",
                               "08. previous close": "348.9100", "10. change percent": "-1.2267%"}}
down = lambda: ProviderUnavailable("down")


# ---------- FallbackProvider ----------
def test_primary_success_never_touches_the_second():
    first, second = Scripted("a"), Scripted("b")
    assert FallbackProvider([first, second]).get_quote("X").source == "a" and second.calls == 0


def test_transient_failure_falls_back_to_the_second():
    fallback = FallbackProvider([Scripted("a", down()), Scripted("b")])
    assert fallback.get_quote("X").source == "b"
    assert fallback.served_by["b"] == 1 and fallback.failed["a"] == 1


def test_ticker_not_found_is_trusted_and_not_retried_elsewhere():
    first, second = Scripted("a", TickerNotFound("nope")), Scripted("b")
    with pytest.raises(TickerNotFound):
        FallbackProvider([first, second]).get_quote("ZZZ")
    assert second.calls == 0


def test_all_providers_down_raises_a_clean_error():
    with pytest.raises(ProviderUnavailable) as error:
        FallbackProvider([Scripted("a", down()), Scripted("b", down())]).get_history("X", "1mo")
    assert error.value.user_message == "Market data is not available right now."


# ---------- AlphaVantageProvider ----------
def test_quote_is_parsed_into_the_shared_quote_shape():
    quote = make_alpha(FakeHttp(GOOD_QUOTE)).get_quote("AAPL")
    assert (quote.price, quote.previous_close, quote.change_pct, quote.as_of, quote.source) == \
           (344.63, 348.91, -1.23, "2026-10-30", "alpha_vantage")


def test_http_200_with_a_rate_limit_message_is_a_transient_error():
    with pytest.raises(ProviderUnavailable):
        make_alpha(FakeHttp({"Information": "Thank you ... 25 requests per day ..."})).get_quote("AAPL")


def test_empty_global_quote_means_unknown_ticker():
    with pytest.raises(TickerNotFound):
        make_alpha(FakeHttp({"Global Quote": {}})).get_quote("FAKEXYZ")


def test_error_message_about_the_key_is_not_reported_as_unknown_ticker():
    with pytest.raises(ProviderUnavailable):
        make_alpha(FakeHttp({"Error Message": "the parameter apikey is invalid or missing"})).get_quote("AAPL")


@pytest.mark.parametrize("payload", [{"Global Quote": {"05. price": "abc"}}, {"Global Quote": None}, ["not", "a", "dict"], {}])
def test_malformed_payloads_become_clean_errors(payload):
    with pytest.raises(ProviderUnavailable):
        make_alpha(FakeHttp(payload)).get_quote("AAPL")


def test_network_error_and_http_500_become_provider_unavailable_without_leaking_the_key():
    with pytest.raises(ProviderUnavailable) as error:
        make_alpha(FakeHttp(None, raises=ConnectionError("failed for apikey=SECRETKEY123"))).get_quote("AAPL")
    assert "SECRETKEY123" not in str(error.value)
    with pytest.raises(ProviderUnavailable):
        make_alpha(FakeHttp(GOOD_QUOTE, status=500)).get_quote("AAPL")


def test_history_is_sorted_oldest_first_and_labelled_with_what_it_really_covers():
    series = {f"2026-10-{day:02d}": {"4. close": str(100 + day)} for day in range(1, 11)}
    history = make_alpha(FakeHttp({"Time Series (Daily)": series})).get_history("AAPL", "1y")
    assert [p.date for p in history.points] == sorted(series) and history.period == "10d"   # asked 1y, only 10 bars exist
    assert make_alpha(FakeHttp({"Time Series (Daily)": series})).get_history("AAPL", "5d").period == "5d"


# ---------- CallBudget ----------
def test_daily_cap_blocks_locally_with_no_network_call():
    http = FakeHttp(GOOD_QUOTE)
    alpha = make_alpha(http, per_day=2)
    alpha.get_quote("AAPL"), alpha.get_quote("AAPL")
    with pytest.raises(ProviderUnavailable):
        alpha.get_quote("AAPL")
    assert len(http.calls) == 2


def test_calls_are_spaced_by_the_minimum_interval():
    now, slept = [0.0], []
    budget = CallBudget(25, 1.1, clock=lambda: now[0], sleep=slept.append)
    budget.acquire()
    now[0] = 0.3
    budget.acquire()
    assert slept == [pytest.approx(0.8)]


def test_daily_count_survives_a_restart_and_resets_on_a_new_day(tmp_path):
    state = tmp_path / "usage.json"
    day = ["2026-10-31"]
    first = CallBudget(25, 0, state, sleep=lambda s: None, today=lambda: day[0])
    first.acquire(), first.acquire()
    assert CallBudget(25, 0, state, today=lambda: day[0]).used_today == 2   # new process, same day
    day[0] = "2026-11-01"
    assert CallBudget(25, 0, state, today=lambda: day[0]).used_today == 0
    assert json.loads(state.read_text())["count"] == 2
