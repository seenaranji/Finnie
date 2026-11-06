"""RedisTTLCache against fakeredis: same behaviour as TTLCache, plus the banking-domain guarantees."""
import json

import fakeredis
import pytest
import redis

from src.core.config import load_settings
from src.data.factory import build_cache
from src.data.cache import TTLCache
from src.data.models import Quote
from src.data.redis_cache import KEY_NAMESPACE, RedisTTLCache


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def make(grace=3600):
    clock = Clock()
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisTTLCache(client, grace, clock), clock, client


def quote(price=100.0):
    return Quote(symbol="AAPL", price=price, source="test")


def test_roundtrip_fresh_then_expired():
    c, clock, _ = make()
    c.set("quote:AAPL", quote(), ttl=60)
    assert c.get("quote:AAPL").price == 100.0 and c.hits == 1
    clock.now += 60
    assert c.get("quote:AAPL") is None and c.misses == 1


def test_expired_value_is_still_available_as_stale_with_its_age():
    c, clock, _ = make()
    c.set("quote:AAPL", quote(), ttl=60)
    clock.now += 120
    value, age = c.get_stale("quote:AAPL")
    assert value.price == 100.0 and age == 120


def test_redis_itself_expires_the_entry_after_ttl_plus_grace():
    c, _, client = make(grace=3600)
    c.set("quote:AAPL", quote(), ttl=60)
    assert client.ttl(KEY_NAMESPACE + "quote:AAPL") == 3660   # bounded retention: nothing lives forever


def test_customer_data_keys_are_refused():
    c, _, client = make()
    for key in ("portfolio:alice", "chat:abc", "user:1:holdings", "answer:what-is-an-etf"):
        with pytest.raises(ValueError):
            c.set(key, quote(), ttl=60)
        with pytest.raises(ValueError):
            c.get(key)
    assert client.keys("*") == []


def test_only_known_public_models_can_be_stored():
    c, _, _ = make()
    with pytest.raises(ValueError):
        c.set("quote:AAPL", {"anything": "else"}, ttl=60)


def test_values_are_plain_json_not_pickle():
    c, _, client = make()
    c.set("quote:AAPL", quote(), ttl=60)
    stored = json.loads(client.get(KEY_NAMESPACE + "quote:AAPL"))
    assert stored["t"] == "Quote" and stored["v"]["price"] == 100.0


def test_a_corrupt_entry_is_a_miss_and_is_deleted():
    c, _, client = make()
    client.set(KEY_NAMESPACE + "quote:AAPL", "{not json")
    assert c.get("quote:AAPL") is None
    assert client.get(KEY_NAMESPACE + "quote:AAPL") is None


def test_an_entry_naming_an_unknown_type_is_rejected():
    c, _, client = make()
    client.set(KEY_NAMESPACE + "quote:AAPL", json.dumps({"t": "os.system", "stored_at": 0, "ttl": 1, "v": {}}))
    assert c.get("quote:AAPL") is None


def test_redis_down_fails_open():
    class Broken:
        def get(self, *a, **k): raise redis.ConnectionError("down")
        def set(self, *a, **k): raise redis.ConnectionError("down")

    c = RedisTTLCache(Broken())
    c.set("quote:AAPL", quote(), ttl=60)       # does not raise
    assert c.get("quote:AAPL") is None         # a miss, so the caller fetches from the provider
    assert c.get_stale("quote:AAPL") is None
    assert c.errors == 3


def test_build_cache_defaults_to_memory_and_falls_back_without_a_url():
    settings = load_settings()
    assert isinstance(build_cache(settings), TTLCache)
    settings.cache.backend, settings.secrets.redis_url = "redis", None
    assert isinstance(build_cache(settings), TTLCache)
    settings.secrets.redis_url = "redis://127.0.0.1:1/0"   # nothing listens there
    assert isinstance(build_cache(settings), TTLCache)
