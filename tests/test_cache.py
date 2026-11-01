"""TTLCache on its own, with a clock the test moves by hand (no sleeping)."""
from src.data.cache import TTLCache


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def cache():
    clock = Clock()
    return TTLCache(clock), clock


def test_a_value_is_fresh_before_its_ttl_and_counts_a_hit():
    c, clock = cache()
    c.set("AAPL", 123, ttl=60)
    clock.now += 59.9
    assert c.get("AAPL") == 123 and (c.hits, c.misses) == (1, 0)


def test_a_value_is_expired_at_exactly_its_ttl():
    # the boundary: "younger than the ttl" means age < ttl, so an entry that is exactly ttl seconds old is already stale
    c, clock = cache()
    c.set("AAPL", 123, ttl=60)
    clock.now += 60
    assert c.get("AAPL") is None and (c.hits, c.misses) == (0, 1)


def test_a_missing_key_is_a_miss():
    c, _ = cache()
    assert c.get("nope") is None and c.misses == 1 and c.get_stale("nope") is None


def test_expired_values_are_kept_and_served_stale_with_their_age():
    c, clock = cache()
    c.set("AAPL", 123, ttl=60)
    clock.now += 600
    assert c.get("AAPL") is None
    assert c.get_stale("AAPL") == (123, 600.0) and c.stale_served == 1


def test_each_key_has_its_own_ttl():
    c, clock = cache()
    c.set("quote", 1, ttl=60)
    c.set("history", 2, ttl=900)
    clock.now += 120
    assert c.get("quote") is None and c.get("history") == 2


def test_setting_again_replaces_the_value_and_restarts_the_clock():
    c, clock = cache()
    c.set("k", "old", ttl=60)
    clock.now += 50
    c.set("k", "new", ttl=60)
    clock.now += 50
    assert c.get("k") == "new"   # 100 s after the first set, but only 50 after the second


def test_hit_rate():
    c, _ = cache()
    assert c.hit_rate == 0.0   # no division by zero before the first lookup
    c.set("k", 1, ttl=60)
    c.get("k"); c.get("k"); c.get("other")
    assert c.hit_rate == 2 / 3
