"""A small in-memory cache with a TTL (time to live).

Two reads on purpose:
  get()        fresh values only (younger than their TTL); counts a hit or a miss
  get_stale()  the saved value even when expired, with its age; used only when the provider fails

Expired entries are NOT deleted, because an old price is still better than no price when the API is down
(as long as we flag it). The clock is injectable so tests can move time without sleeping.
"""
import time
from typing import Any, Callable


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._items: dict[str, tuple[Any, float, float]] = {}   # key -> (value, stored_at, ttl)
        self.hits = 0
        self.misses = 0
        self.stale_served = 0

    def set(self, key: str, value: Any, ttl: float) -> None:
        self._items[key] = (value, self._clock(), ttl)

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item and self._clock() - item[1] < item[2]:
            self.hits += 1
            return item[0]
        self.misses += 1
        return None

    def get_stale(self, key: str) -> tuple[Any, float] | None:
        item = self._items.get(key)
        if not item:
            return None
        self.stale_served += 1
        return item[0], self._clock() - item[1]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
