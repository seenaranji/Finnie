"""Redis-backed drop-in for TTLCache (same get / set / get_stale / hit_rate), shared by every API worker.

Banking-domain rules, enforced in code rather than by convention:
  - PUBLIC DATA ONLY. Keys must start with an allow-listed prefix (quote:, history:, news:). Anything else (portfolios, chat history, user ids,
    LLM answers) is refused with a ValueError, so a later change cannot quietly put customer data into a shared cache.
  - JSON, never pickle. A value is a pydantic model stored as JSON with its type name; reading it back can never execute code even if Redis is tampered with.
  - BOUNDED RETENTION. Redis itself expires every entry after ttl + stale_grace. The grace keeps the "provider is down, serve a flagged old price"
    behaviour, but only for `stale_grace_seconds` (not forever).
  - FAIL OPEN. If Redis is down or slow the cache behaves as empty (a miss) and the app fetches from the provider. It never raises into a request,
    and the log line names the error type only, never a value.
  - Wall-clock timestamps (time.time), because the writer and the reader may be different processes; time.monotonic is per-process.
"""
import json
import time
from typing import Any, Callable

import redis

from src.core.logging import get_logger
from src.data.market_data import PriceHistory, Quote   # noqa: F401  (imported so the registry below can name them)
from src.data.news import NewsBatch

log = get_logger(__name__)

ALLOWED_PREFIXES = ("quote:", "history:", "news:")
KEY_NAMESPACE = "finnie:v1:"   # bump v1 when a model's shape changes, so old entries are ignored instead of failing to parse
_TYPES = {"Quote": Quote, "PriceHistory": PriceHistory, "NewsBatch": NewsBatch}


class RedisTTLCache:
    def __init__(self, client: "redis.Redis", stale_grace_seconds: int = 3600, clock: Callable[[], float] = time.time) -> None:
        self._r, self._grace, self._clock = client, stale_grace_seconds, clock
        self.hits = 0
        self.misses = 0
        self.stale_served = 0
        self.errors = 0

    @staticmethod
    def _key(key: str) -> str:
        if not key.startswith(ALLOWED_PREFIXES):
            raise ValueError(f"cache key {key!r} is not allowed: the shared cache holds public market data only")
        return KEY_NAMESPACE + key

    def set(self, key: str, value: Any, ttl: float) -> None:
        redis_key = self._key(key)
        name = type(value).__name__
        if name not in _TYPES:
            raise ValueError(f"cannot cache a {name}: only {sorted(_TYPES)} are allowed")
        payload = json.dumps({"t": name, "stored_at": self._clock(), "ttl": ttl, "v": value.model_dump(mode="json")})
        try:
            self._r.set(redis_key, payload, ex=int(ttl + self._grace))
        except redis.RedisError as error:
            self._failed("set", error)

    def _load(self, key: str) -> tuple[Any, float, float] | None:
        """(value, age_seconds, ttl) or None when absent, unreadable or Redis is down."""
        redis_key = self._key(key)
        try:
            raw = self._r.get(redis_key)
        except redis.RedisError as error:
            self._failed("get", error)
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            value = _TYPES[data["t"]].model_validate(data["v"])
            return value, self._clock() - data["stored_at"], data["ttl"]
        except (ValueError, KeyError, TypeError) as error:   # corrupt or old-shape entry: treat as absent and drop it
            log.warning("redis cache entry %s unreadable (%s); ignoring it", redis_key, type(error).__name__)
            try:
                self._r.delete(redis_key)
            except redis.RedisError:
                pass
            return None

    def get(self, key: str) -> Any | None:
        loaded = self._load(key)
        if loaded and loaded[1] < loaded[2]:
            self.hits += 1
            return loaded[0]
        self.misses += 1
        return None

    def get_stale(self, key: str) -> tuple[Any, float] | None:
        loaded = self._load(key)
        if not loaded:
            return None
        self.stale_served += 1
        return loaded[0], loaded[1]

    def _failed(self, action: str, error: Exception) -> None:
        self.errors += 1
        log.warning("redis cache %s failed (%s); continuing without the cache", action, type(error).__name__)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
