"""CallBudget: protects a rate-limited API from US.

Alpha Vantage's free key allows 25 requests a day and 1 per second, and it answers HTTP 200 with a polite message when
you go over, so the provider cannot rely on an error code. Better to never go over:
  - spacing: wait (not fail) until `min_interval_seconds` has passed since the last call
  - daily cap: refuse locally (no network call) once the day's budget is spent
  - the daily count is saved to a small file, because a restart must not reset what the API still remembers
`calls_in_last_minute()` exists so a run can report calls per minute against the limit.
"""
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.core.logging import get_logger
from src.data.errors import ProviderUnavailable

log = get_logger(__name__)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()   # the provider's own day boundary is not documented; UTC is our assumption


class CallBudget:
    def __init__(self, per_day: int, min_interval_seconds: float, state_file: Path | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 today: Callable[[], str] = _today_utc) -> None:
        self.per_day, self.min_interval = per_day, min_interval_seconds
        self.state_file, self._clock, self._sleep, self._today = state_file, clock, sleep, today
        self._recent: deque[float] = deque()
        self._last_call: float | None = None
        self._day, self._count = self._load()

    def _load(self) -> tuple[str, int]:
        try:
            saved = json.loads(self.state_file.read_text()) if self.state_file else {}
            if saved.get("date") == self._today():
                return saved["date"], int(saved["count"])
        except (OSError, ValueError, KeyError):
            pass
        return self._today(), 0

    def _save(self) -> None:
        if self.state_file:
            try:
                self.state_file.write_text(json.dumps({"date": self._day, "count": self._count}))
            except OSError:
                log.warning("could not save the call budget to %s", self.state_file)

    @property
    def used_today(self) -> int:
        return self._count if self._day == self._today() else 0

    def acquire(self) -> None:
        """Call right before each network request."""
        if self._day != self._today():
            self._day, self._count = self._today(), 0
        if self._count >= self.per_day:
            raise ProviderUnavailable("The backup market data service has reached its daily limit.",
                                      f"call budget spent: {self._count}/{self.per_day} today")
        if self._last_call is not None:
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        now = self._clock()
        self._last_call = now
        self._recent.append(now)
        self._count += 1
        self._save()

    def calls_in_last_minute(self) -> int:
        cutoff = self._clock() - 60
        return sum(1 for t in self._recent if t >= cutoff)
