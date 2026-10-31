"""Typed market data. Every value carries WHERE it came from and WHEN, so freshness can be shown honestly."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Quote(BaseModel):
    symbol: str
    price: float
    previous_close: float | None = None
    change_pct: float | None = None
    currency: str = "USD"
    source: str
    as_of: str | None = None                                # the market date of the price when the provider says (Alpha Vantage free quotes are end-of-day)
    fetched_at: datetime = Field(default_factory=utc_now)   # when WE got it from the provider, not when it was cached
    is_stale: bool = False                                  # True = the provider failed and this is an older saved value
    age_seconds: float = 0.0                                # how old fetched_at is at the moment we hand it out


class PricePoint(BaseModel):
    date: str
    close: float


class PriceHistory(BaseModel):
    symbol: str
    period: str
    points: list[PricePoint]
    source: str
    fetched_at: datetime = Field(default_factory=utc_now)
    is_stale: bool = False
    age_seconds: float = 0.0
