"""Market statistics in plain Python (no LLM). Inputs are daily closing prices, OLDEST FIRST.

Every function returns None when there is not enough data, never a guess. The agent passes that None on as "unavailable".
Everything here is based on CLOSING prices; intraday highs and lows are not used.
"""
import math

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def change_pct(price: float, previous: float | None) -> float | None:
    return round((price - previous) / previous * 100, 2) if previous else None


def moving_average(closes: list[float], window: int) -> float | None:
    """Mean of the last `window` closes; None if fewer than `window` closes exist."""
    return round(float(np.mean(closes[-window:])), 4) if len(closes) >= window else None


def price_vs_average_pct(price: float, average: float | None) -> float | None:
    return round((price - average) / average * 100, 2) if average else None


def range_52w(closes: list[float]) -> dict | None:
    """Lowest and highest close over the last year (up to 252 closes). `bars` says how many were really available."""
    if len(closes) < 20:
        return None
    window = closes[-TRADING_DAYS_PER_YEAR:]
    return {"low": round(min(window), 4), "high": round(max(window), 4), "bars": len(window),
            "covers_full_year": len(window) >= 250}


def annualized_volatility_pct(closes: list[float]) -> float | None:
    """Standard deviation of daily % returns, scaled by sqrt(252) to a yearly figure, as a percent. Needs 21+ closes."""
    if len(closes) < 21:
        return None
    prices = np.array(closes[-(TRADING_DAYS_PER_YEAR + 1):], dtype=float)
    returns = np.diff(prices) / prices[:-1]
    return round(float(np.std(returns, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100, 2)
