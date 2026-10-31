"""The provider interface. The service only knows this; yfinance, Alpha Vantage (Step 12) or a test fake plug in behind it.

A provider does ONE thing: one network call, translated into our typed models or our two errors.
It does not retry or cache; the service does that for every provider the same way.
"""
from abc import ABC, abstractmethod

from src.data.models import PriceHistory, Quote


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Raise TickerNotFound for an unknown symbol, ProviderUnavailable for anything transient."""

    @abstractmethod
    def get_history(self, symbol: str, period: str) -> PriceHistory:
        """Daily closes for `period` (yfinance style: 5d, 1mo, 1y ...). Same error rules as get_quote."""
