"""yfinance provider (unofficial Yahoo wrapper: free, no key, can break without warning).

One path for quotes and history: ask for daily bars with a timeout. The last bar is the latest price (intraday it
updates live) and the one before it is the previous close. An EMPTY result means Yahoo does not know the symbol.
"""
import logging
import warnings

import yfinance as yf

from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.models import PricePoint, PriceHistory, Quote
from src.data.providers.base import MarketDataProvider

warnings.filterwarnings("ignore", module="yfinance")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # it prints its own 404 text; we raise our own clean error


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(self, timeout_seconds: float = 10) -> None:
        self.timeout = timeout_seconds

    def _bars(self, symbol: str, period: str):
        try:
            frame = yf.Ticker(symbol).history(period=period, timeout=self.timeout)
            if frame.empty:   # Yahoo answered but has no bars: the symbol does not exist
                raise TickerNotFound(f"I couldn't find a ticker called {symbol}.", f"no bars for {symbol}")
            closes = frame["Close"].dropna()
            if closes.empty:
                raise ValueError("payload has no usable Close values")
            return closes
        except TickerNotFound:
            raise
        except Exception as error:   # timeouts, 429, DNS, SSL, malformed payload: all "try later", none are our bug
            raise ProviderUnavailable("The market data service is not responding right now.",
                                      f"{type(error).__name__}: {error}") from error

    def get_quote(self, symbol: str) -> Quote:
        closes = self._bars(symbol, "5d")
        price = float(closes.iloc[-1])
        previous = float(closes.iloc[-2]) if len(closes) > 1 else None
        change = round((price - previous) / previous * 100, 2) if previous else None
        return Quote(symbol=symbol, price=round(price, 4), previous_close=None if previous is None else round(previous, 4),
                     change_pct=change, source=self.name)

    def get_history(self, symbol: str, period: str) -> PriceHistory:
        closes = self._bars(symbol, period)
        points = [PricePoint(date=str(day.date()), close=round(float(close), 4)) for day, close in closes.items()]
        return PriceHistory(symbol=symbol, period=period, points=points, source=self.name)
