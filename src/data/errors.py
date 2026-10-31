"""Market data errors. Two kinds, because they need different handling:

  TickerNotFound       permanent: asking again gives the same answer, so never retry
  ProviderUnavailable  transient: timeout, rate limit (429), network down, malformed payload; retry, then fall back to stale data

Both carry a `user_message` that is safe to show a user (no stack trace, no internals).
"""


class MarketDataError(Exception):
    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message


class TickerNotFound(MarketDataError):
    pass


class ProviderUnavailable(MarketDataError):
    pass
