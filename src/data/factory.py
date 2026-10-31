"""Build the market data service from config.yaml: providers in `market_data.provider_order`, wrapped in a FallbackProvider."""
from pathlib import Path

from src.core.config import PROJECT_ROOT, Settings, load_settings
from src.core.logging import get_logger
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.base import MarketDataProvider
from src.data.providers.fallback import FallbackProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.data.rate_limiter import CallBudget

log = get_logger(__name__)


def build_providers(settings: Settings, order: list[str] | None = None) -> list[MarketDataProvider]:
    providers: list[MarketDataProvider] = []
    for name in order or settings.market_data.provider_order:
        if name == "yfinance":
            providers.append(YFinanceProvider(settings.market_data.timeout_seconds))
        elif name == "alpha_vantage":
            key = settings.secrets.alpha_vantage_api_key
            if not key:
                log.warning("alpha_vantage is in provider_order but ALPHA_VANTAGE_API_KEY is not set; skipping it")
                continue
            cfg = settings.alpha_vantage
            budget = CallBudget(cfg.max_calls_per_day, cfg.min_interval_seconds, Path(PROJECT_ROOT) / cfg.usage_file)
            providers.append(AlphaVantageProvider(key, budget, cfg.base_url, settings.market_data.timeout_seconds))
        else:
            raise ValueError(f"unknown market data provider {name!r}")
    if not providers:
        raise ValueError("no market data providers could be built")
    return providers


def build_market_data_service(settings: Settings | None = None, order: list[str] | None = None) -> MarketDataService:
    settings = settings or load_settings()
    providers = build_providers(settings, order)
    provider = providers[0] if len(providers) == 1 else FallbackProvider(providers)
    return MarketDataService(provider, TTLCache(), settings)
