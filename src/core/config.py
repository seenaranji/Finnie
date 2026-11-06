"""Load settings: non-secret values from config.yaml, secrets from environment (.env).

Why two sources? config.yaml is committed to git and safe to share. API keys must
never be, so they come from environment variables that python-dotenv loads from .env.
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class MissingSecretError(RuntimeError):
    """Raised when code needs a secret that is not set. Message says how to fix it."""


# --- Typed sections of config.yaml. A typo or wrong type fails loudly at load time. ---
class AppConfig(BaseModel):
    log_level: str = "INFO"


class Price(BaseModel):
    input: float = 0.0   # $ per 1M input tokens
    output: float = 0.0  # $ per 1M output tokens


class LLMConfig(BaseModel):
    answer_model: str
    router_model: str
    timeout_seconds: float = 30
    max_retries: int = 3
    prices_per_million_tokens: dict[str, Price] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    embedding_model: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.0
    vector_store: str = "pgvector"           # only pgvector exists now (Chroma removed after Step 23b); the keyword search is PostgreSQL full-text
    candidate_k: int = 20                    # chunks fetched from the store before filtering and reranking
    embedding_prefetch: bool = True          # start embedding the user's message while the router runs (Step 25)
    embedding_cache_size: int = 256          # remember this many question embeddings (0 = remember none)
    rerank: bool = False                     # second-stage re-ordering by an LLM (off until Step 8 shows it helps)
    dedupe_overlap: float = 0.8              # drop a chunk if this share of its words appears in a better chunk
    # --- Step 7c: hybrid search. Default stays "dense" so Step 7 behaviour is unchanged until Step 8 measures. ---
    retrieval_mode: str = "dense"            # dense | bm25 | hybrid
    rrf_k: int = 60                          # RRF damping constant
    candidates_per_retriever: int = 20       # how many chunks each retriever contributes to the fusion
    bm25_min_score: float | None = None      # a BM25 score at/above this lets a chunk in even if its cosine is below the threshold; None = BM25 cannot open the gate


class DatabaseConfig(BaseModel):
    """PostgreSQL settings. The URL (it holds the password) comes from the DATABASE_URL environment variable, never from this file."""
    pool_min_size: int = 1
    pool_max_size: int = 10
    pool_timeout_seconds: float = 10     # how long a request waits for a free connection before failing
    max_holdings_per_portfolio: int = 500
    keyword_match: str = "any"           # any = a chunk matching ANY query word qualifies (like BM25) | all = it must match every word (websearch_to_tsquery default)


class AuthConfig(BaseModel):
    token_minutes: int = 60              # how long a login stays valid; after that the user logs in again (no refresh tokens yet)
    min_password_length: int = 10
    max_password_length: int = 128       # also stops absurdly long inputs from being hashed
    max_failed_logins: int = 5           # per email + client address ...
    lockout_minutes: int = 15            # ... within this window, then 429 until it passes
    max_registrations_per_hour: int = 10 # per client address


class CacheConfig(BaseModel):
    quote_ttl_seconds: int = 60
    history_ttl_seconds: int = 900
    backend: str = "memory"              # memory | redis. Redis holds PUBLIC market data and news only (see src/data/redis_cache.py)
    stale_grace_seconds: int = 3600      # redis only: how long past its TTL an entry may still be served (flagged stale) when the provider is down
    redis_socket_timeout_seconds: float = 0.5   # a slow cache must never be slower than the provider it protects


class MarketDataConfig(BaseModel):
    timeout_seconds: float = 10       # per provider call
    max_attempts: int = 3             # 1 try + 2 retries, transient errors only
    retry_base_seconds: float = 0.5   # first wait; doubles each retry, with random jitter
    provider_order: list[str] = ["yfinance", "alpha_vantage"]   # tried left to right


class AlphaVantageConfig(BaseModel):
    base_url: str = "https://www.alphavantage.co/query"
    max_calls_per_day: int = 20       # free key allows 25; keep a margin for manual checks
    min_interval_seconds: float = 1.1   # free key allows 1 request per second
    usage_file: str = ".alpha_usage.json"   # remembers today's count across restarts (git-ignored)


class ScenarioReturns(BaseModel):
    """Assumed ANNUAL returns in percent for one risk profile. Illustrative assumptions, not forecasts."""
    conservative: float
    expected: float
    optimistic: float

    @model_validator(mode="after")
    def _ordered_and_sane(self):
        if not (self.conservative <= self.expected <= self.optimistic):
            raise ValueError("scenario returns must satisfy conservative <= expected <= optimistic")
        if not all(-50 <= x <= 30 for x in (self.conservative, self.expected, self.optimistic)):
            raise ValueError("scenario returns must be between -50 and 30 percent a year")
        return self


class GoalPlanningConfig(BaseModel):
    inflation_pct: float                          # assumed average yearly inflation
    large_monthly_usd: float = 5000               # ASSUMPTION: a monthly saving above this is flagged as a large commitment
    risk_profiles: dict[str, ScenarioReturns]     # profile name -> assumed returns


class NewsFeed(BaseModel):
    name: str
    url: str


class NewsConfig(BaseModel):
    ttl_seconds: int
    timeout_seconds: float
    max_items: int
    max_per_feed: int             # so one busy feed cannot crowd out the others
    max_age_days: int
    max_title_chars: int
    max_summary_chars: int
    user_agent: str
    ticker_feed_url: str          # contains {symbol}
    market_feeds: list[NewsFeed]


class WorkflowConfig(BaseModel):
    history_turns_for_rewrite: int
    clarify_reply: str
    fallback_reply: str
    needs_portfolio_reply: str
    scope_reply: str
    scope_log_path: str
    execution: str
    max_tasks: int
    max_concurrency: int
    branch_timeout_seconds: float


class Secrets(BaseSettings):
    """Read from environment variables / .env. Missing is allowed here (None);
    code that actually needs a key calls the require_* method for a clear error."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    database_url: str | None = None
    jwt_secret: str | None = None
    redis_url: str | None = None   # rediss://:password@host:6379/0 (TLS) in production; contains the password, so it lives in .env

    def require_jwt_secret(self) -> str:
        if not self.jwt_secret or len(self.jwt_secret) < 32:
            raise MissingSecretError(
                "JWT_SECRET is missing or shorter than 32 characters. Add it to .env (see .env.example); generate one with: "
                "python -c \"import secrets;print(secrets.token_urlsafe(48))\"")
        return self.jwt_secret

    def require_database_url(self) -> str:
        if not self.database_url:
            raise MissingSecretError(
                "DATABASE_URL is not set. Add it to .env (see .env.example) and start the database with: docker compose up -d")
        return self.database_url

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise MissingSecretError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    rag: RAGConfig
    database: DatabaseConfig = DatabaseConfig()
    auth: AuthConfig = AuthConfig()
    cache: CacheConfig
    market_data: MarketDataConfig = MarketDataConfig()
    alpha_vantage: AlphaVantageConfig = AlphaVantageConfig()
    goal_planning: GoalPlanningConfig
    news: NewsConfig
    workflow: WorkflowConfig
    secrets: Secrets

    def masked(self) -> dict:
        """Settings as a dict that is safe to print: secrets show only last 4 chars."""
        data = self.model_dump(exclude={"secrets"})
        data["secrets"] = {
            name: ("<not set>" if not v else "****" + v[-4:])
            for name, v in self.secrets.model_dump().items()
        }
        return data


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(path) as config_file:
        raw = yaml.safe_load(config_file)
    for env_name, key, kind in (("FINNIE_EMBEDDING_PREFETCH", "embedding_prefetch", lambda v: v.lower() in ("1", "true", "yes")),
                                ("FINNIE_EMBEDDING_CACHE_SIZE", "embedding_cache_size", int)):   # benchmark switches (Step 25): same idea as FINNIE_VECTOR_STORE
        if os.environ.get(env_name):
            raw["rag"][key] = kind(os.environ[env_name])
    if os.environ.get("FINNIE_VECTOR_STORE"):   # run one command against the other store without editing config.yaml (Step 23b side-by-side runs)
        raw["rag"]["vector_store"] = os.environ["FINNIE_VECTOR_STORE"]
    return Settings(**raw, secrets=Secrets())
