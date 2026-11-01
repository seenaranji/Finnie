"""Request and response shapes for the API. FastAPI validates every request against these before any of our code runs; a bad one becomes a 422."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from src.agents.goal_calc import MAX_AMOUNT, MAX_YEARS
from src.core.schemas import Source
from src.data.models import Quote


class ErrorBody(BaseModel):
    """Every error, from every endpoint, has this one shape, so a client needs only one error handler."""
    code: str                      # short machine-readable reason, e.g. ticker_not_found
    message: str                   # safe to show a user: no stack traces or internals
    request_id: str                # quote it to find the log lines for this request
    details: list[dict] = Field(default_factory=list)   # for 422: one entry per bad field


class HealthResponse(BaseModel):
    status: str
    version: str
    ready: dict[str, bool]         # which lazily built parts exist yet


class IndexQuote(BaseModel):
    name: str                      # "S&P 500"
    quote: Quote


class MarketOverview(BaseModel):
    indices: list[IndexQuote]
    unavailable: list[str]         # indices that could not be fetched: a partial overview is still returned
    note: str = "Index levels are for information only, not advice. Check each quote's as_of and is_stale fields."


class GoalRequest(BaseModel):
    goal: float = Field(gt=0, le=MAX_AMOUNT, description="Target amount in dollars, in future money")
    years: float = Field(gt=0, le=MAX_YEARS)
    current_savings: float = Field(0, ge=0, le=MAX_AMOUNT)
    risk_profile: Literal["low", "medium", "high"] = "medium"


class ScenarioOut(BaseModel):
    scenario: str
    annual_return_pct: float
    monthly_needed: float
    monthly_needed_inflation_adjusted: float
    current_savings_grow_to: float
    total_contributed: float
    growth_from_returns: float
    already_reached: bool
    path_by_year: list[float] = Field(default_factory=list)   # balance at the end of each year (year 0 first), for the chart


class GoalResponse(BaseModel):
    goal: float
    years: float
    current_savings: float
    risk_profile: str
    inflation_pct: float
    inflation_adjusted_goal: float
    scenarios: list[ScenarioOut]
    flags: list[str]
    assumptions: str               # always shown with the numbers: these are assumptions, not forecasts
    disclaimer: str


class PassageOut(BaseModel):
    doc_id: str
    title: str
    section: str
    source_name: str
    source_url: str
    last_updated: str
    text: str
    similarity: float


class KnowledgeResponse(BaseModel):
    query: str
    answerable: bool               # False = nothing scored above the similarity threshold
    results: list[PassageOut]


class PortfolioRequest(BaseModel):
    csv_text: str = Field(min_length=1, max_length=20000, description="CSV with the columns ticker, shares, avg_cost")
    question: str = Field("Please explain my portfolio in simple terms.", min_length=1, max_length=500)


class RegisterRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256, description="Stored only as an Argon2 hash; never logged or returned")


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class UserOut(BaseModel):
    id: UUID
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int                # seconds
    user: UserOut


class PortfolioSaved(BaseModel):
    id: UUID


class PortfolioResponse(BaseModel):
    text: str
    disclaimer: str
    sources: list[Source]
    metrics: dict                  # the numbers behind the explanation (weights, sectors, correlation ...), for charts
    correlation: dict | None = None   # {"tickers": [...], "matrix": [[...]]} for a heatmap; None when there is too little price history
    cost_usd: float
    latency_s: float


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$",
                                   description="Omit to start a new conversation; send back the id you get to continue it")
    portfolio_csv: str | None = Field(None, max_length=20000, description="Optional holdings (ticker, shares, avg_cost); remembered for the session")


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    text: str
    answered_by: str               # which agent (or fixed reply) produced the text
    route: str                     # what the router chose: finance_qa, market, multi, out_of_scope ...
    sources: list[Source]
    disclaimer: str
    cost_usd: float                # all model calls for this turn
    latency_s: float
    degraded: bool                 # True when a fallback reply replaced a failed step


class ArticleSummary(BaseModel):
    id: str
    title: str
    category: str
    difficulty: str | None = None
    source_name: str | None = None
    last_updated: str | None = None


class Article(ArticleSummary):
    source_url: str | None = None
    text: str                      # markdown body


class NewsItemOut(BaseModel):
    id: int
    title: str
    summary: str
    link: str
    source: str
    published: datetime | None = None


class NewsResponse(BaseModel):
    topic: str
    items: list[NewsItemOut]
    fetched_at: datetime
    withheld: int = 0              # items left out because their text looked like instructions to an AI (same quarantine as the news agent)
    feed_errors: list[str] = Field(default_factory=list)
