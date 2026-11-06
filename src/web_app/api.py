"""Finnie's HTTP API. Start it with:  uvicorn src.web_app.api:app --reload   then open http://127.0.0.1:8000/docs

This file only translates between HTTP and the logic that already exists (agents, calculators, graph). No finance logic lives here, so any client
(the Streamlit UI, curl, another program) gets identical behaviour.

Status codes used: 200 ok | 404 unknown ticker or route | 422 the request body/params are invalid (FastAPI or our own domain checks)
                   500 a bug on our side | 502 a data provider failed and nothing saved could stand in | 503 not ready / dependency down
"""
import math
from decimal import Decimal
from contextlib import asynccontextmanager
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from uuid import UUID

from psycopg import OperationalError
from psycopg_pool import PoolTimeout
from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.auth.repository import EmailTaken
from src.auth.security import AuthInputError, AuthUser, InvalidToken, TooManyAttempts
from src.auth.service import BadCredentials
from src.core.config import load_settings
from src.core.logging import get_logger, setup_logging
from src.agents import goal_calc, mortgage, portfolio_calc
from src.agents.portfolio_agent import NothingToAnalyze, PortfolioAnalysisAgent
from src.agents.portfolio_summary import summarize_portfolio
from src.core.guardrails import DISCLAIMER_STRONG
from src.data.errors import MarketDataError, ProviderUnavailable, TickerNotFound
from src.data.models import PriceHistory, Quote
from src.data.news import NewsUnavailable
from src.data.portfolio_repository import PortfolioIn, PortfolioNotFound, PortfolioSummary, PropertyNotFound, StoredPortfolio
from src.rag.hybrid_retriever import StaleKeywordIndexError
from src.workflow.factory import config_for
from src.web_app.models import (Article, ArticleSummary, ChatRequest, ChatResponse, ErrorBody, GoalRequest, GoalResponse, HealthResponse, IndexQuote, KnowledgeResponse, MarketOverview, NewsItemOut, NewsResponse, PassageOut,
                                GroupedPortfolio, LoginRequest, MortgagePayoffResponse, MortgageTermsIn, PortfolioRequest, PortfolioResponse, PortfolioSaved, RegisterRequest, TokenResponse, UserOut)
from src.web_app.services import Services

log = get_logger("finnie.api")
INDICES = {"^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite"}
TICKER = Path(pattern=r"^\^?[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="Ticker such as AAPL, BRK-B or ^GSPC (encode ^ as %5E in a URL)")
CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]   # the Streamlit UI (Step 23); any other origin is refused by browsers


class NothingToExplain(Exception):
    """The portfolio holds only real estate: there are no market holdings for the explanation to work on."""


class AuthRequired(Exception):
    """No valid login token on a request to a protected endpoint."""


def _error(status: int, code: str, message: str, request: Request, details: list | None = None, headers: dict | None = None) -> JSONResponse:
    body = ErrorBody(code=code, message=message, request_id=request.state.request_id, details=details or [])
    return JSONResponse(status_code=status, content=body.model_dump(), headers={"X-Request-ID": request.state.request_id, **(headers or {})})


def _summary(doc) -> ArticleSummary:
    m = doc.metadata
    return ArticleSummary(id=doc.id, title=m.get("title", doc.id), category=m.get("category", ""), difficulty=m.get("difficulty"),
                          source_name=m.get("source_name"), last_updated=m.get("last_updated"))


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services.settings.secrets.require_jwt_secret()   # refuse to START without a signing secret, instead of failing on the first login
        yield

    app = FastAPI(title="Finnie API", version="0.1.0", description="Educational finance tutor. Not financial advice.", lifespan=lifespan)
    app.state.services = services or Services()
    bearer = HTTPBearer(auto_error=False, description="Log in at POST /auth/login, then send the access_token as: Authorization: Bearer <token>")

    def require_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> AuthUser:
        """The one gate for every protected endpoint. A request is let in only with a valid, unexpired token signed with our secret."""
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise AuthRequired()
        try:
            user = request.app.state.services.auth.verify_token(credentials.credentials)
        except InvalidToken as reason:
            log.info("req=%s rejected token: %s", request.state.request_id, reason)
            raise AuthRequired() from None
        request.state.user = user
        return user

    # Everything on this router needs a login. /health, /auth/register, /auth/login and the docs are the only routes NOT on it, and a test walks
    # every route in the app to prove that nothing else is open (tests/test_auth_api.py).
    protected = APIRouter(dependencies=[Depends(require_user)])
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type", "X-Request-ID", "Authorization"],
                       expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def request_id_and_timing(request: Request, call_next):
        # Honour a caller-supplied id (so one id follows a request across systems) but only if it is short and plain; otherwise make our own.
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if 0 < len(supplied) <= 64 and supplied.replace("-", "").replace("_", "").isalnum() else uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("req=%s %s %s unhandled", request.state.request_id, request.method, request.url.path)
            raise
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        log.info("req=%s %s %s -> %s in %.0f ms", request.state.request_id, request.method, request.url.path, response.status_code, ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, error: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]} for e in error.errors()]
        return _error(422, "invalid_request", "The request was not valid. See details.", request, details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        # 404 for an unknown URL and 405 for a wrong verb come from the router, not from our code; give them the same shape as every other error.
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(error.status_code, codes.get(error.status_code, "http_error"), str(error.detail), request)

    @app.exception_handler(goal_calc.GoalInputError)
    async def bad_goal(request: Request, error: goal_calc.GoalInputError):
        return _error(422, "invalid_goal", str(error), request)

    @app.exception_handler(portfolio_calc.PortfolioInputError)
    async def bad_portfolio(request: Request, error: portfolio_calc.PortfolioInputError):
        return _error(422, "invalid_portfolio", "Your portfolio file could not be read. It needs the columns ticker, shares and avg_cost.", request,
                      [{"problem": p} for p in error.problems])

    @app.exception_handler(NothingToExplain)
    async def nothing_to_explain(request: Request, error: NothingToExplain):
        return _error(422, "nothing_to_explain", "This portfolio has no stocks, funds or other market holdings to explain. Its real estate is shown in the grouped view.", request)

    @app.exception_handler(AuthRequired)
    async def not_logged_in(request: Request, error: AuthRequired):
        return _error(401, "not_authenticated", "Please log in to use Finnie.", request, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(BadCredentials)
    async def bad_credentials(request: Request, error: BadCredentials):
        return _error(401, "invalid_credentials", "The email or password is incorrect.", request, headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(AuthInputError)
    async def bad_auth_input(request: Request, error: AuthInputError):
        return _error(422, "invalid_auth_input", "Please fix the problems below.", request, [{"problem": p} for p in error.problems])

    @app.exception_handler(TooManyAttempts)
    async def too_many(request: Request, error: TooManyAttempts):
        return _error(429, "too_many_attempts", f"Too many attempts. Please wait {max(1, round(error.retry_after_seconds / 60))} minute(s) and try again.", request,
                      headers={"Retry-After": str(error.retry_after_seconds)})

    @app.exception_handler(PortfolioNotFound)
    async def portfolio_not_found(request: Request, error: PortfolioNotFound):
        return _error(404, "portfolio_not_found", "No saved portfolio has that id.", request)

    @app.exception_handler(PropertyNotFound)
    async def property_not_found(request: Request, error: PropertyNotFound):
        return _error(404, "property_not_found", "That portfolio has no property with that name.", request)

    @app.exception_handler(PoolTimeout)
    async def database_busy(request: Request, error: PoolTimeout):
        return _error(503, "database_busy", "The database is busy. Please try again in a moment.", request)

    @app.exception_handler(OperationalError)
    async def database_down(request: Request, error: OperationalError):
        log.error("req=%s database unavailable: %s", request.state.request_id, str(error)[:200])
        return _error(503, "database_unavailable", "The database is not reachable right now. Please try again shortly.", request)

    @app.exception_handler(StaleKeywordIndexError)
    async def index_stale(request: Request, error: StaleKeywordIndexError):
        return _error(503, "knowledge_index_unavailable", "The knowledge search index is being rebuilt. Please try again shortly.", request)

    @app.exception_handler(TickerNotFound)
    async def ticker_not_found(request: Request, error: TickerNotFound):
        return _error(404, "ticker_not_found", error.user_message, request)

    @app.exception_handler(ProviderUnavailable)
    async def provider_down(request: Request, error: ProviderUnavailable):
        return _error(502, "data_provider_unavailable", error.user_message, request)

    @app.exception_handler(MarketDataError)
    async def market_error(request: Request, error: MarketDataError):
        return _error(502, "market_data_error", error.user_message, request)

    @app.exception_handler(NewsUnavailable)
    async def news_down(request: Request, error: NewsUnavailable):
        return _error(502, "news_unavailable", error.user_message, request)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.error("req=%s unexpected %s: %s", request.state.request_id, type(error).__name__, str(error)[:200])   # full detail stays in the log, never in the reply
        return _error(500, "internal_error", "Something went wrong on our side. Quote this request id if you report it.", request)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        """Is the server up? Cheap on purpose: never calls a model, a data provider or the vector store."""
        return HealthResponse(status="ok", version=app.version, ready=request.app.state.services.ready())

    @app.post("/auth/register", response_model=TokenResponse, status_code=201, tags=["auth"])
    def register(body: RegisterRequest, request: Request):
        """Create an account and log in straight away. Anyone may register; nobody can use Finnie without an account."""
        auth = request.app.state.services.auth
        user = auth.register(body.email, body.password, request.client.host if request.client else "unknown")
        return TokenResponse(access_token=auth.token_for(user), expires_in=auth.config.token_minutes * 60, user=UserOut(id=user.id, email=user.email))

    @app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
    def login(body: LoginRequest, request: Request):
        auth = request.app.state.services.auth
        user = auth.login(body.email, body.password, request.client.host if request.client else "unknown")
        return TokenResponse(access_token=auth.token_for(user), expires_in=auth.config.token_minutes * 60, user=UserOut(id=user.id, email=user.email))

    @protected.get("/auth/me", response_model=UserOut, tags=["auth"])
    def me(user: AuthUser = Depends(require_user)):
        return UserOut(id=user.id, email=user.email)

    @protected.get("/market/quote/{ticker}", response_model=Quote, tags=["market"])
    def quote(request: Request, ticker: str = TICKER):
        """Latest price. Carries its source, as_of date and is_stale flag: a stale value is returned (flagged) rather than an error when the provider is down."""
        return request.app.state.services.market.get_quote(ticker)

    @protected.get("/market/history/{ticker}", response_model=PriceHistory, tags=["market"])
    def history(request: Request, ticker: str = TICKER, period: Literal["5d", "1mo", "3mo", "6mo", "1y", "ytd"] = Query("1mo")):
        """Daily closes for charting (used by the Markets tab in Step 23)."""
        return request.app.state.services.market.get_history(ticker, period)

    @protected.get("/market/overview", response_model=MarketOverview, tags=["market"])
    def overview(request: Request):
        """The main US indices side by side. If one fails the others are still returned and the failure is listed, not hidden."""
        market = request.app.state.services.market

        def fetch(symbol: str):
            try:
                return IndexQuote(name=INDICES[symbol], quote=market.get_quote(symbol))
            except MarketDataError as error:
                log.warning("overview: %s unavailable: %s", symbol, error.user_message)
                return INDICES[symbol]
        with ThreadPoolExecutor(max_workers=len(INDICES)) as pool:
            results = list(pool.map(fetch, INDICES))
        return MarketOverview(indices=[r for r in results if isinstance(r, IndexQuote)], unavailable=[r for r in results if isinstance(r, str)])

    def _news_response(batch) -> NewsResponse:
        clean = [i for i in batch.items if not i.injection_markers]
        return NewsResponse(topic=batch.topic, items=[NewsItemOut(**i.model_dump(exclude={"injection_markers"})) for i in clean],
                            fetched_at=batch.fetched_at, withheld=len(batch.items) - len(clean), feed_errors=batch.feed_errors)

    @protected.get("/news/market", response_model=NewsResponse, tags=["news"])
    def market_news(request: Request):
        """Recent general market headlines from the configured feeds, newest first. Headlines only; the Chat tab summarizes them."""
        return _news_response(request.app.state.services.news.get_market_news())

    @protected.get("/news/ticker/{ticker}", response_model=NewsResponse, tags=["news"])
    def ticker_news(request: Request, ticker: str = TICKER):
        """Recent headlines about one ticker."""
        return _news_response(request.app.state.services.news.get_ticker_news(ticker))

    @protected.post("/goals/plan", response_model=GoalResponse, tags=["goals"])
    def plan_goal(body: GoalRequest, request: Request):
        """Three-scenario savings plan. Pure arithmetic with assumed returns from config.yaml: no model is called, so it is instant, free and repeatable."""
        config = request.app.state.services.settings.goal_planning
        plan = goal_calc.plan_goal(body.goal, body.years, body.current_savings, body.risk_profile, config)
        data = {k: getattr(plan, k) for k in ("goal", "years", "current_savings", "risk_profile", "inflation_pct", "inflation_adjusted_goal", "assumptions")}
        years_whole = max(1, math.ceil(body.years))
        scenarios = [{**s.__dict__, "path_by_year": [round(x, 2) for x in goal_calc.balance_path(body.current_savings, s.monthly_needed, s.annual_return_pct, years_whole)]}
                     for s in plan.scenarios]
        return GoalResponse(**data, scenarios=scenarios, flags=goal_calc.assess(plan, config.large_monthly_usd),
                            disclaimer="Educational illustration only, not financial advice. Real returns vary and can be negative.")

    @protected.get("/knowledge/search", response_model=KnowledgeResponse, tags=["knowledge"])
    def search_knowledge(request: Request, q: str = Query(min_length=2, max_length=300), category: str | None = Query(None, max_length=40),
                         top_k: int = Query(5, ge=1, le=10)):
        """Search the knowledge base. Returns the matching passages with their sources; no model writes anything here."""
        found = request.app.state.services.retriever.retrieve(q, top_k=top_k, category=category)
        results = [PassageOut(doc_id=p.citation.doc_id, title=p.citation.title, section=p.citation.section, source_name=p.citation.source_name,
                              source_url=p.citation.source_url, last_updated=p.citation.last_updated, text=p.text, similarity=round(p.similarity, 3))
                   for p in found.passages]
        return KnowledgeResponse(query=q, answerable=found.answerable, results=results)

    @protected.get("/knowledge/articles", response_model=list[ArticleSummary], tags=["knowledge"])
    def list_articles(request: Request, category: str | None = Query(None, max_length=40)):
        """Every article in the knowledge base (no text), optionally one category."""
        return [_summary(d) for d in request.app.state.services.articles() if category is None or d.metadata.get("category") == category]

    @protected.get("/knowledge/articles/{doc_id}", response_model=Article, tags=["knowledge"])
    def get_article(request: Request, doc_id: str = Path(pattern=r"^[A-Za-z0-9_\-]{1,80}$")):
        for d in request.app.state.services.articles():
            if d.id == doc_id:
                return Article(**_summary(d).model_dump(), source_url=d.metadata.get("source_url"), text=d.text)
        raise StarletteHTTPException(404, "No article with that id.")

    @protected.post("/portfolio/analyze", response_model=PortfolioResponse, tags=["portfolio"])
    def analyze_portfolio(body: PortfolioRequest, request: Request):
        """Explain a portfolio from CSV text. Bad CSV is the caller's fault (422 listing every problem); no prices at all is ours to report (502)."""
        services = request.app.state.services
        portfolio_calc.parse_portfolio_csv(body.csv_text)   # raises PortfolioInputError -> 422 with every problem, before any market call or model call
        agent = PortfolioAnalysisAgent(services.llm, services.market, services.metadata)   # one per request: the agent keeps per-call scratch state, so sharing one across threads is unsafe
        return _portfolio_reply(agent.run(body.csv_text, body.question))

    def _portfolio_reply(result) -> PortfolioResponse:
        metrics = result.metadata.get("metrics")
        if metrics is None:
            raise ProviderUnavailable("I couldn't get prices for your holdings right now. Please try again in a few minutes.")
        return PortfolioResponse(text=result.text, disclaimer=result.disclaimer or DISCLAIMER_STRONG, sources=result.sources, metrics=metrics,
                                 correlation=result.metadata.get("correlation"), cost_usd=result.metadata.get("cost_usd", 0.0), latency_s=result.metadata.get("latency_s", 0.0))

    # ---- saved portfolios (PostgreSQL). The UI never touches the database: it calls these. A missing id is 404, invalid holdings are 422. ----
    @protected.post("/portfolios", response_model=PortfolioSaved, status_code=201, tags=["portfolio"])
    def save_portfolio(body: PortfolioIn, request: Request, user: AuthUser = Depends(require_user)):
        """Save a portfolio (name + holdings) and get back its id. Duplicate tickers are merged."""
        return PortfolioSaved(id=request.app.state.services.portfolios.create(body.name, body.holdings, owner_id=user.id, properties=body.properties))

    @protected.get("/portfolios", response_model=list[PortfolioSummary], tags=["portfolio"])
    def list_portfolios(request: Request, user: AuthUser = Depends(require_user)):
        return request.app.state.services.portfolios.list(owner_id=user.id)

    @protected.get("/portfolios/{portfolio_id}", response_model=StoredPortfolio, tags=["portfolio"])
    def get_portfolio(request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user)):
        return request.app.state.services.portfolios.get(portfolio_id, owner_id=user.id)

    @protected.get("/portfolios/{portfolio_id}/summary", response_model=GroupedPortfolio, tags=["portfolio"])
    def summarize_saved_portfolio(request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user)):
        """A saved portfolio grouped by the kind of holding the user chose, with cost, current value, gain and share of the total per group. No model call: quick and free."""
        services = request.app.state.services
        stored = services.portfolios.get(portfolio_id, owner_id=user.id)
        prices, as_of = {}, []

        def price_of(holding):
            if holding.asset_type == "cash":
                return holding.ticker, Decimal(1), None   # cash is worth exactly one dollar per unit; there is nothing to look up
            try:
                quote = services.market.get_quote(holding.ticker)
                return holding.ticker, Decimal(str(quote.price)), quote.as_of
            except MarketDataError:
                return holding.ticker, None, None   # shown with its cost, without a value, and listed under "unpriced"
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(stored.holdings)))) as pool:
            for ticker, price, quote_date in pool.map(price_of, stored.holdings):
                prices[ticker] = price
                if quote_date:
                    as_of.append(quote_date)
        summary = summarize_portfolio(stored.holdings, stored.properties, prices)
        return GroupedPortfolio(id=stored.id, name=stored.name, prices_as_of=max(as_of) if as_of else None, **summary)

    @protected.get("/portfolios/{portfolio_id}/analysis", response_model=PortfolioResponse, tags=["portfolio"])
    def analyze_saved_portfolio(request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user), question: str = Query("Please explain my portfolio in simple terms.", min_length=1, max_length=500)):
        """Explain a SAVED portfolio: holdings come from the database, prices and metrics are computed now."""
        services = request.app.state.services
        agent = PortfolioAnalysisAgent(services.llm, services.market, services.metadata, repository=services.portfolios)
        try:
            return _portfolio_reply(agent.run_by_id(portfolio_id, user.id, question))
        except NothingToAnalyze:
            raise NothingToExplain() from None

    @protected.get("/portfolios/{portfolio_id}/mortgage-payoff", response_model=MortgagePayoffResponse, tags=["portfolio"])
    def mortgage_payoff(request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user),
                        extra: Decimal = Query(Decimal(0), ge=0, le=Decimal(1_000_000), allow_inf_nan=False, description="extra paid every month on top of the regular payment")):
        """How long until each mortgage in a saved portfolio is paid off, and how much sooner with `extra` more each month. Only computed when asked: no model call, nothing stored."""
        stored = request.app.state.services.portfolios.get(portfolio_id, owner_id=user.id)
        answers = [mortgage.mortgage_payoff(p, extra) for p in stored.properties if p.mortgage_balance > 0]
        return MortgagePayoffResponse(portfolio=stored.name, extra=float(extra), properties=answers)

    @protected.put("/portfolios/{portfolio_id}/mortgage-terms", status_code=204, tags=["portfolio"])
    def set_mortgage_terms(body: MortgageTermsIn, request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user)):
        """Add the monthly payment and yearly rate to ONE property of an existing portfolio, without re-entering the rest."""
        request.app.state.services.portfolios.set_mortgage_terms(portfolio_id, body.property, body.monthly_payment, body.rate_pct, owner_id=user.id)

    @protected.put("/portfolios/{portfolio_id}", status_code=204, tags=["portfolio"])
    def replace_portfolio(body: PortfolioIn, request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user)):
        """Replace the name, ALL holdings and ALL properties in one transaction (a full replacement: properties left out of the request are removed)."""
        request.app.state.services.portfolios.update(portfolio_id, body.holdings, body.name, owner_id=user.id, properties=body.properties)

    @protected.delete("/portfolios/{portfolio_id}", status_code=204, tags=["portfolio"])
    def delete_portfolio(request: Request, portfolio_id: UUID, user: AuthUser = Depends(require_user)):
        request.app.state.services.portfolios.delete(portfolio_id, owner_id=user.id)

    @protected.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(body: ChatRequest, request: Request, user: AuthUser = Depends(require_user)):
        """One conversational turn through the full workflow (router, agents, planner for multi-part questions). Memory is kept per session_id."""
        services = request.app.state.services
        session_id = body.session_id or uuid.uuid4().hex[:16]
        # A conversation belongs to the user who started it. The memory is keyed by "<user id>:<session id>", so user B sending user A's session id
        # simply starts a new conversation of their own and can never read or continue A's. The client only ever sees the plain session id.
        internal_id = f"{user.id}:{session_id}"
        state_in = {"message": body.message, "session_id": internal_id, "owner_id": str(user.id), **({"portfolio_id": str(body.portfolio_id)} if body.portfolio_id else {}), **({"portfolio_csv": body.portfolio_csv} if body.portfolio_csv else {})}
        start = time.perf_counter()
        with services.session_lock(internal_id):   # two simultaneous turns of ONE conversation would interleave its saved history; different sessions never wait
            state = services.graph.invoke(state_in, config_for(internal_id, services.settings.workflow.max_concurrency))
        reply = state["response"]
        trace = state.get("trace", [])
        routing = sum(t.get("cost_usd") or 0.0 for t in trace if t.get("turn") == state.get("turn") and t["node"] in ("contextualize", "router"))   # the agent's own cost is in its reply
        return ChatResponse(session_id=session_id, turn=state.get("turn", 1), text=reply["text"], answered_by=reply["agent_name"], route=state.get("route", ""),
                            sources=reply.get("sources", []), disclaimer=reply.get("disclaimer", ""),
                            cost_usd=round((reply.get("metadata", {}).get("cost_usd") or 0.0) + routing, 6),
                            latency_s=round(time.perf_counter() - start, 3), degraded=state.get("error") is not None)

    app.include_router(protected)
    return app


setup_logging(load_settings().app.log_level)
app = create_app()
