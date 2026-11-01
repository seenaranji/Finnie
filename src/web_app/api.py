"""Finnie's HTTP API. Start it with:  uvicorn src.web_app.api:app --reload   then open http://127.0.0.1:8000/docs

This file only translates between HTTP and the logic that already exists (agents, calculators, graph). No finance logic lives here, so any client
(the Streamlit UI, curl, another program) gets identical behaviour.

Status codes used: 200 ok | 404 unknown ticker or route | 422 the request body/params are invalid (FastAPI or our own domain checks)
                   500 a bug on our side | 502 a data provider failed and nothing saved could stand in | 503 not ready / dependency down
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import load_settings
from src.core.logging import get_logger, setup_logging
from src.agents import goal_calc, portfolio_calc
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.core.guardrails import DISCLAIMER_STRONG
from src.data.errors import MarketDataError, ProviderUnavailable, TickerNotFound
from src.data.models import PriceHistory, Quote
from src.rag.hybrid_retriever import StaleKeywordIndexError
from src.workflow.factory import config_for
from src.web_app.models import (ChatRequest, ChatResponse, ErrorBody, GoalRequest, GoalResponse, HealthResponse, IndexQuote, KnowledgeResponse, MarketOverview, PassageOut,
                                PortfolioRequest, PortfolioResponse)
from src.web_app.services import Services

log = get_logger("finnie.api")
INDICES = {"^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite"}
TICKER = Path(pattern=r"^\^?[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="Ticker such as AAPL, BRK-B or ^GSPC (encode ^ as %5E in a URL)")
CORS_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]   # the Streamlit UI (Step 23); any other origin is refused by browsers


def _error(status: int, code: str, message: str, request: Request, details: list | None = None) -> JSONResponse:
    body = ErrorBody(code=code, message=message, request_id=request.state.request_id, details=details or [])
    return JSONResponse(status_code=status, content=body.model_dump(), headers={"X-Request-ID": request.state.request_id})


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="Finnie API", version="0.1.0", description="Educational finance tutor. Not financial advice.")
    app.state.services = services or Services()
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"],
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

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.error("req=%s unexpected %s: %s", request.state.request_id, type(error).__name__, str(error)[:200])   # full detail stays in the log, never in the reply
        return _error(500, "internal_error", "Something went wrong on our side. Quote this request id if you report it.", request)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        """Is the server up? Cheap on purpose: never calls a model, a data provider or the vector store."""
        return HealthResponse(status="ok", version=app.version, ready=request.app.state.services.ready())

    @app.get("/market/quote/{ticker}", response_model=Quote, tags=["market"])
    def quote(request: Request, ticker: str = TICKER):
        """Latest price. Carries its source, as_of date and is_stale flag: a stale value is returned (flagged) rather than an error when the provider is down."""
        return request.app.state.services.market.get_quote(ticker)

    @app.get("/market/history/{ticker}", response_model=PriceHistory, tags=["market"])
    def history(request: Request, ticker: str = TICKER, period: Literal["5d", "1mo", "3mo", "6mo", "1y", "ytd"] = Query("1mo")):
        """Daily closes for charting (used by the Markets tab in Step 23)."""
        return request.app.state.services.market.get_history(ticker, period)

    @app.get("/market/overview", response_model=MarketOverview, tags=["market"])
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

    @app.post("/goals/plan", response_model=GoalResponse, tags=["goals"])
    def plan_goal(body: GoalRequest, request: Request):
        """Three-scenario savings plan. Pure arithmetic with assumed returns from config.yaml: no model is called, so it is instant, free and repeatable."""
        config = request.app.state.services.settings.goal_planning
        plan = goal_calc.plan_goal(body.goal, body.years, body.current_savings, body.risk_profile, config)
        data = {k: getattr(plan, k) for k in ("goal", "years", "current_savings", "risk_profile", "inflation_pct", "inflation_adjusted_goal", "assumptions")}
        return GoalResponse(**data, scenarios=[s.__dict__ for s in plan.scenarios], flags=goal_calc.assess(plan, config.large_monthly_usd),
                            disclaimer="Educational illustration only, not financial advice. Real returns vary and can be negative.")

    @app.get("/knowledge/search", response_model=KnowledgeResponse, tags=["knowledge"])
    def search_knowledge(request: Request, q: str = Query(min_length=2, max_length=300), category: str | None = Query(None, max_length=40),
                         top_k: int = Query(5, ge=1, le=10)):
        """Search the knowledge base. Returns the matching passages with their sources; no model writes anything here."""
        found = request.app.state.services.retriever.retrieve(q, top_k=top_k, category=category)
        results = [PassageOut(doc_id=p.citation.doc_id, title=p.citation.title, section=p.citation.section, source_name=p.citation.source_name,
                              source_url=p.citation.source_url, last_updated=p.citation.last_updated, text=p.text, similarity=round(p.similarity, 3))
                   for p in found.passages]
        return KnowledgeResponse(query=q, answerable=found.answerable, results=results)

    @app.post("/portfolio/analyze", response_model=PortfolioResponse, tags=["portfolio"])
    def analyze_portfolio(body: PortfolioRequest, request: Request):
        """Explain a portfolio from CSV text. Bad CSV is the caller's fault (422 listing every problem); no prices at all is ours to report (502)."""
        services = request.app.state.services
        portfolio_calc.parse_portfolio_csv(body.csv_text)   # raises PortfolioInputError -> 422 with every problem, before any market call or model call
        agent = PortfolioAnalysisAgent(services.llm, services.market, services.metadata)   # one per request: the agent keeps per-call scratch state, so sharing one across threads is unsafe
        result = agent.run(body.csv_text, body.question)
        metrics = result.metadata.get("metrics")
        if metrics is None:
            raise ProviderUnavailable("I couldn't get prices for your holdings right now. Please try again in a few minutes.")
        return PortfolioResponse(text=result.text, disclaimer=result.disclaimer or DISCLAIMER_STRONG, sources=result.sources, metrics=metrics,
                                 cost_usd=result.metadata.get("cost_usd", 0.0), latency_s=result.metadata.get("latency_s", 0.0))

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(body: ChatRequest, request: Request):
        """One conversational turn through the full workflow (router, agents, planner for multi-part questions). Memory is kept per session_id."""
        services = request.app.state.services
        session_id = body.session_id or uuid.uuid4().hex[:16]
        state_in = {"message": body.message, "session_id": session_id, **({"portfolio_csv": body.portfolio_csv} if body.portfolio_csv else {})}
        start = time.perf_counter()
        with services.session_lock(session_id):   # two simultaneous turns of ONE conversation would interleave its saved history; different sessions never wait
            state = services.graph.invoke(state_in, config_for(session_id, services.settings.workflow.max_concurrency))
        reply = state["response"]
        trace = state.get("trace", [])
        routing = sum(t.get("cost_usd") or 0.0 for t in trace if t.get("turn") == state.get("turn") and t["node"] in ("contextualize", "router"))   # the agent's own cost is in its reply
        return ChatResponse(session_id=session_id, turn=state.get("turn", 1), text=reply["text"], answered_by=reply["agent_name"], route=state.get("route", ""),
                            sources=reply.get("sources", []), disclaimer=reply.get("disclaimer", ""),
                            cost_usd=round((reply.get("metadata", {}).get("cost_usd") or 0.0) + routing, 6),
                            latency_s=round(time.perf_counter() - start, 3), degraded=state.get("error") is not None)

    return app


setup_logging(load_settings().app.log_level)
app = create_app()
