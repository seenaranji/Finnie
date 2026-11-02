"""The four Finnie tools, as plain functions. server.py only registers them with MCP; the tests call them directly.

They reuse the same code as the HTTP API (market service, goal_calc, the retriever, PortfolioAnalysisAgent), so the numbers are identical. Every failure becomes a ToolError with a plain message the assistant can read to the user: no stack traces, no secrets.
"""
import math
import os
import re
from uuid import UUID

from mcp.server.mcpserver.exceptions import ToolError

from src.agents import goal_calc, portfolio_calc
from src.agents.portfolio_agent import NothingToAnalyze, PortfolioAnalysisAgent
from src.auth.security import AuthInputError, TooManyAttempts
from src.auth.service import BadCredentials
from src.core.guardrails import DISCLAIMER_STRONG
from src.data.errors import MarketDataError, TickerNotFound
from src.data.portfolio_repository import PortfolioNotFound
from src.web_app.services import Services

TICKER_RE = re.compile(r"^\^?[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")   # same rule as the API's path parameter
GOAL_DISCLAIMER = "Educational illustration only, not financial advice. Real returns vary and can be negative."
MAX_AMOUNT, MAX_YEARS = 1e12, 80   # same ceilings as the API's GoalRequest


class FinnieTools:
    def __init__(self, services: Services | None = None, email: str | None = None, password: str | None = None) -> None:
        self.services = services or Services()
        self._email = email if email is not None else os.environ.get("FINNIE_EMAIL", "")
        self._password = password if password is not None else os.environ.get("FINNIE_PASSWORD", "")
        self._owner_id = None

    def _owner(self):
        """Saved portfolios belong to a user. Log in once, with the account named in FINNIE_EMAIL / FINNIE_PASSWORD, and keep only the id."""
        if self._owner_id is None:
            if not (self._email and self._password):
                raise ToolError("Saved portfolios need an account: set FINNIE_EMAIL and FINNIE_PASSWORD in the MCP client's config. "
                                "Or pass holdings_csv instead of portfolio_id.")
            try:
                self._owner_id = self.services.auth.login(self._email, self._password, "mcp-stdio").id
            except (BadCredentials, AuthInputError):
                raise ToolError("Finnie could not sign in with FINNIE_EMAIL / FINNIE_PASSWORD (wrong email or password).") from None
            except TooManyAttempts:
                raise ToolError("Too many failed sign-ins. Wait a few minutes and try again.") from None
        return self._owner_id

    def get_quote(self, ticker: str) -> dict:
        ticker = (ticker or "").strip()
        if not TICKER_RE.match(ticker):
            raise ToolError("That is not a valid ticker. Use something like AAPL, BRK-B or ^GSPC.")
        try:
            return self.services.market.get_quote(ticker).model_dump(mode="json")
        except TickerNotFound:
            raise ToolError(f"No ticker named {ticker.upper()} was found.") from None
        except MarketDataError:
            raise ToolError("Market data is unavailable right now. Please try again in a few minutes.") from None

    def analyze_portfolio(self, portfolio_id: str | None = None, holdings_csv: str | None = None, question: str = "Please explain my portfolio in simple terms.") -> dict:
        if bool(portfolio_id) == bool(holdings_csv):
            raise ToolError("Give exactly one of portfolio_id (a saved portfolio) or holdings_csv (CSV text with the columns ticker, shares, avg_cost).")
        if not 1 <= len(question) <= 500:
            raise ToolError("question must be between 1 and 500 characters.")
        s = self.services
        try:
            if holdings_csv:
                if len(holdings_csv) > 20000:
                    raise ToolError("holdings_csv is too long (limit 20,000 characters).")
                portfolio_calc.parse_portfolio_csv(holdings_csv)   # every problem listed, before any market or model call
                result = PortfolioAnalysisAgent(s.llm, s.market, s.metadata).run(holdings_csv, question)
            else:
                try:
                    pid = UUID(portfolio_id)
                except ValueError:
                    raise ToolError("portfolio_id is not a valid id (it looks like 123e4567-e89b-12d3-a456-426614174000).") from None
                owner = self._owner()
                result = PortfolioAnalysisAgent(s.llm, s.market, s.metadata, repository=s.portfolios).run_by_id(pid, owner, question)
        except portfolio_calc.PortfolioInputError as error:
            raise ToolError(str(error)) from None
        except PortfolioNotFound:
            raise ToolError("No saved portfolio with that id for this account.") from None
        except NothingToAnalyze:
            raise ToolError("That portfolio has no market holdings to explain (real estate and cash are not analyzed).") from None
        metrics = result.metadata.get("metrics")
        if metrics is None:
            raise ToolError("Prices for the holdings are unavailable right now. Please try again in a few minutes.")
        return {"text": result.text, "disclaimer": result.disclaimer or DISCLAIMER_STRONG, "sources": result.sources, "metrics": metrics,
                "correlation": result.metadata.get("correlation"), "cost_usd": result.metadata.get("cost_usd", 0.0)}

    def plan_goal(self, goal: float, years: float, current_savings: float = 0, risk_profile: str = "medium") -> dict:
        if not (math.isfinite(goal) and 0 < goal <= MAX_AMOUNT and math.isfinite(years) and 0 < years <= MAX_YEARS
                and math.isfinite(current_savings) and 0 <= current_savings <= MAX_AMOUNT):
            raise ToolError(f"goal and years must be above 0 (years at most {MAX_YEARS}); current_savings must be 0 or more.")
        if risk_profile not in ("low", "medium", "high"):
            raise ToolError("risk_profile must be low, medium or high.")
        try:
            plan = goal_calc.plan_goal(goal, years, current_savings, risk_profile, self.services.settings.goal_planning)
        except goal_calc.GoalInputError as error:
            raise ToolError(str(error)) from None
        data = {k: getattr(plan, k) for k in ("goal", "years", "current_savings", "risk_profile", "inflation_pct", "inflation_adjusted_goal", "assumptions")}
        return {**data, "scenarios": [s.__dict__ for s in plan.scenarios],
                "flags": goal_calc.assess(plan, self.services.settings.goal_planning.large_monthly_usd), "disclaimer": GOAL_DISCLAIMER}

    def search_knowledge(self, query: str, category: str | None = None, top_k: int = 5) -> dict:
        if not 2 <= len((query or "").strip()) <= 300:
            raise ToolError("query must be between 2 and 300 characters.")
        if not 1 <= top_k <= 10:
            raise ToolError("top_k must be between 1 and 10.")
        found = self.services.retriever.retrieve(query, top_k=top_k, category=category)
        return {"query": query, "answerable": found.answerable,
                "results": [{"doc_id": p.citation.doc_id, "title": p.citation.title, "section": p.citation.section, "source_name": p.citation.source_name,
                             "source_url": p.citation.source_url, "last_updated": p.citation.last_updated, "text": p.text, "similarity": round(p.similarity, 3)}
                            for p in found.passages]}
