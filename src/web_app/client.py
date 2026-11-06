"""The UI's only way to reach Finnie: HTTP calls to the API. No finance logic here or anywhere in the UI.

Every failure becomes one ApiError with a message that is safe to show, so a tab needs a single `except ApiError` and never sees httpx, a traceback or JSON.
"""
import os
from uuid import UUID

import httpx

DEFAULT_URL = os.environ.get("FINNIE_API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0, code: str = "", request_id: str = "", details: list | None = None) -> None:
        super().__init__(message)
        self.message, self.status, self.code, self.request_id, self.details = message, status, code, request_id, details or []

    @property
    def retryable(self) -> bool:
        """5xx or no connection: trying again later may work. A 4xx will fail the same way until the input changes."""
        return self.status == 0 or self.status >= 500


def portfolio_id_or_error(text: str) -> str:
    """A portfolio id typed or pasted by a person goes into a URL, so it must be a real UUID and nothing else (not '../auth/me', not an empty string)."""
    try:
        return str(UUID(str(text).strip()))
    except ValueError:
        raise ApiError("That doesn't look like a portfolio id. It is 36 characters, like 3f2b8c1e-5d4a-4c7e-9b1a-0a1b2c3d4e5f.", 422, "invalid_portfolio_id") from None


class FinnieClient:
    def __init__(self, base_url: str = DEFAULT_URL, transport: httpx.BaseTransport | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.token: str | None = None   # set after login; sent as 'Authorization: Bearer <token>' on every call
        self._http = httpx.Client(base_url=base_url, transport=transport, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs):
        try:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            response = self._http.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException:
            raise ApiError("The server took too long to answer. Please try again.") from None
        except httpx.HTTPError:
            raise ApiError(f"I can't reach the Finnie server at {self.base_url}. Is it running?") from None
        if response.status_code >= 400:
            try:
                body = response.json()
                raise ApiError(body["message"], response.status_code, body.get("code", ""), body.get("request_id", ""), body.get("details", []))
            except (ValueError, KeyError):
                raise ApiError("The server sent a reply I did not understand.", response.status_code) from None
        return response.json() if response.content else None   # 204 No Content (delete / replace) has no body

    def health(self) -> dict:
        return self._call("GET", "/health")

    def register(self, email: str, password: str) -> dict:
        return self._call("POST", "/auth/register", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> dict:
        return self._call("POST", "/auth/login", json={"email": email, "password": password})

    def save_portfolio(self, name: str, holdings: list[dict], properties: list[dict] | None = None) -> dict:
        return self._call("POST", "/portfolios", json={"name": name, "holdings": holdings, **({"properties": properties} if properties else {})})

    def update_portfolio(self, portfolio_id: str, name: str, holdings: list[dict], properties: list[dict] | None = None) -> None:
        """Replace a saved portfolio's name, holdings and properties (an empty properties list removes them all)."""
        self._call("PUT", f"/portfolios/{portfolio_id_or_error(portfolio_id)}", json={"name": name, "holdings": holdings, "properties": properties or []})

    def list_portfolios(self) -> list[dict]:
        return self._call("GET", "/portfolios")

    def portfolio_summary(self, portfolio_id: str) -> dict:
        return self._call("GET", f"/portfolios/{portfolio_id_or_error(portfolio_id)}/summary")

    def analyze_saved_portfolio(self, portfolio_id: str, question: str | None = None) -> dict:
        return self._call("GET", f"/portfolios/{portfolio_id_or_error(portfolio_id)}/analysis", params={"question": question} if question else None)

    def get_portfolio(self, portfolio_id: str) -> dict:
        return self._call("GET", f"/portfolios/{portfolio_id_or_error(portfolio_id)}")

    def mortgage_payoff(self, portfolio_id: str, extra: float = 0) -> dict:
        return self._call("GET", f"/portfolios/{portfolio_id_or_error(portfolio_id)}/mortgage-payoff", params={"extra": repr(float(extra))})

    def set_mortgage_terms(self, portfolio_id: str, property_name: str, monthly_payment: float, rate_pct: float) -> None:
        self._call("PUT", f"/portfolios/{portfolio_id_or_error(portfolio_id)}/mortgage-terms",
                   json={"property": property_name, "monthly_payment": repr(float(monthly_payment)), "rate_pct": repr(float(rate_pct))})

    def delete_portfolio(self, portfolio_id: str) -> None:
        self._call("DELETE", f"/portfolios/{portfolio_id_or_error(portfolio_id)}")

    def chat(self, message: str, session_id: str | None = None, portfolio_csv: str | None = None, portfolio_id: str | None = None) -> dict:
        body = {"message": message, **({"portfolio_id": portfolio_id} if portfolio_id else {}), **({"session_id": session_id} if session_id else {}), **({"portfolio_csv": portfolio_csv} if portfolio_csv else {})}
        return self._call("POST", "/chat", json=body)

    def quote(self, ticker: str) -> dict:
        return self._call("GET", f"/market/quote/{ticker.strip().replace('^', '%5E')}")

    def history(self, ticker: str, period: str = "1mo") -> dict:
        return self._call("GET", f"/market/history/{ticker.strip().replace('^', '%5E')}", params={"period": period})

    def overview(self) -> dict:
        return self._call("GET", "/market/overview")

    def market_news(self) -> dict:
        return self._call("GET", "/news/market")

    def ticker_news(self, ticker: str) -> dict:
        return self._call("GET", f"/news/ticker/{ticker.strip()}")

    def plan_goal(self, goal: float, years: float, current_savings: float, risk_profile: str) -> dict:
        return self._call("POST", "/goals/plan", json={"goal": goal, "years": years, "current_savings": current_savings, "risk_profile": risk_profile})

    def analyze_portfolio(self, csv_text: str, question: str | None = None) -> dict:
        return self._call("POST", "/portfolio/analyze", json={"csv_text": csv_text, **({"question": question} if question else {})})

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> dict:
        return self._call("GET", "/knowledge/search", params={"q": query, "top_k": top_k, **({"category": category} if category else {})})

    def articles(self, category: str | None = None) -> list:
        return self._call("GET", "/knowledge/articles", params={"category": category} if category else None)

    def article(self, doc_id: str) -> dict:
        return self._call("GET", f"/knowledge/articles/{doc_id}")
