"""The UI's only way to reach Finnie: HTTP calls to the API. No finance logic here or anywhere in the UI.

Every failure becomes one ApiError with a message that is safe to show, so a tab needs a single `except ApiError` and never sees httpx, a traceback or JSON.
"""
import os

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


class FinnieClient:
    def __init__(self, base_url: str = DEFAULT_URL, transport: httpx.BaseTransport | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self._http = httpx.Client(base_url=base_url, transport=transport, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
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
        return response.json()

    def health(self) -> dict:
        return self._call("GET", "/health")

    def chat(self, message: str, session_id: str | None = None, portfolio_csv: str | None = None) -> dict:
        body = {"message": message, **({"session_id": session_id} if session_id else {}), **({"portfolio_csv": portfolio_csv} if portfolio_csv else {})}
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
