"""API tests: FastAPI's TestClient calls the app in-process (no server, no network), with fake market data, retriever and models."""
import pytest
from fastapi.testclient import TestClient

from src.core.config import load_settings
from src.core.llm import LLMResult
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.market_data import MarketDataService
from src.rag.hybrid_retriever import StaleKeywordIndexError
from src.web_app.api import create_app
from src.web_app.services import Services
from tests.test_portfolio_agent import CSV, META, FakeProvider
from tests.test_workflow import make

SETTINGS = load_settings()


class FlakyProvider(FakeProvider):
    """FakeProvider that can fail for chosen symbols, or reject some as unknown."""
    def __init__(self, down=(), unknown=()):
        super().__init__(down)
        self.unknown = set(unknown)

    def get_quote(self, symbol):
        if symbol in self.unknown:
            raise TickerNotFound(f"I couldn't find a ticker called {symbol}.")
        return super().get_quote(symbol)


class ReplyLLM:
    def chat(self, prompt, system=None, **kwargs):
        return LLMResult("Your portfolio is split across two holdings.", "fake", 100, 10, 0.1, 0.002, 1)


class FakeRetriever:
    def __init__(self, passages=(), error=None):
        self.passages, self.error = list(passages), error

    def retrieve(self, question, top_k=None, category=None, **kw):
        if self.error:
            raise self.error
        from types import SimpleNamespace
        return SimpleNamespace(passages=self.passages[:top_k], answerable=bool(self.passages))


def passage():
    from src.rag.passages import Citation, Passage
    return Passage("c1", "An expense ratio is a yearly fee.", 0.61, Citation("etf-basics", "ETF Basics", "Costs", "Investor.gov", "https://x.example", "2026-10-30"))


def client(provider=None, retriever=None, graph=None, llm=None, raise_server_exceptions=False):
    market = MarketDataService(provider or FlakyProvider(), TTLCache(), SETTINGS, sleep=lambda s: None)
    services = Services(SETTINGS, market=market, retriever=retriever or FakeRetriever([passage()]), graph=graph, llm=llm or ReplyLLM())
    services._metadata = META
    return TestClient(create_app(services), raise_server_exceptions=raise_server_exceptions)


# ---------- system: health, request id, error shape, CORS ----------
def test_health_is_ok_and_reports_what_is_built():
    r = client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and set(r.json()["ready"]) == {"market", "retriever", "graph"}


def test_every_response_has_a_request_id_and_a_caller_supplied_one_is_kept():
    c = client()
    assert len(c.get("/health").headers["x-request-id"]) == 12
    assert c.get("/health", headers={"X-Request-ID": "trace-42"}).headers["x-request-id"] == "trace-42"


def test_a_hostile_request_id_is_replaced_not_echoed():
    r = client().get("/health", headers={"X-Request-ID": "x" * 200})
    assert r.headers["x-request-id"] != "x" * 200 and len(r.headers["x-request-id"]) == 12


def test_unknown_url_and_wrong_verb_use_the_standard_error_shape():
    c = client()
    for response, code, status in ((c.get("/nope"), "not_found", 404), (c.post("/health"), "method_not_allowed", 405)):
        body = response.json()
        assert response.status_code == status and body["code"] == code and body["request_id"] == response.headers["x-request-id"]


def test_cors_allows_the_streamlit_origin_and_not_others():
    c = client()
    assert c.get("/health", headers={"Origin": "http://localhost:8501"}).headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": "http://evil.example"}).headers


# ---------- market ----------
def test_quote_ok_and_lowercase_ticker_is_accepted():
    r = client().get("/market/quote/aaa")
    assert r.status_code == 200 and r.json()["symbol"] == "AAA" and r.json()["is_stale"] is False


def test_unknown_ticker_is_404_and_malformed_ticker_is_422():
    c = client(FlakyProvider(unknown={"ZZZZ"}))
    assert c.get("/market/quote/ZZZZ").status_code == 404 and c.get("/market/quote/ZZZZ").json()["code"] == "ticker_not_found"
    bad = c.get("/market/quote/DROP%20TABLE")
    assert bad.status_code == 422 and bad.json()["details"][0]["field"] == "path.ticker"


def test_provider_down_with_nothing_saved_is_502():
    r = client(FlakyProvider(down={"AAA"})).get("/market/quote/AAA")
    assert r.status_code == 502 and r.json()["code"] == "data_provider_unavailable"


def test_provider_down_but_a_saved_value_exists_is_200_and_flagged_stale():
    settings = SETTINGS.model_copy(update={"cache": SETTINGS.cache.model_copy(update={"quote_ttl_seconds": 0})})   # ttl 0: every entry is expired at once, only get_stale can serve it
    provider = FlakyProvider()
    market = MarketDataService(provider, TTLCache(), settings, sleep=lambda s: None)
    c = TestClient(create_app(Services(settings, market=market, retriever=FakeRetriever(), llm=ReplyLLM())))
    assert c.get("/market/quote/AAA").json()["is_stale"] is False
    provider.down = {"AAA"}
    r = c.get("/market/quote/AAA")
    assert r.status_code == 200 and r.json()["is_stale"] is True


def test_history_rejects_an_unsupported_period():
    r = client().get("/market/history/AAA?period=99y")
    assert r.status_code == 422 and r.json()["details"][0]["field"] == "query.period"


def test_overview_returns_the_indices_that_worked_and_lists_the_ones_that_did_not():
    r = client(FlakyProvider(down={"^DJI"})).get("/market/overview")
    body = r.json()
    assert r.status_code == 200 and len(body["indices"]) == 2 and body["unavailable"] == ["Dow Jones Industrial Average"]


# ---------- goals ----------
def test_goal_plan_returns_three_scenarios_and_the_assumptions():
    r = client().post("/goals/plan", json={"goal": 50000, "years": 5, "current_savings": 5000, "risk_profile": "medium"})
    body = r.json()
    assert r.status_code == 200 and [s["scenario"] for s in body["scenarios"]] == ["conservative", "expected", "optimistic"]
    assert "not forecasts" in body["assumptions"] and body["disclaimer"]


def test_goal_plan_validation_reports_every_bad_field_at_once():
    r = client().post("/goals/plan", json={"goal": -5, "years": "soon", "risk_profile": "yolo"})
    assert r.status_code == 422 and {d["field"] for d in r.json()["details"]} == {"goal", "years", "risk_profile"}


def test_goal_plan_rejects_a_non_json_body():
    assert client().post("/goals/plan", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


# ---------- knowledge ----------
def test_knowledge_search_returns_passages_with_their_source():
    body = client().get("/knowledge/search?q=expense ratio").json()
    assert body["answerable"] is True and body["results"][0]["source_name"] == "Investor.gov" and body["results"][0]["doc_id"] == "etf-basics"


def test_knowledge_search_with_no_match_is_200_with_answerable_false():
    body = client(retriever=FakeRetriever([])).get("/knowledge/search?q=pizza place").json()
    assert body == {"query": "pizza place", "answerable": False, "results": []}


def test_knowledge_search_bounds_are_enforced_and_a_stale_index_is_503():
    c = client()
    assert c.get("/knowledge/search?q=x").status_code == 422 and c.get("/knowledge/search?q=ok&top_k=50").status_code == 422
    stale = client(retriever=FakeRetriever(error=StaleKeywordIndexError("rebuild")))
    assert stale.get("/knowledge/search?q=expense ratio").status_code == 503


# ---------- portfolio ----------
def test_portfolio_analyze_returns_text_and_metrics():
    r = client().post("/portfolio/analyze", json={"csv_text": CSV})
    body = r.json()
    assert r.status_code == 200 and body["metrics"]["holding_count"] == 2 and body["disclaimer"] and body["sources"]


def test_portfolio_with_a_bad_csv_is_422_and_lists_every_problem_before_any_model_call():
    r = client(llm=object()).post("/portfolio/analyze", json={"csv_text": "ticker,shares\nAAA,abc"})   # object() would crash if the model were called
    assert r.status_code == 422 and r.json()["code"] == "invalid_portfolio" and r.json()["details"]


def test_portfolio_with_no_prices_at_all_is_502_not_a_fake_success():
    r = client(FlakyProvider(down={"AAA", "BBB"})).post("/portfolio/analyze", json={"csv_text": CSV})
    assert r.status_code == 502 and r.json()["code"] == "data_provider_unavailable"


def test_portfolio_rejects_an_oversized_payload():
    assert client().post("/portfolio/analyze", json={"csv_text": "a" * 20001}).status_code == 422


# ---------- chat ----------
def chat_client(**kwargs):
    graph, llm, agents = make(saver=True, **kwargs)
    return client(graph=graph), llm, agents


def test_chat_creates_a_session_and_the_second_turn_continues_it():
    c, *_ = chat_client(route=[{"route": "finance_qa", "confidence": "high"}, {"route": "finance_qa", "confidence": "high"}], standalone=["What is VOO?"])
    first = c.post("/chat", json={"message": "What is an ETF?"}).json()
    second = c.post("/chat", json={"message": "and VOO?", "session_id": first["session_id"]}).json()
    assert first["turn"] == 1 and first["answered_by"] == "finance_qa" and second["session_id"] == first["session_id"] and second["turn"] == 2


def test_chat_cost_counts_each_model_call_once():
    c, *_ = chat_client(route=[{"route": "finance_qa", "confidence": "high"}])
    body = c.post("/chat", json={"message": "What is an ETF?"}).json()
    assert body["cost_usd"] == pytest.approx(0.00005 + 0.001)    # one router call (fake: 0.00005) + the fake agent's 0.001, nothing double counted


def test_chat_out_of_scope_gets_the_fixed_reply():
    c, *_ = chat_client(route=[{"route": "out_of_scope", "confidence": "high"}])
    body = c.post("/chat", json={"message": "price of a pizza"}).json()
    assert body["answered_by"] == "scope_reply" and body["route"] == "out_of_scope" and body["degraded"] is False


def test_chat_validation_empty_message_and_bad_session_id():
    c, *_ = chat_client()
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "hi", "session_id": "bad id!"}).status_code == 422
    assert c.post("/chat", json={"message": "x" * 2001}).status_code == 422


def test_an_unexpected_exception_is_a_500_with_no_internals_in_the_reply():
    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("secret database password is hunter2")
    r = client(graph=Boom()).post("/chat", json={"message": "hello"})
    assert r.status_code == 500 and "hunter2" not in r.text and r.json()["code"] == "internal_error" and r.json()["request_id"] == r.headers["x-request-id"]


def test_different_sessions_get_different_locks_and_the_same_session_the_same_lock():
    services = client().app.state.services
    assert services.session_lock("a") is services.session_lock("a") and services.session_lock("a") is not services.session_lock("b")


# ---------- additions for the UI (Step 23) ----------
def test_goal_plan_includes_a_yearly_balance_path_per_scenario():
    body = client().post("/goals/plan", json={"goal": 50000, "years": 5, "current_savings": 5000}).json()
    for scenario in body["scenarios"]:
        path = scenario["path_by_year"]
        assert len(path) == 6 and path[0] == 5000 and path[-1] == pytest.approx(50000, rel=0.001)   # the plan's monthly amount lands on the goal


def test_portfolio_analyze_returns_the_correlation_matrix_for_a_heatmap():
    body = client().post("/portfolio/analyze", json={"csv_text": CSV}).json()
    corr = body["correlation"]
    assert corr["tickers"] == sorted(corr["tickers"]) or len(corr["tickers"]) == 2
    assert len(corr["matrix"]) == 2 and all(row[i] == 1.0 for i, row in enumerate(corr["matrix"]))


def test_articles_list_filter_and_fetch_one():
    from types import SimpleNamespace
    docs = [SimpleNamespace(id="a-1", text="# A\nbody", metadata={"title": "A", "category": "bonds", "source_url": "https://x"}),
            SimpleNamespace(id="b-2", text="# B", metadata={"title": "B", "category": "taxes"})]
    c = client()
    c.app.state.services._articles = docs
    assert [a["id"] for a in c.get("/knowledge/articles").json()] == ["a-1", "b-2"]
    assert [a["id"] for a in c.get("/knowledge/articles?category=taxes").json()] == ["b-2"]
    one = c.get("/knowledge/articles/a-1").json()
    assert one["text"].startswith("# A") and one["source_url"] == "https://x"
    assert c.get("/knowledge/articles/missing").status_code == 404 and c.get("/knowledge/articles/bad%20id").status_code == 422
