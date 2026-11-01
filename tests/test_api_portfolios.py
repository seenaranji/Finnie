"""Saved-portfolio endpoints, with the real repository on the throwaway PostgreSQL database and fake market data / model."""
import pytest
from fastapi.testclient import TestClient

from src.core.config import load_settings
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.portfolio_repository import PortfolioRepository
from src.web_app.api import create_app
from src.web_app.services import Services
from tests.test_api import FlakyProvider, ReplyLLM
from tests.test_portfolio_agent import CSV, META

HOLDINGS = [{"ticker": "AAA", "shares": "10", "avg_cost": "50"}, {"ticker": "BBB", "shares": "20", "avg_cost": "40"}]   # the same portfolio as CSV


@pytest.fixture
def api(pg):
    market = MarketDataService(FlakyProvider(), TTLCache(), load_settings(), sleep=lambda s: None)
    services = Services(load_settings(), market=market, llm=ReplyLLM(), portfolios=PortfolioRepository(pg))
    services._metadata = META
    return TestClient(create_app(services), raise_server_exceptions=False)


def test_save_then_get_by_id(api):
    created = api.post("/portfolios", json={"name": "Mine", "holdings": HOLDINGS})
    assert created.status_code == 201
    body = api.get(f"/portfolios/{created.json()['id']}").json()
    assert body["name"] == "Mine" and [(h["ticker"], h["shares"], h["avg_cost"]) for h in body["holdings"]] == [("AAA", "10", "50"), ("BBB", "20", "40")]


def test_list_shows_name_id_and_count(api):
    portfolio_id = api.post("/portfolios", json={"name": "Mine", "holdings": HOLDINGS}).json()["id"]
    assert [(p["id"], p["name"], p["holding_count"]) for p in api.get("/portfolios").json()] == [(portfolio_id, "Mine", 2)]


def test_missing_id_is_404_and_a_malformed_id_is_422(api):
    missing = api.get("/portfolios/00000000-0000-4000-8000-000000000000")
    assert missing.status_code == 404 and missing.json()["code"] == "portfolio_not_found" and missing.json()["request_id"]
    assert api.get("/portfolios/not-a-uuid").status_code == 422
    assert api.delete("/portfolios/00000000-0000-4000-8000-000000000000").status_code == 404
    assert api.put("/portfolios/00000000-0000-4000-8000-000000000000", json={"name": "x", "holdings": HOLDINGS}).status_code == 404


def test_invalid_holdings_are_422_and_name_the_field(api):
    r = api.post("/portfolios", json={"name": "Bad", "holdings": [{"ticker": "", "shares": "-1", "avg_cost": "5"}]})
    assert r.status_code == 422 and r.json()["code"] == "invalid_request"
    assert {d["field"] for d in r.json()["details"]} >= {"holdings.0.ticker", "holdings.0.shares"}
    assert api.get("/portfolios").json() == []   # nothing was saved


def test_put_replaces_and_delete_removes(api):
    portfolio_id = api.post("/portfolios", json={"name": "Old", "holdings": HOLDINGS}).json()["id"]
    assert api.put(f"/portfolios/{portfolio_id}", json={"name": "New", "holdings": HOLDINGS[:1]}).status_code == 204
    after = api.get(f"/portfolios/{portfolio_id}").json()
    assert after["name"] == "New" and len(after["holdings"]) == 1
    assert api.delete(f"/portfolios/{portfolio_id}").status_code == 204
    assert api.get(f"/portfolios/{portfolio_id}").status_code == 404


def test_analysis_by_id_gives_the_same_numbers_as_the_csv_route(api):
    portfolio_id = api.post("/portfolios", json={"name": "Same", "holdings": HOLDINGS}).json()["id"]
    by_id = api.get(f"/portfolios/{portfolio_id}/analysis")
    by_csv = api.post("/portfolio/analyze", json={"csv_text": CSV})
    assert by_id.status_code == 200 and by_id.json()["metrics"] == by_csv.json()["metrics"]
    assert by_id.json()["correlation"] == by_csv.json()["correlation"]


def test_analysis_of_a_missing_id_is_404_before_any_market_or_model_call(pg):
    services = Services(load_settings(), llm=object(), market=object(), portfolios=PortfolioRepository(pg))   # object() would crash if used
    r = TestClient(create_app(services), raise_server_exceptions=False).get("/portfolios/00000000-0000-4000-8000-000000000000/analysis")
    assert r.status_code == 404


def test_database_down_is_a_503_with_the_usual_error_shape(api):
    from psycopg import OperationalError

    class Down:
        def list(self):
            raise OperationalError("connection refused")
    api.app.state.services._portfolios = Down()
    r = api.get("/portfolios")
    assert r.status_code == 503 and r.json()["code"] == "database_unavailable" and "refused" not in r.text
