"""Saved-portfolio endpoints, with the real repository on the throwaway PostgreSQL database and fake market data / model."""
import pytest
from fastapi.testclient import TestClient

from src.core.config import load_settings
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.portfolio_repository import PortfolioRepository
from src.web_app.api import create_app
from src.web_app.services import Services
from tests.auth_support import ALICE, BOB, headers_for, settings_with_secret
from tests.test_api import FlakyProvider, ReplyLLM
from tests.test_portfolio_agent import CSV, META

HOLDINGS = [{"ticker": "AAA", "shares": "10", "avg_cost": "50"}, {"ticker": "BBB", "shares": "20", "avg_cost": "40"}]   # the same portfolio as CSV


@pytest.fixture
def api(pg, two_users):
    market = MarketDataService(FlakyProvider(), TTLCache(), settings_with_secret(), sleep=lambda s: None)
    services = Services(settings_with_secret(), market=market, llm=ReplyLLM(), portfolios=PortfolioRepository(pg))
    services._metadata = META
    return TestClient(create_app(services), raise_server_exceptions=False, headers=headers_for(ALICE))


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


def test_analysis_of_a_missing_id_is_404_before_any_market_or_model_call(pg, two_users):
    services = Services(settings_with_secret(), llm=object(), market=object(), portfolios=PortfolioRepository(pg))   # object() would crash if used
    r = TestClient(create_app(services), raise_server_exceptions=False, headers=headers_for(ALICE)).get("/portfolios/00000000-0000-4000-8000-000000000000/analysis")
    assert r.status_code == 404


def test_database_down_is_a_503_with_the_usual_error_shape(api):
    from psycopg import OperationalError

    class Down:
        def list(self, *, owner_id):
            raise OperationalError("connection refused")
    api.app.state.services._portfolios = Down()
    r = api.get("/portfolios")
    assert r.status_code == 503 and r.json()["code"] == "database_unavailable" and "refused" not in r.text


# ---------- Step 27: one user cannot touch another's portfolios through the API ----------
def test_bob_cannot_see_change_analyze_or_delete_alices_portfolio(api):
    pid = api.post("/portfolios", json={"name": "alices", "holdings": HOLDINGS}).json()["id"]
    bob = {"headers": headers_for(BOB)}
    assert api.get(f"/portfolios/{pid}", **bob).status_code == 404
    assert api.get(f"/portfolios/{pid}/analysis", **bob).status_code == 404
    assert api.put(f"/portfolios/{pid}", json={"name": "hijack", "holdings": HOLDINGS[:1]}, **bob).status_code == 404
    assert api.delete(f"/portfolios/{pid}", **bob).status_code == 404
    assert api.get("/portfolios", **bob).json() == []
    assert api.get(f"/portfolios/{pid}").json()["name"] == "alices"   # still Alice's, unchanged
    assert api.get(f"/portfolios/{pid}", **bob).json()["code"] == api.get("/portfolios/00000000-0000-4000-8000-000000000000", **bob).json()["code"]   # same answer as "no such id"


# ---------- typed-in portfolios and the grouped summary ----------
TYPED = [{"ticker": "AAA", "asset_type": "stock", "shares": "10", "avg_cost": "50", "purchase_date": "2024-01-02"},     # price 100: cost 500, value 1000
         {"ticker": "BBB", "asset_type": "bond", "shares": "20", "avg_cost": "40"},                                     # price 50:  cost 800, value 1000
         {"ticker": "CASH", "asset_type": "cash", "shares": "500", "avg_cost": "1"}]                                    # always $1: cost 500, value 500


def save_typed(api, holdings=TYPED, name="typed"):
    return api.post("/portfolios", json={"name": name, "holdings": holdings}).json()["id"]


def test_the_kind_and_date_are_stored_and_returned_with_a_unique_id(api):
    one, two = save_typed(api), save_typed(api)
    assert one != two and len(one) == 36                                       # a new UUID every time
    body = api.get(f"/portfolios/{one}").json()
    assert [(h["ticker"], h["asset_type"], h["purchase_date"]) for h in body["holdings"]] == [("AAA", "stock", "2024-01-02"), ("BBB", "bond", None), ("CASH", "cash", None)]


def test_an_unknown_kind_or_a_future_date_is_422_naming_the_field(api):
    bad_kind = api.post("/portfolios", json={"name": "x", "holdings": [{**TYPED[0], "asset_type": "unicorn"}]})
    future = api.post("/portfolios", json={"name": "x", "holdings": [{**TYPED[0], "purchase_date": "2999-01-01"}]})
    assert bad_kind.status_code == future.status_code == 422
    assert "holdings.0.asset_type" in {d["field"] for d in bad_kind.json()["details"]} and "holdings.0.purchase_date" in {d["field"] for d in future.json()["details"]}
    assert api.get("/portfolios").json() == []


def test_the_summary_groups_by_kind_with_cost_value_gain_and_weight(api):
    body = api.get(f"/portfolios/{save_typed(api)}/summary").json()
    assert (body["total_cost"], body["total_value"], body["holding_count"], body["unpriced"]) == (1800.0, 2500.0, 3, [])
    groups = {g["asset_type"]: g for g in body["groups"]}
    assert (groups["stock"]["cost"], groups["stock"]["value"], groups["stock"]["gain_pct"], groups["stock"]["weight_pct"]) == (500.0, 1000.0, 100.0, 40.0)
    assert (groups["bond"]["cost"], groups["bond"]["value"], groups["bond"]["gain_pct"]) == (800.0, 1000.0, 25.0)
    assert (groups["cash"]["value"], groups["cash"]["gain_pct"], groups["cash"]["weight_pct"]) == (500.0, 0.0, 20.0)
    assert body["prices_as_of"] == "2026-10-30" and body["name"] == "typed"


def test_a_ticker_that_cannot_be_priced_is_listed_not_hidden_and_does_not_break_the_page(pg, two_users):
    market = MarketDataService(FlakyProvider(down={"BBB"}), TTLCache(), settings_with_secret(), sleep=lambda s: None)
    client = TestClient(create_app(Services(settings_with_secret(), market=market, llm=ReplyLLM(), portfolios=PortfolioRepository(pg))), raise_server_exceptions=False, headers=headers_for(ALICE))
    body = client.get(f"/portfolios/{save_typed(client)}/summary").json()
    assert body["unpriced"] == ["BBB"] and body["total_cost"] == 1800.0 and body["total_value"] == 1500.0
    bond = {g["asset_type"]: g for g in body["groups"]}["bond"]
    assert bond["cost"] == 800.0 and bond["value"] == 0.0 and bond["weight_pct"] is None and bond["unpriced_count"] == 1


def test_a_market_that_is_totally_down_still_shows_the_portfolio_and_values_the_cash(pg, two_users):
    market = MarketDataService(FlakyProvider(down={"AAA", "BBB"}), TTLCache(), settings_with_secret(), sleep=lambda s: None)
    client = TestClient(create_app(Services(settings_with_secret(), market=market, llm=ReplyLLM(), portfolios=PortfolioRepository(pg))), raise_server_exceptions=False, headers=headers_for(ALICE))
    r = client.get(f"/portfolios/{save_typed(client)}/summary")
    assert r.status_code == 200 and r.json()["total_value"] == 500.0 and sorted(r.json()["unpriced"]) == ["AAA", "BBB"]


def test_bob_cannot_see_alices_summary(api):
    pid = save_typed(api)
    assert api.get(f"/portfolios/{pid}/summary", headers=headers_for(BOB)).status_code == 404
    assert api.get("/portfolios/00000000-0000-4000-8000-000000000000/summary").status_code == 404


def test_the_explanation_uses_the_kinds_the_user_entered(api):
    pid = save_typed(api, [{"ticker": "AAA", "asset_type": "bond", "shares": "10", "avg_cost": "50"}, {"ticker": "BBB", "asset_type": "stock", "shares": "20", "avg_cost": "40"}])
    classes = api.get(f"/portfolios/{pid}/analysis").json()["metrics"]["asset_class_pct"]
    assert set(classes) == {"Bond", "Stock"}                 # the hand-written list says AAA is a 'US stock'; the user said bond, and the user wins


# ---------- real estate ----------
MAPLE = {"name": "Maple St", "property_type": "house", "purchase_price": "300000", "purchase_date": "2019-06-01", "current_value": "400000",
         "mortgage_balance": "200000", "monthly_rent": "2500", "monthly_expenses": "800"}
OAK = {"name": "Oak condo", "property_type": "condo", "purchase_price": "150000", "current_value": "140000", "mortgage_balance": "90000", "monthly_rent": "1000", "monthly_expenses": "450"}


def save_mixed(api, properties=(MAPLE, OAK), holdings=TYPED):
    return api.post("/portfolios", json={"name": "everything", "holdings": list(holdings), "properties": list(properties)}).json()["id"]


def test_properties_are_saved_with_the_portfolio_and_listed_with_their_count(api):
    pid = save_mixed(api)
    body = api.get(f"/portfolios/{pid}").json()
    assert [p["name"] for p in body["properties"]] == ["Maple St", "Oak condo"] and body["properties"][0]["current_value"] == "400000"
    [row] = api.get("/portfolios").json()
    assert (row["holding_count"], row["property_count"]) == (3, 2)


def test_the_summary_shows_real_estate_as_a_group_with_equity_yields_and_the_mortgage(api):
    body = api.get(f"/portfolios/{save_mixed(api)}/summary").json()
    # market part: AAA 1,000 + BBB 1,000 + cash 500 = 2,500 (cost 1,800); real estate: value 540,000, purchase 450,000, mortgage 290,000
    assert (body["total_value"], body["total_cost"], body["mortgage_owed"], body["net_value"], body["property_count"], body["holding_count"]) == (542500.0, 451800.0, 290000.0, 252500.0, 2, 5)
    groups = {g["asset_type"]: g for g in body["groups"]}
    assert body["groups"][0]["asset_type"] == "real_estate" and groups["real_estate"]["label"] == "Real estate" and groups["real_estate"]["weight_pct"] == 99.54
    re = body["real_estate"]
    assert (re["equity"], re["gross_yield_pct"], re["net_yield_pct"], re["loan_to_value_pct"], re["gain_pct"]) == (250000.0, 7.78, 5.0, 53.7, 20.0)
    assert [(p["name"], p["equity"], p["net_operating_income"]) for p in re["properties"]] == [("Maple St", 200000.0, 20400.0), ("Oak condo", 50000.0, 6600.0)]


def test_a_portfolio_without_properties_has_no_real_estate_section(api):
    body = api.get(f"/portfolios/{save_typed(api)}/summary").json()
    assert body["real_estate"] is None and body["mortgage_owed"] == 0.0 and body["net_value"] == body["total_value"]


def test_properties_need_no_market_data_so_a_down_market_still_shows_them(pg, two_users):
    market = MarketDataService(FlakyProvider(down={"AAA", "BBB"}), TTLCache(), settings_with_secret(), sleep=lambda s: None)
    client = TestClient(create_app(Services(settings_with_secret(), market=market, llm=ReplyLLM(), portfolios=PortfolioRepository(pg))), raise_server_exceptions=False, headers=headers_for(ALICE))
    pid = client.post("/portfolios", json={"name": "rentals", "properties": [MAPLE]}).json()["id"]          # only properties, no holdings key at all
    body = client.get(f"/portfolios/{pid}/summary").json()
    assert body["total_value"] == 400000.0 and body["unpriced"] == [] and body["net_value"] == 200000.0


def test_explaining_a_real_estate_only_portfolio_says_why_not_a_crash(api):
    pid = api.post("/portfolios", json={"name": "rentals", "properties": [MAPLE]}).json()["id"]
    r = api.get(f"/portfolios/{pid}/analysis")
    assert r.status_code == 422 and r.json()["code"] == "nothing_to_explain" and "real estate" in r.json()["message"]


def test_explaining_a_mixed_portfolio_covers_the_market_holdings(api):
    r = api.get(f"/portfolios/{save_mixed(api)}/analysis")
    assert r.status_code == 200 and r.json()["metrics"]["holding_count"] == 3


def test_bad_property_figures_are_422_naming_the_field_and_nothing_is_saved(api):
    r = api.post("/portfolios", json={"name": "x", "properties": [{**MAPLE, "monthly_rent": "-5"}, {**MAPLE, "name": "B", "property_type": "castle"}]})
    fields = {d["field"] for d in r.json()["details"]}
    assert r.status_code == 422 and {"properties.0.monthly_rent", "properties.1.property_type"} <= fields and api.get("/portfolios").json() == []
    assert api.post("/portfolios", json={"name": "x"}).status_code == 422                                   # neither holdings nor properties


def test_put_is_a_full_replacement_including_properties(api):
    pid = save_mixed(api)
    assert api.put(f"/portfolios/{pid}", json={"name": "everything", "holdings": TYPED[:1]}).status_code == 204     # properties left out: removed
    assert api.get(f"/portfolios/{pid}").json()["properties"] == []
    assert api.put(f"/portfolios/{pid}", json={"name": "x", "holdings": TYPED[:1], "properties": [OAK]}).status_code == 204
    assert [p["name"] for p in api.get(f"/portfolios/{pid}").json()["properties"]] == ["Oak condo"]


def test_bob_cannot_see_alices_properties(api):
    pid = save_mixed(api)
    bob = {"headers": headers_for(BOB)}
    assert api.get(f"/portfolios/{pid}", **bob).status_code == 404 and api.get(f"/portfolios/{pid}/summary", **bob).status_code == 404
