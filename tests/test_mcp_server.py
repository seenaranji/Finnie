"""Step 28: the MCP server, driven through the real MCP client (in-process, no network, no models, no cost)."""
import asyncio
import json
import logging
import sys

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.mcp_server import server as server_module
from src.mcp_server.server import build_server
from src.mcp_server.tools import FinnieTools
from src.web_app.services import Services
from tests.auth_support import settings_with_secret
from tests.test_api import FakeRetriever, FlakyProvider, ReplyLLM, passage
from tests.test_portfolio_agent import CSV, META

SETTINGS = settings_with_secret()


def tools(provider=None, retriever=None, **kw):
    market = MarketDataService(provider or FlakyProvider(), TTLCache(), SETTINGS, sleep=lambda s: None)
    services = Services(SETTINGS, market=market, retriever=retriever or FakeRetriever([passage()]), llm=ReplyLLM())
    services._metadata = META
    return FinnieTools(services, **kw)


def call(name, args, **kw):
    async def go():
        async with Client(build_server(tools(**kw))) as client:
            return await client.call_tool(name, args)
    return asyncio.run(go())


def data(result):
    return json.loads(result.content[0].text)


def test_the_four_tools_are_listed_with_their_inputs():
    async def go():
        async with Client(build_server(tools())) as client:
            return {t.name: t for t in (await client.list_tools()).tools}
    listed = asyncio.run(go())
    assert set(listed) == {"get_quote", "analyze_portfolio", "plan_goal", "search_knowledge"}
    assert set(listed["plan_goal"].input_schema["properties"]) == {"goal", "years", "current_savings", "risk_profile"}
    assert listed["plan_goal"].input_schema["required"] == ["goal", "years"]
    assert all(t.annotations.read_only_hint for t in listed.values())


def test_get_quote_matches_the_market_service():
    result = call("get_quote", {"ticker": "AAPL"})
    assert not result.is_error and data(result)["symbol"] == "AAPL" and data(result)["price"] > 0


@pytest.mark.parametrize("bad", ["", "A B", "../etc", "X" * 11, "AAPL;DROP"])
def test_get_quote_rejects_malformed_tickers_before_any_lookup(bad):
    result = call("get_quote", {"ticker": bad})
    assert result.is_error and "valid ticker" in result.content[0].text


def test_get_quote_unknown_and_down_have_plain_messages():
    assert "No ticker named ZZZZ" in call("get_quote", {"ticker": "ZZZZ"}, provider=FlakyProvider(unknown=["ZZZZ"])).content[0].text
    down = call("get_quote", {"ticker": "AAPL"}, provider=FlakyProvider(down=["AAPL"]))
    assert down.is_error and "unavailable" in down.content[0].text and "Traceback" not in down.content[0].text


def test_plan_goal_known_answer_and_same_as_the_api_arithmetic():
    from src.agents import goal_calc
    out = data(call("plan_goal", {"goal": 100000, "years": 10, "current_savings": 5000}))
    expected = goal_calc.plan_goal(100000, 10, 5000, "medium", SETTINGS.goal_planning)
    assert [s["monthly_needed"] for s in out["scenarios"]] == [s.monthly_needed for s in expected.scenarios]
    assert "not financial advice" in out["disclaimer"]


@pytest.mark.parametrize("args", [{"goal": 0, "years": 5}, {"goal": 1000, "years": -1}, {"goal": 1000, "years": 500},
                                  {"goal": 1000, "years": 5, "current_savings": -1}, {"goal": 1000, "years": 5, "risk_profile": "wild"},
                                  {"goal": "lots", "years": 5}])
def test_plan_goal_rejects_bad_input(args):
    assert call("plan_goal", args).is_error


def test_plan_goal_rejects_nan_and_infinity_directly():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ToolError):
            tools().plan_goal(bad, 5)


def test_search_knowledge_returns_passages_with_sources():
    out = data(call("search_knowledge", {"query": "expense ratio"}))
    assert out["answerable"] and out["results"][0]["doc_id"] == "etf-basics" and out["results"][0]["source_url"]


def test_search_knowledge_nothing_found_is_not_an_error():
    out = data(call("search_knowledge", {"query": "weather tomorrow"}, retriever=FakeRetriever([])))
    assert out["answerable"] is False and out["results"] == []


@pytest.mark.parametrize("args", [{"query": "x"}, {"query": "ok query", "top_k": 0}, {"query": "ok query", "top_k": 11}, {"query": "q" * 301}])
def test_search_knowledge_rejects_bad_input(args):
    assert call("search_knowledge", args).is_error


def test_analyze_portfolio_from_csv():
    out = data(call("analyze_portfolio", {"holdings_csv": CSV}))
    assert out["text"] and out["metrics"] and "advice" in out["disclaimer"].lower()


def test_analyze_portfolio_needs_exactly_one_source():
    for args in ({}, {"portfolio_id": "123e4567-e89b-12d3-a456-426614174000", "holdings_csv": CSV}):
        result = call("analyze_portfolio", args)
        assert result.is_error and "exactly one" in result.content[0].text


def test_analyze_portfolio_bad_csv_lists_the_problem():
    result = call("analyze_portfolio", {"holdings_csv": "ticker,shares,avg_cost\nAAPL,abc,1"})
    assert result.is_error and "Traceback" not in result.content[0].text


def test_saved_portfolio_without_credentials_explains_what_to_set():
    result = call("analyze_portfolio", {"portfolio_id": "123e4567-e89b-12d3-a456-426614174000"}, email="", password="")
    assert result.is_error and "FINNIE_EMAIL" in result.content[0].text


def test_saved_portfolio_with_a_malformed_id():
    result = call("analyze_portfolio", {"portfolio_id": "../../etc/passwd"}, email="a@b.co", password="x")
    assert result.is_error and "not a valid id" in result.content[0].text


def test_wrong_password_is_reported_without_echoing_it():
    class Auth:
        def login(self, *a):
            from src.auth.service import BadCredentials
            raise BadCredentials()
    t = tools(email="a@b.co", password="hunter2-secret")
    t.services._auth = Auth()
    with pytest.raises(ToolError) as e:
        t.analyze_portfolio(portfolio_id="123e4567-e89b-12d3-a456-426614174000")
    assert "hunter2-secret" not in str(e.value)


def test_main_moves_logging_off_stdout_so_the_protocol_stays_clean(monkeypatch):
    ran = []
    monkeypatch.setattr(server_module, "build_server", lambda: type("S", (), {"run": lambda self, t: ran.append(t)})())
    server_module.main()
    streams = [getattr(h, "stream", None) for h in logging.getLogger().handlers]
    assert ran == ["stdio"] and sys.stdout not in streams


def test_real_subprocess_over_stdio_from_another_folder():
    """What Claude Desktop does: start the launcher as a child process from an unrelated working directory and talk JSON-RPC over its pipes."""
    from pathlib import Path
    from mcp import StdioServerParameters

    async def go():
        params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve().parent.parent / "scripts" / "run_mcp_server.py")], cwd="/")
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            goal = await client.call_tool("plan_goal", {"goal": 1000, "years": 1})
            return names, goal
    names, goal = asyncio.run(go())
    assert names == {"get_quote", "analyze_portfolio", "plan_goal", "search_knowledge"} and not goal.is_error
