import json

from src.agents.market_agent import MAX_STEPS, MarketAnalysisAgent
from src.agents.market_tools import MarketTools
from src.core.config import load_settings
from src.core.llm import LLMTurn, ToolCall
from src.core.schemas import AgentRequest
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable, TickerNotFound
from src.data.market_data import MarketDataService
from src.data.models import PriceHistory, PricePoint, Quote
from src.data.providers.base import MarketDataProvider


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, bars=260):
        self.bars = bars

    def get_quote(self, symbol):
        if symbol == "FAKE":
            raise TickerNotFound(f"I couldn't find a ticker called {symbol}.")
        if symbol == "DOWN":
            raise ProviderUnavailable("The market data service is not responding right now.")
        return Quote(symbol=symbol, price=110.0, previous_close=100.0, change_pct=10.0, source=self.name, as_of="2026-10-30")

    def get_history(self, symbol, period):
        return PriceHistory(symbol=symbol, period=period, source=self.name,
                            points=[PricePoint(date=f"2026-01-{i % 28 + 1:02d}", close=float(100 + i % 5)) for i in range(self.bars)])


def make_tools(bars=260):
    return MarketTools(MarketDataService(FakeProvider(bars), TTLCache(), load_settings(), sleep=lambda s: None))


class ScriptedLLM:
    """Plays back a list of turns. A turn is either text or a list of (tool, arguments)."""
    def __init__(self, *turns):
        self.turns, self.seen_messages = list(turns), []

    def complete(self, messages, tools=None, **kwargs):
        self.seen_messages.append(json.loads(json.dumps(messages)))
        turn = self.turns.pop(0) if self.turns else [("get_quote", {"symbol": "AAPL"})]   # endless tool requests if the script runs out
        if isinstance(turn, str):
            return LLMTurn(turn, "fake", 10, 5, 0.1, 0.001, 1)
        calls = [ToolCall(f"call_{i}", name, args, json.dumps(args)) for i, (name, args) in enumerate(turn)]
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}} for c in calls]}
        return LLMTurn("", "fake", 10, 5, 0.1, 0.001, 1, tool_calls=calls, assistant_message=message)


def ask(llm, tools=None, message="how is AAPL doing"):
    return MarketAnalysisAgent(llm, tools or make_tools()).handle(AgentRequest(message=message))


def test_tool_result_goes_back_to_the_model_and_the_answer_is_returned():
    llm = ScriptedLLM([("get_quote", {"symbol": "AAPL"})], "AAPL is $110.0 [fake].")
    response = ask(llm)
    assert response.text == "AAPL is $110.0 [fake]."
    tool_message = llm.seen_messages[1][-1]   # what the model saw on its second turn
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_0"
    assert json.loads(tool_message["content"])["price"] == 110.0
    assert response.metadata["tool_names"] == ["get_quote"] and response.metadata["llm_calls"] == 2
    assert [s.id for s in response.sources] == ["fake:AAPL"]


def test_no_tool_needed_means_one_model_call():
    llm = ScriptedLLM("An ETF is a basket of investments.")
    response = ask(llm, message="what is an ETF")
    assert response.metadata["tool_names"] == [] and response.metadata["llm_calls"] == 1


def test_unknown_ticker_and_provider_outage_become_tool_errors_not_exceptions():
    llm = ScriptedLLM([("get_quote", {"symbol": "FAKE"}), ("get_quote", {"symbol": "DOWN"})], "I could not get those.")
    response = ask(llm)
    results = [t["result"] for t in response.metadata["tool_trace"]]
    assert [r["kind"] for r in results] == ["ticker_not_found", "unavailable"]
    assert response.sources == []   # errors are not sources


def test_bad_tool_arguments_and_unknown_tools_are_reported_to_the_model():
    tools = make_tools()
    assert tools.run("get_quote", None)[0]["kind"] == "bad_request"          # the model sent invalid JSON
    assert tools.run("get_quote", {"symbol": 5})[0]["kind"] == "bad_request"
    assert tools.run("delete_everything", {"symbol": "A"})[0]["kind"] == "bad_request"
    assert tools.run("get_quote", {"symbol": "AAPL", "extra": 1})[0]["kind"] == "bad_request"
    assert tools.run("get_history", {"symbol": "AAPL", "period": "10y"})[0]["kind"] == "bad_request"


def test_a_model_that_never_stops_asking_is_cut_off():
    response = ask(ScriptedLLM())
    assert response.metadata["llm_calls"] == MAX_STEPS and "couldn't finish" in response.text


def test_stats_come_from_python_and_say_when_data_is_short():
    full = make_tools(260).get_stats("AAPL")
    assert full["ma50"] is not None and full["ma200"] is not None and full["range_52w"]["covers_full_year"] is True
    short = make_tools(100).get_stats("AAPL")
    assert short["ma50"] is not None and short["ma200"] is None and short["price_vs_ma200_pct"] is None
    assert any("200-day average unavailable" in note for note in short["notes"])


def test_the_tool_trace_records_cache_hits():
    tools = make_tools()
    first = tools.run("get_quote", {"symbol": "AAPL"})[1]
    second = tools.run("get_quote", {"symbol": "AAPL"})[1]
    assert (first["cache_misses"], second["cache_hits"]) == (1, 1)
