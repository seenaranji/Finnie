import json

import pytest

from src.core.llm import LLMResult
from src.core.schemas import AgentRequest, AgentResponse
from src.workflow.graph import build_graph
from src.workflow.router import DEFAULT_ROUTE, RESPONSE_FORMAT, ROUTES, Router


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, prompt, system=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        reply = self.replies.pop(0)
        return LLMResult(reply if isinstance(reply, str) else json.dumps(reply), "fake-router", 300, 6, 0.1, 0.00005, 1)


class FakeAgent:
    def __init__(self, name):
        self.name, self.requests = name, []

    def handle(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(text=f"{self.name} answer", agent_name=self.name, metadata={"model": "fake", "cost_usd": 0.001, "tool_names": []})


def run(*router_replies, message="hello"):
    llm, qa, market = FakeLLM(*router_replies), FakeAgent("finance_qa"), FakeAgent("market_analysis")
    graph = build_graph(Router(llm), qa, market)
    state = graph.invoke({"message": message, "session_id": "s1", "trace": []})
    return state, llm, qa, market


def test_a_market_route_runs_only_the_market_node():
    state, _, qa, market = run({"route": "market"}, message="how is VOO doing")
    assert state["route"] == "market" and state["response"]["agent_name"] == "market_analysis"
    assert len(market.requests) == 1 and qa.requests == []                      # the conditional edge skipped the other node entirely
    assert market.requests[0].message == "how is VOO doing" and market.requests[0].session_id == "s1"


def test_a_finance_route_runs_only_the_qa_node():
    state, _, qa, market = run({"route": "finance_qa"})
    assert state["response"]["text"] == "finance_qa answer" and len(qa.requests) == 1 and market.requests == []


def test_the_trace_reducer_appends_each_nodes_entry_instead_of_overwriting():
    state, *_ = run({"route": "market"})
    assert [t["node"] for t in state["trace"]] == ["router", "market"]           # without the reducer only the last entry would survive
    assert state["trace"][0]["route"] == "market" and state["trace"][0]["tokens"] == 306 and state["trace"][1]["agent"] == "market_analysis"


def test_the_final_state_is_plain_data_a_checkpointer_could_save():
    state, *_ = run({"route": "finance_qa"})
    json.dumps(state)   # raises if anything in the state is not plain JSON data


@pytest.mark.parametrize("bad_reply", ["not json at all", {"wrong_key": 1}, {"route": "weather"}, {"route": None}])
def test_an_unusable_router_reply_falls_back_to_the_safe_default_and_says_so(bad_reply):
    state, *_ = run(bad_reply)
    assert state["route"] == DEFAULT_ROUTE and state["router"]["fell_back"] is True and state["trace"][0]["fell_back"] is True


def test_the_router_uses_the_cheap_model_temperature_zero_and_a_strict_enum_schema():
    _, llm, *_ = run({"route": "finance_qa"})
    call = llm.calls[0]
    assert call["model_key"] == "router_model" and call["temperature"] == 0 and call["max_output_tokens"] == 20
    assert call["response_format"] is RESPONSE_FORMAT and call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["schema"]["properties"]["route"]["enum"] == list(ROUTES)


def test_the_graph_draws_itself_with_the_router_and_both_branches():
    graph = build_graph(Router(FakeLLM()), FakeAgent("a"), FakeAgent("b"))
    mermaid = graph.get_graph().draw_mermaid()
    assert all(name in mermaid for name in ("router", "finance_qa", "market", "__start__", "__end__"))
