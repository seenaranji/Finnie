import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.goal_extraction import GoalState
from src.core.config import load_settings
from src.core.guardrails import DISCLAIMER_STANDARD, DISCLAIMER_STRONG
from src.core.llm import LLMResult
from src.core.schemas import AgentRequest, AgentResponse
from src.data.errors import ProviderUnavailable
from src.data.news import NewsUnavailable
from src.workflow.graph import Agents, build_graph
from src.workflow.nodes import extract_news_topic, rewrite_message
from src.workflow.router import DEFAULT_ROUTE, PROMPTS, RESPONSE_FORMAT, ROUTES, Router

CONFIG = load_settings().workflow


class FakeLLM:
    """Scripted replies per purpose, chosen by the name of the JSON schema the caller asks for: route, standalone, news_topic."""
    def __init__(self, route=(), standalone=(), news_topic=()):
        self.queues = {"route": list(route), "standalone": list(standalone), "news_topic": list(news_topic)}
        self.calls = {"route": [], "standalone": [], "news_topic": []}

    def chat(self, prompt, system=None, **kwargs):
        purpose = kwargs["response_format"]["json_schema"]["name"]
        self.calls[purpose].append({"prompt": prompt, "system": system, **kwargs})
        reply = self.queues[purpose].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(reply if isinstance(reply, str) else json.dumps(reply), "fake", 300, 6, 0.1, 0.00005, 1)


class FakeAgent:
    def __init__(self, name, raises=None):
        self.name, self.raises, self.requests = name, raises, []

    def _answer(self, text=None):
        if self.raises:
            raise self.raises
        return AgentResponse(text=text or f"{self.name} answer", agent_name=self.name, disclaimer="", metadata={"model": "fake", "cost_usd": 0.001, "tool_names": []})

    def handle(self, request: AgentRequest):
        self.requests.append(request)
        return self._answer()

    def run(self, *args):
        self.requests.append(args)
        return self._answer()


class FakeGoal(FakeAgent):
    def __init__(self):
        super().__init__("goal_planning")
        self.script = []   # each item: (state to store, plan_made)

    def run(self, message, state):
        self.requests.append((message, state.model_copy()))
        new_state, plan_made = self.script.pop(0)
        return SimpleNamespace(response=self._answer(), state=new_state, plan_made=plan_made)


def make(route=(), standalone=(), news_topic=(), saver=False, **overrides):
    llm = FakeLLM(route, standalone, news_topic)
    agents = Agents(FakeAgent("finance_qa"), FakeAgent("market_analysis"), FakeAgent("portfolio_analysis"), FakeGoal(), FakeAgent("news_synthesizer"), FakeAgent("tax_education"))
    for key, value in overrides.items():
        setattr(agents, key, value)
    graph = build_graph(llm, Router(llm), agents, CONFIG, checkpointer=InMemorySaver() if saver else None)
    return graph, llm, agents


def ask(graph, message, session="s1", **extra):
    return graph.invoke({"message": message, "session_id": session, **extra}, {"configurable": {"thread_id": session}} if graph.checkpointer else None)


def called(agents):
    return {name: bool(getattr(agents, name).requests) for name in ("finance_qa", "market", "portfolio", "goal", "news", "tax")}


@pytest.mark.parametrize("route,agent_attr,expected_name", [("finance_qa", "finance_qa", "finance_qa"), ("market", "market", "market_analysis"), ("tax", "tax", "tax_education"),
                                                            ("portfolio", "portfolio", "portfolio_analysis"), ("news", "news", "news_synthesizer")])
def test_each_route_runs_only_its_own_agent(route, agent_attr, expected_name):
    graph, _, agents = make(route=[{"route": route, "confidence": "high"}], news_topic=[{"topic": "AAPL"}])
    state = ask(graph, "hello", portfolio_csv="ticker,shares,avg_cost\nAAA,1,1")
    assert state["route"] == route and state["response"]["agent_name"] == expected_name
    assert called(agents) == {name: name == agent_attr for name in called(agents)}   # the conditional edge skipped every other node


def test_the_goal_route_runs_only_the_goal_agent():
    graph, _, agents = make(route=[{"route": "goal", "confidence": "high"}])
    agents.goal.script = [(GoalState(goal_amount_usd=1000), False)]
    state = ask(graph, "save 1000")
    assert state["response"]["agent_name"] == "goal_planning" and called(agents)["goal"] and not any(v for k, v in called(agents).items() if k != "goal")


def test_clarify_is_a_fixed_reply_with_no_agent_and_no_extra_model_call():
    graph, llm, agents = make(route=[{"route": "clarify", "confidence": "high"}])
    state = ask(graph, "what about that one?")
    assert state["response"]["text"] == CONFIG.clarify_reply and not any(called(agents).values())
    assert state["response"]["disclaimer"] == DISCLAIMER_STANDARD and len(llm.calls["route"]) == 1 and llm.calls["standalone"] == []


def test_a_low_confidence_route_becomes_clarify_and_the_trace_says_so():
    graph, *_ = make(route=[{"route": "market", "confidence": "low"}])
    state = ask(graph, "hmm")
    assert state["route"] == "clarify" and state["router"]["raw_route"] == "market" and state["router"]["overridden"] is True
    assert [t for t in state["trace"] if t["node"] == "router"][0]["overridden"] is True


@pytest.mark.parametrize("bad", ["not json", {"route": "weather", "confidence": "high"}, {"route": "market"}, {"route": "market", "confidence": "certain"}])
def test_an_unusable_router_reply_falls_back_to_clarify_and_says_so(bad):
    graph, *_ = make(route=[bad])
    state = ask(graph, "hello")
    assert state["route"] == DEFAULT_ROUTE and state["router"]["fell_back"] is True and state["response"]["text"] == CONFIG.clarify_reply


def test_a_router_failure_gives_the_fixed_fallback_reply_and_does_not_raise():
    graph, *_ = make(route=[ConnectionError("network down")])
    state = ask(graph, "hello")
    assert state["route"] == "fallback" and state["response"]["text"] == CONFIG.fallback_reply and state["error"]["node"] == "router"


def test_an_agent_failure_gives_a_safe_reply_and_records_the_error_without_leaking_the_exception():
    graph, _, agents = make(route=[{"route": "finance_qa", "confidence": "high"}], finance_qa=FakeAgent("finance_qa", raises=RuntimeError("boom: secret internals")))
    state = ask(graph, "what is an ETF")
    assert state["response"]["text"] == CONFIG.fallback_reply and "secret" not in state["response"]["text"]
    assert state["error"] == {"node": "finance_qa", "type": "RuntimeError", "message": "boom: secret internals"} and state["response"]["disclaimer"] == DISCLAIMER_STANDARD


def test_our_own_errors_show_their_user_safe_message():
    graph, *_ = make(route=[{"route": "market", "confidence": "high"}], market=FakeAgent("m", raises=ProviderUnavailable("The market data service is not responding right now.")))
    assert ask(graph, "price of VOO")["response"]["text"].startswith("The market data service is not responding right now.")
    graph, *_ = make(route=[{"route": "news", "confidence": "high"}], news_topic=[{"topic": "AAPL"}], news=FakeAgent("n", raises=NewsUnavailable("News is not available right now.")))
    assert "News is not available right now." in ask(graph, "news on Apple")["response"]["text"]


def test_portfolio_asks_for_the_holdings_when_there_are_none_and_uses_them_when_there_are():
    graph, _, agents = make(route=[{"route": "portfolio", "confidence": "high"}])
    state = ask(graph, "analyze my portfolio")
    assert state["response"]["text"] == CONFIG.needs_portfolio_reply and agents.portfolio.requests == []
    graph, _, agents = make(route=[{"route": "portfolio", "confidence": "high"}])
    ask(graph, "analyze my portfolio", portfolio_csv="ticker,shares,avg_cost\nAAA,1,1")
    assert agents.portfolio.requests == [("ticker,shares,avg_cost\nAAA,1,1", "analyze my portfolio")]


def test_news_topic_extraction_falls_back_to_markets_for_anything_unusable():
    llm = FakeLLM(news_topic=[{"topic": "aapl"}, {"topic": "markets"}, {"topic": "not a ticker!"}, "garbage"])
    assert [extract_news_topic(llm, "x")[0] for _ in range(4)] == ["AAPL", "markets", "markets", "markets"]


@pytest.mark.parametrize("route,strong", [("tax", True), ("portfolio", True), ("goal", True), ("finance_qa", False), ("market", False), ("news", False), ("clarify", False)])
def test_the_final_node_sets_the_disclaimer_by_route(route, strong):
    graph, _, agents = make(route=[{"route": route, "confidence": "high"}], news_topic=[{"topic": "AAPL"}])
    agents.goal.script = [(GoalState(), False)]
    state = ask(graph, "hello", portfolio_csv="ticker,shares,avg_cost\nAAA,1,1")
    assert state["response"]["disclaimer"] == (DISCLAIMER_STRONG if strong else DISCLAIMER_STANDARD)


# ---------------- memory ----------------
def test_the_second_turn_is_rewritten_with_the_saved_history_and_the_first_turn_costs_no_rewrite_call():
    graph, llm, agents = make(route=[{"route": "finance_qa", "confidence": "high"}, {"route": "market", "confidence": "high"}], standalone=[{"standalone": "How is VOO doing today?"}], saver=True)
    ask(graph, "What is an ETF?")
    assert llm.calls["standalone"] == []                                  # nothing to resolve yet
    state = ask(graph, "How is it doing today?")
    assert state["turn"] == 2 and state["resolved_message"] == "How is VOO doing today?" and state["message"] == "How is it doing today?"
    assert agents.market.requests[0].message == "How is VOO doing today?"   # the agent sees the resolved question
    assert "What is an ETF?" in llm.calls["standalone"][0]["prompt"] and [h["role"] for h in state["history"]] == ["user", "assistant", "user", "assistant"]


def test_different_session_ids_do_not_share_memory():
    graph, llm, _ = make(route=[{"route": "finance_qa", "confidence": "high"}] * 3, standalone=[{"standalone": "x"}], saver=True)
    ask(graph, "What is an ETF?", session="a")
    ask(graph, "What is a bond?", session="b")
    assert llm.calls["standalone"] == []                                  # session b's first turn had no history
    ask(graph, "and that?", session="a")
    assert len(llm.calls["standalone"]) == 1 and "ETF" in llm.calls["standalone"][0]["prompt"] and "bond" not in llm.calls["standalone"][0]["prompt"]


def test_a_bad_rewrite_falls_back_to_the_original_message():
    for reply in ["not json", {"standalone": ""}, {"standalone": "x" * 500}, ConnectionError("down")]:
        llm = FakeLLM(standalone=[reply])
        assert rewrite_message(llm, "and that?", [{"role": "user", "content": "What is an ETF?"}], 3)[0] == "and that?"


def test_goal_slots_persist_between_turns_and_are_cleared_after_a_plan():
    graph, _, agents = make(route=[{"route": "goal", "confidence": "high"}] * 3, standalone=[{"standalone": "a"}, {"standalone": "b"}], saver=True)
    agents.goal.script = [(GoalState(goal_amount_usd=50000), False), (GoalState(goal_amount_usd=50000, years=5), True), (GoalState(goal_amount_usd=1000), False)]
    ask(graph, "I want 50k")
    ask(graph, "in 5 years")
    state = ask(graph, "a different goal: 1000")
    seen = [r[1].model_dump() for r in agents.goal.requests]
    assert seen[0]["goal_amount_usd"] is None and seen[1]["goal_amount_usd"] == 50000      # turn 2 received turn 1's slots
    assert seen[2]["goal_amount_usd"] is None                                              # turn 3 started fresh because the plan was finished
    assert state["goal_state"]["goal_amount_usd"] == 1000


def test_the_portfolio_is_remembered_after_it_is_supplied_once():
    graph, _, agents = make(route=[{"route": "portfolio", "confidence": "high"}] * 2, standalone=[{"standalone": "explain it again"}], saver=True)
    ask(graph, "analyze my portfolio", portfolio_csv="ticker,shares,avg_cost\nAAA,1,1")
    ask(graph, "explain it again")
    assert [r[0] for r in agents.portfolio.requests] == ["ticker,shares,avg_cost\nAAA,1,1"] * 2


def test_the_trace_grows_across_turns_and_is_tagged_with_the_turn_and_the_state_is_plain_data():
    graph, *_ = make(route=[{"route": "finance_qa", "confidence": "high"}] * 2, standalone=[{"standalone": "x"}], saver=True)
    ask(graph, "What is an ETF?")
    state = ask(graph, "and a bond?")
    assert [(t["turn"], t["node"]) for t in state["trace"] if t["node"] in ("contextualize", "router")] == [(1, "contextualize"), (1, "router"), (2, "contextualize"), (2, "router")]
    json.dumps(state)   # raises if anything in the state is not plain data


# ---------------- router ----------------
def test_the_router_uses_the_cheap_model_temperature_zero_a_strict_enum_schema_and_prompt_v2():
    graph, llm, _ = make(route=[{"route": "finance_qa", "confidence": "high"}])
    ask(graph, "hello")
    call = llm.calls["route"][0]
    assert call["model_key"] == "router_model" and call["temperature"] == 0 and call["response_format"] is RESPONSE_FORMAT and call["system"] == PROMPTS["v2"]
    schema = call["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["route"]["enum"] == list(ROUTES) and schema["additionalProperties"] is False and call["response_format"]["json_schema"]["strict"] is True


def test_the_graph_draws_itself_with_every_node():
    graph, *_ = make()
    mermaid = graph.get_graph().draw_mermaid()
    assert all(name in mermaid for name in ("contextualize", "router", "finance_qa", "market", "portfolio", "goal", "news", "tax", "clarify", "fallback", "finalize"))
