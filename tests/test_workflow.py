import json
from pathlib import Path
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
from src.workflow.router import FINANCE_WORDS, DEFAULT_PROMPT_VERSION, DEFAULT_ROUTE, LABELS, PROMPTS, RESPONSE_FORMAT, ROUTES, ROUTES_V2, ROUTES_V5, Router, response_format_for

CONFIG = load_settings().workflow


class FakeLLM:
    """Scripted replies per purpose, chosen by the name of the JSON schema the caller asks for: route, standalone, news_topic."""
    def __init__(self, route=(), standalone=(), news_topic=(), plan=(), answer=()):
        self.queues = {"route": list(route), "standalone": list(standalone), "news_topic": list(news_topic), "plan": list(plan), "answer": list(answer)}
        self.calls = {"route": [], "standalone": [], "news_topic": [], "plan": [], "answer": []}

    def chat(self, prompt, system=None, **kwargs):
        purpose = (kwargs.get("response_format") or {}).get("json_schema", {}).get("name") or "answer"
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


def make(route=(), standalone=(), news_topic=(), saver=False, config=CONFIG, plan=(), answer=(), **overrides):
    llm = FakeLLM(route, standalone, news_topic, plan, answer)
    agents = Agents(FakeAgent("finance_qa"), FakeAgent("market_analysis"), FakeAgent("portfolio_analysis"), FakeGoal(), FakeAgent("news_synthesizer"), FakeAgent("tax_education"))
    for key, value in overrides.items():
        setattr(agents, key, value)
    graph = build_graph(llm, Router(llm), agents, config, checkpointer=InMemorySaver() if saver else None)
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
def test_the_router_uses_the_cheap_model_temperature_zero_a_strict_enum_schema_and_the_default_prompt():
    graph, llm, _ = make(route=[{"route": "finance_qa", "confidence": "high"}])
    ask(graph, "hello")
    call = llm.calls["route"][0]
    assert DEFAULT_PROMPT_VERSION == "v6" and call["model_key"] == "router_model" and call["temperature"] == 0 and call["system"] == PROMPTS[DEFAULT_PROMPT_VERSION]
    assert call["response_format"] == response_format_for(ROUTES) and call["response_format"]["json_schema"]["strict"] is True
    schema = call["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["route"]["enum"] == list(ROUTES) and schema["additionalProperties"] is False


def test_old_prompt_versions_cannot_produce_the_new_label_and_v3_can():
    assert "out_of_scope" not in LABELS["v1"] and "out_of_scope" not in LABELS["v2"] and "out_of_scope" in LABELS["v3"] and "multi" not in LABELS["v5"]
    assert RESPONSE_FORMAT == response_format_for(ROUTES_V2) and "out_of_scope" not in PROMPTS["v2"] and "out_of_scope" in PROMPTS["v3"]
    llm = FakeLLM(route=[{"route": "out_of_scope", "confidence": "high"}])
    assert Router(llm, prompt_version="v2").classify("pizza price").fell_back is True   # v2 may not say out_of_scope: treated as unusable, not obeyed


# ---------------- Step 21a: the scope gate ----------------
def scope_config(tmp_path):
    return CONFIG.model_copy(update={"scope_log_path": str(tmp_path / "scope.jsonl")})


def test_an_out_of_scope_message_gets_the_fixed_reply_and_touches_nothing_else(tmp_path):
    graph, llm, agents = make(route=[{"route": "out_of_scope", "confidence": "high"}], config=scope_config(tmp_path))
    state = ask(graph, "What's the price of a pizza?")
    assert state["route"] == "out_of_scope" and state["response"]["text"] == CONFIG.scope_reply and state["response"]["agent_name"] == "scope_reply"
    assert not any(called(agents).values())                                   # no agent, so no retrieval, no tool, no answering model
    assert len(llm.calls["route"]) == 1 and llm.calls["standalone"] == [] and llm.calls["news_topic"] == []   # one cheap classification, nothing more
    assert state["response"]["disclaimer"] == DISCLAIMER_STANDARD and [t["node"] for t in state["trace"]] == ["contextualize", "router", "scope_reply", "finalize"]


def test_every_out_of_scope_decision_is_logged_for_review(tmp_path):
    config = scope_config(tmp_path)
    graph, *_ = make(route=[{"route": "out_of_scope", "confidence": "high"}] * 2, config=config)
    ask(graph, "Write me a poem", session="a")
    ask(graph, "Who won the game?", session="b")
    entries = [json.loads(line) for line in open(config.scope_log_path)]
    assert [e["query"] for e in entries] == ["Write me a poem", "Who won the game?"] and entries[0]["session_id"] == "a" and entries[0]["confidence"] == "high"


def test_a_low_confidence_out_of_scope_becomes_clarify_so_a_real_question_is_never_blocked(tmp_path):
    config = scope_config(tmp_path)
    graph, _, agents = make(route=[{"route": "out_of_scope", "confidence": "low"}], config=config)
    state = ask(graph, "hmm what about that pizza company")
    assert state["route"] == "clarify" and state["router"]["raw_route"] == "out_of_scope" and state["router"]["overridden"] is True
    assert state["response"]["text"] == CONFIG.clarify_reply and not any(called(agents).values()) and not Path(config.scope_log_path).exists()   # asked, not blocked, not logged


def test_a_failing_log_does_not_stop_the_reply(tmp_path):
    bad = CONFIG.model_copy(update={"scope_log_path": str(tmp_path / "a_file_not_a_folder" / "x.jsonl")})
    (tmp_path / "a_file_not_a_folder").write_text("this is a file, so a log cannot be created inside it")
    graph, *_ = make(route=[{"route": "out_of_scope", "confidence": "high"}], config=bad)
    assert ask(graph, "Tell me a joke")["response"]["text"] == CONFIG.scope_reply


def test_the_v3_prompt_defines_the_scope_and_warns_against_blocking_real_finance_questions():
    prompt = PROMPTS["v3"]
    assert "NOT about money" in prompt and "Blocking a real finance question is worse" in prompt
    assert prompt.count("-> ") >= 8 + 4   # at least 8 new few-shot lines on top of the v2 examples


def test_the_graph_draws_itself_with_every_node():
    graph, *_ = make()
    mermaid = graph.get_graph().draw_mermaid()
    assert all(name in mermaid for name in ("contextualize", "router", "finance_qa", "market", "portfolio", "goal", "news", "tax", "clarify", "scope_reply", "fallback", "finalize"))


def test_v4_adds_the_rules_that_the_stress_sets_called_for():
    assert "whatever the everyday topic" in PROMPTS["v4"] and "SURE" in PROMPTS["v4"] and PROMPTS["v4"].startswith(PROMPTS["v3"])
    assert LABELS["v4"] == ROUTES_V5


def test_the_finance_word_veto_is_off_by_default_and_turns_a_block_into_a_question_when_on():
    reply = {"route": "out_of_scope", "confidence": "high"}
    off = Router(FakeLLM(route=[reply])).classify("Is a pizza franchise a good investment?")
    assert off.route == "out_of_scope" and off.vetoed is False                    # default: the model's decision stands
    on = Router(FakeLLM(route=[reply]), finance_veto=True).classify("Is a pizza franchise a good investment?")
    assert on.route == "clarify" and on.vetoed is True and on.raw_route == "out_of_scope"
    clean = Router(FakeLLM(route=[reply]), finance_veto=True).classify("Write me a haiku about autumn")
    assert clean.route == "out_of_scope" and clean.vetoed is False               # no money word: still blocked
    assert FINANCE_WORDS.search("Should I pay off my loan?") and not FINANCE_WORDS.search("What time does the sun set in London?")


def test_v5_tells_the_router_that_uploaded_files_are_portfolio_requests():
    assert "NEVER out_of_scope" in PROMPTS["v5"] and PROMPTS["v5"].startswith(PROMPTS["v4"]) and LABELS["v5"] == ROUTES_V5


def test_a_user_who_uploaded_a_portfolio_is_never_blocked_even_if_the_router_says_out_of_scope(tmp_path):
    config = scope_config(tmp_path)
    graph, _, agents = make(route=[{"route": "out_of_scope", "confidence": "high"}], config=config)
    state = ask(graph, "Review the CSV I uploaded", portfolio_csv="ticker,shares,avg_cost\nAAA,1,1")
    assert state["route"] == "clarify" and state["router"]["guard"] == "has_uploaded_portfolio" and state["response"]["text"] == CONFIG.clarify_reply
    assert not Path(config.scope_log_path).exists() and not any(called(agents).values())   # asked, not blocked, not logged


def test_without_an_upload_the_out_of_scope_decision_stands(tmp_path):
    graph, *_ = make(route=[{"route": "out_of_scope", "confidence": "high"}], config=scope_config(tmp_path))
    assert ask(graph, "Tell me a joke")["route"] == "out_of_scope"


def test_v6_adds_the_multi_label_and_leans_toward_it_when_two_sources_are_asked_for():
    assert "multi" in LABELS["v6"] and "multi" not in LABELS["v5"] and "multi" not in PROMPTS["v5"]
    assert PROMPTS["v6"].startswith(PROMPTS["v5"].split("\n\nConfidence:")[0]) and "TWO OR MORE separate things" in PROMPTS["v6"] and "choose multi" in PROMPTS["v6"]
