import json
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.core.guardrails import DISCLAIMER_STANDARD, DISCLAIMER_STRONG
from src.core.llm import LLMResult
from src.core.schemas import AgentRequest, AgentResponse, Source
from src.workflow.graph import Agents, build_graph, run_with_timeout
from src.workflow.planner import PLAN_FORMAT, AGENTS, PlanError, Planner, levels, validate_plan
from src.workflow.router import Router
from src.workflow.state import merge_results
from src.workflow.synthesizer import Synthesizer, concat, missing_note
from src.workflow.timeline import render_timeline, timing_summary
from tests.test_workflow import CONFIG, FakeLLM

MULTI = {"route": "multi", "confidence": "high"}


def task(id, agent, question="q", deps=()):
    return {"id": id, "agent": agent, "question": question or "q", "depends_on": list(deps)}


def plan_reply(*tasks):
    return {"tasks": list(tasks)}


class SlowAgent:
    """A fake agent that takes `delay` seconds, can fail, and remembers what it was asked."""
    def __init__(self, name, delay=0.0, text=None, raises=None, source=None, metadata=None):
        self.name, self.delay, self.text, self.raises, self.source, self.questions, self.metadata = name, delay, text, raises, source, [], metadata or {}

    def _go(self, question):
        self.questions.append(question)
        time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return AgentResponse(text=self.text or f"{self.name} says 42 [{self.name}#0]", agent_name=self.name, disclaimer="", sources=[Source(id=self.source, title="s")] if self.source else [],
                             metadata={"model": "fake", "cost_usd": 0.001, "tool_names": [], **self.metadata})

    def handle(self, request: AgentRequest):
        return self._go(request.message)

    def run(self, *args):
        return self._go(args[-1])


def build(plan, delays=None, route=MULTI, answer=("merged answer",), config=CONFIG, saver=False, **agent_overrides):
    delays = delays or {}
    llm = FakeLLM(route=[route] * 4, plan=[plan] * 4, answer=list(answer) * 4, news_topic=[{"topic": "AAPL"}] * 4)
    names = {"finance_qa": "finance_qa", "market": "market", "portfolio": "portfolio", "news": "news", "tax": "tax"}
    made = {k: SlowAgent(v, delays.get(k, 0.0), source=f"src-{k}") for k, v in names.items()}
    made.update(agent_overrides)
    goal = SlowAgent("goal", delays.get("goal", 0.0))
    goal.run = lambda message, state: type("T", (), {"response": goal._go(message), "state": state, "plan_made": True})()
    agents = Agents(made["finance_qa"], made["market"], made["portfolio"], goal, made["news"], made["tax"])
    return build_graph(llm, Router(llm), agents, config, checkpointer=InMemorySaver() if saver else None), llm, agents


def ask(graph, message="a multi question", session="m1", **extra):
    return graph.invoke({"message": message, "session_id": session, "portfolio_csv": "ticker,shares,avg_cost\nAAA,1,1", **extra}, {"configurable": {"thread_id": session}})


# ---------------- the plan ----------------
@pytest.mark.parametrize("tasks,reason", [
    ([], "1 to 4"), ([task(f"t{i}", "market") for i in range(1, 6)], "1 to 4"), ([task("t1", "market"), task("t1", "news")], "unique"),
    ([task("t1", "weather")], "unknown agent"), ([task("t1", "market", deps=["t9"])], "not another task"), ([task("t1", "market", deps=["t1"])], "not another task"),
    ([task("t1", "market", deps=["t2"]), task("t2", "news", deps=["t1"])], "circle"), ([task("t1", "market", question="  ")], "no question")])
def test_bad_plans_are_rejected_in_code(tasks, reason):
    with pytest.raises(PlanError, match=reason):
        validate_plan(tasks, 4)


def test_a_valid_plan_is_grouped_into_waves():
    tasks = validate_plan([task("t1", "portfolio"), task("t2", "market"), task("t3", "news", deps=["t1"]), task("t4", "tax", deps=["t3", "t2"])], 4)
    assert [[t.id for t in wave] for wave in levels(tasks)] == [["t1", "t2"], ["t3"], ["t4"]]


def test_the_planner_uses_a_strict_schema_the_cheap_model_and_reports_unusable_output():
    llm = FakeLLM(plan=[plan_reply(task("t1", "market")), "not json", {"tasks": [{"id": "t1"}]}])
    plan = Planner(llm).plan("x")
    call = llm.calls["plan"][0]
    assert [t.agent for t in plan.tasks] == ["market"] and call["model_key"] == "router_model" and call["temperature"] == 0 and call["response_format"] is PLAN_FORMAT
    assert PLAN_FORMAT["json_schema"]["strict"] is True and PLAN_FORMAT["json_schema"]["schema"]["properties"]["tasks"]["items"]["properties"]["agent"]["enum"] == list(AGENTS)
    for _ in range(2):
        with pytest.raises(PlanError):
            Planner(llm).plan("x")


# ---------------- parallel vs sequential ----------------
THREE = plan_reply(task("t1", "market", "q1"), task("t2", "news", "q2"), task("t3", "finance_qa", "q3"))


def test_independent_tasks_overlap_in_time_when_run_in_parallel():
    graph, *_ = build(THREE, delays={"market": 0.4, "news": 0.4, "finance_qa": 0.4})
    start = time.perf_counter()
    state = ask(graph)
    wall = time.perf_counter() - start
    results = state["sub_results"]
    assert wall < 0.9 and set(results) == {"t1", "t2", "t3"}                               # three 0.4 s branches took far less than the 1.2 s they add up to
    assert max(r["started"] for r in results.values()) < min(r["ended"] for r in results.values())   # every branch started before any branch had finished
    assert timing_summary(results)["overlap_factor"] > 2.0


def test_sequential_mode_runs_one_task_at_a_time_through_the_same_graph():
    graph, *_ = build(THREE, delays={"market": 0.3, "news": 0.3, "finance_qa": 0.3})
    start = time.perf_counter()
    state = ask(graph, execution="sequential")
    wall = time.perf_counter() - start
    rows = sorted(state["sub_results"].values(), key=lambda r: r["started"])
    assert wall >= 0.9 and all(rows[i + 1]["started"] >= rows[i]["ended"] for i in range(2))   # each started only after the previous one ended
    assert timing_summary(state["sub_results"])["overlap_factor"] < 1.05


def test_a_dependent_task_waits_for_its_dependency_and_receives_its_result():
    plan = plan_reply(task("t1", "portfolio", "which is my smallest holding?"), task("t2", "market", "how is the smallest holding doing?", deps=["t1"]))
    graph, _, agents = build(plan, delays={"portfolio": 0.2, "market": 0.1})
    r = ask(graph)["sub_results"]
    assert r["t2"]["started"] >= r["t1"]["ended"]
    assert "RESULT OF AN EARLIER STEP (portfolio)" in agents.market.questions[0] and "portfolio says 42" in agents.market.questions[0]


def test_for_the_news_agent_the_earlier_result_reaches_the_topic_extractor_which_picks_the_ticker():
    plan = plan_reply(task("t1", "portfolio", "which is my smallest holding?"), task("t2", "news", "news on the smallest holding", deps=["t1"]))
    graph, llm, agents = build(plan)
    ask(graph)
    assert "RESULT OF AN EARLIER STEP (portfolio)" in llm.calls["news_topic"][0]["prompt"] and agents.news.questions == ["AAPL"]   # the agent itself only gets the topic


def test_the_trace_records_the_planner_every_task_and_the_synthesis():
    graph, *_ = build(THREE)
    nodes = [t["node"] for t in ask(graph)["trace"]]
    assert nodes[:3] == ["contextualize", "router", "planner"] and {"task:t1", "task:t2", "task:t3"} <= set(nodes) and nodes[-2:] == ["synthesize", "finalize"]


# ---------------- partial results ----------------
def test_one_failed_branch_still_gives_the_rest_and_says_what_is_missing():
    graph, *_ = build(THREE, market=SlowAgent("market", raises=RuntimeError("boom: secret internals")))
    state = ask(graph)
    text = state["response"]["text"]
    assert state["response"]["metadata"]["partial"] and state["response"]["metadata"]["missing"] == ["market"] and state["sub_results"]["t1"]["status"] == "error"
    assert "news says 42" in text or "merged answer" in text                      # the other branches are still in the answer
    assert "Market data: that part isn't available right now" in text and "secret" not in text and "RuntimeError" not in text


def test_a_slow_branch_times_out_and_is_reported_without_holding_up_the_answer():
    config = CONFIG.model_copy(update={"branch_timeout_seconds": 0.2})
    graph, *_ = build(THREE, delays={"news": 1.5}, config=config)
    start = time.perf_counter()
    state = ask(graph)
    assert time.perf_counter() - start < 1.2 and state["sub_results"]["t2"]["status"] == "timeout"
    assert "News: that part took too long" in state["response"]["text"]


def test_a_task_whose_dependency_failed_is_skipped_not_attempted():
    plan = plan_reply(task("t1", "portfolio"), task("t2", "news", deps=["t1"]), task("t3", "market"))
    graph, _, agents = build(plan, portfolio=SlowAgent("portfolio", raises=RuntimeError("x")))
    state = ask(graph)
    assert state["sub_results"]["t2"]["status"] == "skipped" and agents.news.questions == []
    assert "News: it depended on another part that isn't available" in state["response"]["text"]


def test_when_every_branch_fails_the_user_gets_a_clear_message_not_an_empty_answer():
    boom = {k: SlowAgent(k, raises=RuntimeError("x")) for k in ("market", "news", "finance_qa")}
    graph, *_ = build(THREE, **boom)
    text = ask(graph)["response"]["text"]
    assert text.startswith("I couldn't get any part of that right now:") and "Market data" in text and "News" in text and text.count("couldn't") == 1


def test_a_single_task_plan_returns_that_agents_answer_unchanged_with_no_synthesis_call():
    graph, llm, _ = build(plan_reply(task("t1", "tax", "how are dividends taxed?")))
    state = ask(graph)
    assert state["response"]["text"] == "tax says 42 [tax#0]" and llm.calls["answer"] == [] and state["response"]["metadata"]["plan"][0]["agent"] == "tax"


def test_an_unusable_plan_gives_the_fixed_fallback_reply_and_records_the_error():
    graph, *_ = build("not json")
    state = ask(graph)
    assert state["response"]["text"] == CONFIG.fallback_reply and state["error"]["node"] == "planner"


def test_the_results_of_one_turn_do_not_leak_into_the_next():
    graph, _, agents = build(THREE, saver=True)
    ask(graph, "first")
    ask(graph, "second")
    assert len(agents.market.questions) == 2 and len(agents.news.questions) == 2          # turn 2 ran its tasks again; turn 1's finished t1/t2/t3 were not reused


# ---------------- one answer, one disclaimer ----------------
def test_sources_are_merged_without_duplicates_and_there_is_one_disclaimer():
    graph, *_ = build(plan_reply(task("t1", "market"), task("t2", "market")), answer=["ok"])
    r = ask(graph)["response"]
    assert [s["id"] for s in r["sources"]] == ["src-market"] and r["disclaimer"] == DISCLAIMER_STANDARD


def test_the_disclaimer_is_strong_when_the_plan_includes_tax_portfolio_or_goal():
    for agent in ("tax", "portfolio", "goal"):
        graph, *_ = build(plan_reply(task("t1", "market"), task("t2", agent)))
        assert ask(graph)["response"]["disclaimer"] == DISCLAIMER_STRONG


def test_the_cost_is_the_sum_of_planner_branches_and_synthesis():
    graph, *_ = build(THREE)
    meta = ask(graph)["response"]["metadata"]
    assert meta["cost_usd"] == pytest.approx(meta["planner_cost_usd"] + meta["branch_cost_usd"] + meta["synthesis_cost_usd"]) and meta["branch_cost_usd"] == pytest.approx(0.003)


# ---------------- pieces ----------------
def test_the_reducer_merges_branch_results_and_a_reset_clears_them():
    assert merge_results({"t1": 1}, {"t2": 2}) == {"t1": 1, "t2": 2} and merge_results(None, {"t1": 1}) == {"t1": 1}
    assert merge_results({"t1": 1}, {"__reset__": True}) == {} and merge_results({"t1": 1}, {"__reset__": True, "t9": 9}) == {"t9": 9}


SECTIONS = [{"agent": "market", "text": "VOO closed at $694.81 [src-market#0]."}, {"agent": "news", "text": "Stocks fell 1.77% on Friday [2]."}]


def test_a_clean_merge_is_accepted_and_one_with_invented_numbers_markers_or_links_falls_back_to_the_code_merge():
    ok = Synthesizer(FakeLLM(answer=["VOO closed at $694.81 [src-market#0]. Stocks fell 1.77% [2]."])).merge("q", SECTIONS)
    assert ok.mode == "llm" and ok.issues == []
    for bad, expect in [("VOO closed at $700.00 [src-market#0].", "numbers"), ("VOO closed at $694.81 [src-market#9].", "citation"), ("See https://x.example for $694.81", "link")]:
        merged = Synthesizer(FakeLLM(answer=[bad])).merge("q", SECTIONS)
        assert merged.mode == "concat" and merged.text == concat(SECTIONS) and expect in " ".join(merged.issues)


def test_the_missing_note_names_the_part_and_a_reason_without_internals():
    note = missing_note([{"agent": "news", "status": "timeout"}, {"agent": "market", "status": "error"}])
    assert "News: that part took too long" in note and "Market data: that part isn't available right now" in note and missing_note([]) == ""


def test_the_timeline_draws_a_bar_per_branch():
    results = {"t1": {"id": "t1", "agent": "market", "status": "ok", "started": 10.0, "ended": 12.0}, "t2": {"id": "t2", "agent": "news", "status": "ok", "started": 10.0, "ended": 11.0}}
    text = render_timeline(results)
    assert "t1" in text and "t2" in text and "overlap factor 1.50x" in text and text.count("#") > 10


def test_run_with_timeout_distinguishes_a_slow_call_from_a_failing_one():
    assert run_with_timeout(lambda: 7, 1) == ("ok", 7)
    assert run_with_timeout(lambda: time.sleep(0.5), 0.05) == ("timeout", None)
    status, error = run_with_timeout(lambda: (_ for _ in ()).throw(TimeoutError("the agent itself raised this")), 1)
    assert status == "error" and isinstance(error, TimeoutError)


def test_a_branch_that_returns_normally_but_could_not_do_its_job_is_degraded_left_out_of_the_merge_and_listed_as_missing():
    apology = SlowAgent("market", text="There was an error retrieving the data for the index.", metadata={"tool_trace": [{"result": {"error": "no such ticker"}}]})
    graph, llm, _ = build(THREE, market=apology, answer=["News says 42 [src-news#0]. finance_qa says 42 [src-finance_qa#0]."])
    state = ask(graph)
    assert state["sub_results"]["t1"]["status"] == "degraded" and state["response"]["metadata"]["partial"] and state["response"]["metadata"]["missing"] == ["market"]
    assert "There was an error retrieving" not in llm.calls["answer"][0]["prompt"]                      # the apology was NOT handed to the synthesizer as content
    assert "PARTS THAT COULD NOT BE ANSWERED (say nothing about them): Market data" in llm.calls["answer"][0]["prompt"]
    assert "Market data: that part isn't available right now" in state["response"]["text"]


def test_with_one_healthy_branch_the_answer_is_that_branch_plus_the_missing_note_and_no_merge_call():
    apology = SlowAgent("market", text="error", metadata={"tool_trace": [{"result": {"error": "x"}}]})
    graph, llm, _ = build(plan_reply(task("t1", "market"), task("t2", "news")), market=apology)
    text = ask(graph)["response"]["text"]
    assert llm.calls["answer"] == [] and text.startswith("news says 42 [news#0]") and "Market data: that part isn't available right now" in text


@pytest.mark.parametrize("model_note,reason", [("none (no context)", "I don't have information on that in my knowledge base"), ("none (needs the portfolio)", "I need your portfolio holdings for that"),
                                                ("none (no usable news)", "no recent news was found")])
def test_each_kind_of_soft_failure_gets_its_own_user_safe_reason(model_note, reason):
    soft = SlowAgent("finance_qa", text="I cannot help with that.", metadata={"model": model_note})
    graph, *_ = build(plan_reply(task("t1", "finance_qa"), task("t2", "market")), finance_qa=soft)
    assert f"Concept: {reason}" in ask(graph)["response"]["text"]


def test_list_numerals_in_a_merged_answer_are_not_mistaken_for_invented_numbers():
    llm = FakeLLM(answer=["1. VOO closed at $694.81 [src-market#0].\n2. Stocks fell 1.77% [2]."])
    merged = Synthesizer(llm).merge("q", SECTIONS)
    assert merged.mode == "llm" and merged.issues == []


def test_index_symbols_with_a_caret_are_valid_tickers():
    from src.core.config import load_settings
    from src.data.cache import TTLCache
    from src.data.market_data import MarketDataService
    service = MarketDataService(provider=None, cache=TTLCache(), settings=load_settings())
    assert service._clean("^gspc") == "^GSPC" and service._clean("brk-b") == "BRK-B"
    with pytest.raises(Exception):
        service._clean("^^GSPC")


def test_a_failing_merge_model_falls_back_to_the_code_merge_and_never_raises():
    class Down:
        def chat(self, *a, **k):
            raise ConnectionError("down")
    merged = Synthesizer(Down()).merge("q", SECTIONS)
    assert merged.mode == "concat" and merged.text == concat(SECTIONS) and "ConnectionError" in merged.issues[0]
    # and through the whole graph: the answer still arrives
    graph, llm, _ = build(THREE)
    llm.queues["answer"] = []   # the fake raises IndexError when asked for a reply it does not have, like a failing model
    state = ask(graph)
    assert state["response"]["metadata"]["synthesis"] == "concat" and "market says 42" in state["response"]["text"]


def test_the_merge_model_is_told_which_parts_are_missing_and_forbidden_to_write_about_them():
    llm = FakeLLM(answer=["VOO closed at $694.81 [src-market#0]."] * 2)
    Synthesizer(llm).merge("q", SECTIONS, missing=[{"agent": "news", "status": "error"}])
    prompt, system = llm.calls["answer"][0]["prompt"], llm.calls["answer"][0]["system"]
    assert "PARTS THAT COULD NOT BE ANSWERED (say nothing about them): News" in prompt and "say NOTHING about them" in system


def test_a_single_surviving_branch_is_returned_as_written_with_no_merge_call():
    llm = FakeLLM()
    merged = Synthesizer(llm).merge("q", SECTIONS[:1], missing=[{"agent": "news", "status": "error"}])
    assert merged.mode == "single" and merged.text == SECTIONS[0]["text"] and llm.calls["answer"] == []


def test_filler_with_no_number_marker_or_link_is_caught_by_the_overlap_check():
    from src.workflow.synthesizer import ungrounded_sentences
    source = "VOO closed at 694.81 which is a decrease of 1.77 percent from the previous close on Friday"
    assert ungrounded_sentences("VOO closed at 694.81 which is a decrease from the previous close on Friday.", source) == []
    filler = "Apple continues to be a leading technology company known for its innovative products and strong presence."
    assert ungrounded_sentences(filler, source) == [filler[:80]]
    merged = Synthesizer(FakeLLM(answer=[filler])).merge("q", SECTIONS)
    assert merged.mode == "concat" and "not supported" in merged.issues[0]


def test_only_a_portfolio_result_may_be_a_dependency_other_edges_are_dropped_and_counted():
    from src.workflow.planner import drop_unjustified_dependencies
    tasks = validate_plan([task("t1", "market"), task("t2", "news", deps=["t1"]), task("t3", "portfolio"), task("t4", "goal", deps=["t3"])], 4)
    cleaned, dropped = drop_unjustified_dependencies(tasks)
    assert dropped == ["news after market"] and cleaned[1].depends_on == () and cleaned[3].depends_on == ("t3",)
    assert [[t.id for t in wave] for wave in levels(cleaned)] == [["t1", "t2", "t3"], ["t4"]]    # market and news now run together; goal still waits for the portfolio


def test_a_plan_with_an_invented_dependency_runs_its_independent_tasks_in_parallel_and_records_the_drop():
    plan = plan_reply(task("t1", "market"), task("t2", "news", deps=["t1"]), task("t3", "finance_qa"))
    graph, *_ = build(plan, delays={"market": 0.4, "news": 0.4, "finance_qa": 0.4})
    start = time.perf_counter()
    state = ask(graph)
    assert time.perf_counter() - start < 0.9 and state["plan"]["dropped_dependencies"] == ["news after market"]
