import json

import pytest

from src.agents import goal_calc as gc
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.goal_extraction import GoalState, follow_up_text, invalid_fields
from src.core.config import GoalPlanningConfig, ScenarioReturns
from src.core.guardrails import DISCLAIMER_STRONG
from src.core.llm import LLMResult

CONFIG = GoalPlanningConfig(inflation_pct=3.0, large_monthly_usd=5000,
                            risk_profiles={"medium": ScenarioReturns(conservative=2.0, expected=5.0, optimistic=8.0)})


class FakeLLM:
    """chat() returns the next scripted reply: a dict becomes the extraction JSON, a string is plain text."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, prompt, system=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        reply = self.replies.pop(0)
        return LLMResult(reply if isinstance(reply, str) else json.dumps(reply), "fake", 50, 10, 0.1, 0.001, 1)


def fields(goal=None, years=None, savings=None, risk=None):
    return {"goal_amount_usd": goal, "years": years, "current_savings_usd": savings, "risk_profile": risk}


def agent(*replies):
    llm = FakeLLM(*replies)
    return GoalPlanningAgent(llm, CONFIG), llm


def test_a_complete_request_makes_a_plan_with_the_computed_numbers_in_the_metrics():
    a, llm = agent(fields(50000, 5, 2000, "medium"), "Here is your plan.")
    turn = a.run("save 50k in 5 years, 2k saved, medium risk")
    assert turn.plan_made and turn.needs == [] and turn.response.text == "Here is your plan."
    metrics = json.loads(llm.calls[1]["prompt"].split("METRICS (the only source of numbers):\n")[1].split("\n\nUSER MESSAGE")[0])
    assert [s["monthly_saving_usd"] for s in metrics["scenarios"]] == [758, 700, 645]
    assert metrics["future_dollars_needed_to_match_the_goals_buying_power_today_usd"] == 57964 and metrics["flags"] == []
    assert llm.calls[0]["response_format"]["json_schema"]["strict"] is True    # extraction used the strict schema
    assert "Use ONLY numbers" in llm.calls[1]["system"] and turn.response.disclaimer == DISCLAIMER_STRONG


def test_a_vague_request_asks_for_everything_and_makes_no_second_model_call():
    a, llm = agent(fields())
    turn = a.run("I want to retire comfortably")
    assert not turn.plan_made and turn.needs == ["goal_amount_usd", "years", "current_savings_usd", "risk_profile"]
    assert len(llm.calls) == 1 and turn.response.text.count("\n- ") == 4        # only the extraction call; four questions
    assert turn.response.metadata["model"].startswith("none")


def test_missing_fields_are_completed_over_several_turns_and_corrections_win():
    a, llm = agent(fields(goal=60000), fields(years=7, savings=0), fields(risk="medium"), "Plan.", fields(years=10), "Plan again.")
    t1 = a.run("I want 60k")
    assert t1.needs == ["years", "current_savings_usd", "risk_profile"] and "a goal of $60,000" in t1.response.text
    t2 = a.run("7 years, nothing saved", t1.state)
    assert t2.needs == ["risk_profile"] and t2.state.goal_amount_usd == 60000     # earlier answers are kept
    t3 = a.run("medium", t2.state)
    assert t3.plan_made and t3.state.model_dump() == {"goal_amount_usd": 60000, "years": 7, "current_savings_usd": 0, "risk_profile": "medium"}
    t4 = a.run("actually make it 10 years", t3.state)
    assert t4.state.years == 10 and t4.plan_made                                  # a later value replaces the earlier one


def test_the_extractor_is_told_what_the_user_is_replying_to():
    a, llm = agent(fields(years=30))
    a.run("30 years", GoalState())
    a2, llm2 = agent(fields(goal=1_500_000))
    a2.run("1.5 million", GoalState(years=30))
    assert "asked for: goal_amount_usd, current_savings_usd, risk_profile" in llm2.calls[0]["system"]


def test_unusable_values_are_dropped_and_explained_not_used():
    a, llm = agent(fields(goal=-500, years=5, savings=0, risk="medium"))
    turn = a.run("save -500 in 5 years")
    assert turn.needs == ["goal_amount_usd"] and "more than $0" in turn.response.text and turn.state.goal_amount_usd is None
    assert invalid_fields(GoalState(years=100)) == {"years": "The time must be more than 0 and at most 60 years."}


def test_an_unrealistic_goal_gets_flagged_in_the_metrics():
    a, llm = agent(fields(1_000_000, 1, 1000, "medium"), "It is a stretch.")
    a.run("turn 1k into 1m in a year")
    metrics = json.loads(llm.calls[1]["prompt"].split("METRICS (the only source of numbers):\n")[1].split("\n\nUSER MESSAGE")[0])
    assert any("very large commitment" in flag for flag in metrics["flags"])


def test_garbage_from_the_extractor_becomes_a_polite_request_to_rephrase_not_a_crash():
    a, llm = agent("this is not json")
    turn = a.run("save money")
    assert not turn.plan_made and "couldn't read that" in turn.response.text


def test_empty_and_oversized_messages_never_reach_the_model():
    a, llm = agent()
    assert "under 2,000" in a.run("   ").response.text and "under 2,000" in a.run("x" * 2001).response.text
    assert llm.calls == []


def test_an_advice_question_adds_the_advice_instruction():
    a, llm = agent(fields(50000, 5, 2000, "medium"), "Plan.")
    a.run("Should I put my 2k savings in stocks to reach 50k in 5 years? medium risk")
    assert "personal recommendation" in llm.calls[1]["system"]


def test_assess_flags():
    already = gc.plan_goal(10_000, 20, 9_000, "medium", CONFIG)
    assert "no extra monthly saving" in gc.assess(already, 5000)[0]
    short = gc.plan_goal(5_000, 0.5, 100, "medium", CONFIG)
    assert any("less than a year" in f for f in gc.assess(short, 5000))
    assert gc.assess(gc.plan_goal(50_000, 5, 2_000, "medium", CONFIG), 5000) == []


def test_follow_up_text_lists_only_what_is_missing():
    text = follow_up_text(GoalState(goal_amount_usd=50000, years=5))
    assert "a goal of $50,000, 5 year(s)" in text and text.count("\n- ") == 2 and "comfortable" in text and "How much money" not in text


def test_handle_without_a_goal_fails_loudly():
    from src.core.schemas import AgentRequest
    with pytest.raises(RuntimeError):
        agent()[0].handle(AgentRequest(message="hi"))
