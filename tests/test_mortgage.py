"""Mortgage payoff arithmetic. Figures worked out by hand or by the closed-form formula n = -ln(1 - B*r/P) / ln(1 + r)."""
from datetime import date
from decimal import Decimal as D

from src.agents.mortgage import amortize, mortgage_payoff
from src.data.portfolio_repository import PropertyIn

TODAY = date(2026, 11, 6)


def prop(balance=200000, payment=1500, rate=6, **extra):
    return PropertyIn(name="Maple St", purchase_price=D(300000), current_value=D(400000), mortgage_balance=D(balance),
                      monthly_mortgage_payment=D(payment), mortgage_rate_pct=D(rate), **extra)


def test_zero_rate_is_simple_division():
    r = amortize(D(12000), D(0), D(1000))
    assert (r["months"], r["total_interest"], r["total_paid"]) == (12, 0, 12000)


def test_zero_rate_last_payment_is_smaller():
    r = amortize(D(2500), D(0), D(1000))
    assert (r["months"], r["last_payment"]) == (3, 500)


def test_known_loan_matches_the_formula():
    # 200,000 at 6% with 1,500 a month: n = -ln(1 - 1000/1500) / ln(1.005) = 220.3 -> 221 payments, the last one partial
    r = amortize(D(200000), D(6), D(1500))
    assert r["months"] == 221
    assert r["total_paid"] == D(200000) + r["total_interest"]


def test_payment_that_only_covers_interest_never_pays_off():
    assert amortize(D(200000), D(6), D(1000)) is None      # interest alone is exactly 1,000 a month
    assert amortize(D(200000), D(6), D(900)) is None


def test_extra_payment_is_sooner_and_cheaper():
    a = mortgage_payoff(prop(), D(500), TODAY)
    assert a["status"] == "ok"
    assert a["regular"]["months"] == 221 and a["with_extra"]["months"] < 221
    assert a["saved"]["months"] == 221 - a["with_extra"]["months"] and a["saved"]["interest"] > 0
    assert a["regular"]["payoff_date"] == "2045-04-01"      # Nov 2026 + 221 months


def test_no_extra_means_no_comparison():
    a = mortgage_payoff(prop(), D(0), TODAY)
    assert a["with_extra"] is None and a["saved"] is None


def test_missing_inputs_are_named_not_guessed():
    assert mortgage_payoff(prop(balance=0), today=TODAY)["status"] == "no_mortgage"
    assert mortgage_payoff(prop(payment=0), today=TODAY)["status"] == "missing_payment"
    assert mortgage_payoff(prop(payment=1000), today=TODAY)["status"] == "never_paid_off"


def test_old_properties_without_the_new_fields_still_load():
    p = PropertyIn(name="Old", purchase_price=D(1), current_value=D(1), mortgage_balance=D(100))
    assert (p.monthly_mortgage_payment, p.mortgage_rate_pct) == (0, 0)


# ---- chat: recognizing the question and answering it from saved properties
from types import SimpleNamespace
from uuid import uuid4

from src.agents.mortgage_chat import MortgagePayoffChat, extra_amount, is_mortgage_payoff_question


def test_payoff_questions_are_recognized_and_general_ones_are_not():
    for q in ("How long will it take to pay off my mortgage?", "If I pay $500 extra on my mortgage, how many years sooner do I finish?", "Show me the amortization for my mortgage"):
        assert is_mortgage_payoff_question(q), q
    for q in ("What is a mortgage?", "Are mortgage rates going up?", "How do I diversify my portfolio?"):
        assert not is_mortgage_payoff_question(q), q


def test_extra_amount_is_read_from_the_question():
    assert extra_amount("what if I pay $500 extra") == D(500)
    assert extra_amount("an additional 1,200 a month") == D(1200)
    assert extra_amount("pay $2k more on my mortgage") == D(2000)
    assert extra_amount("how long is my mortgage") is None


class Repo:
    """Saved portfolios as {name: [properties]}; a plain list is one portfolio called Mine."""
    def __init__(self, properties):
        self.portfolios = properties if isinstance(properties, dict) else {"Mine": properties}
    def list(self, owner_id): return [SimpleNamespace(id=name) for name in self.portfolios]
    def get(self, portfolio_id, owner_id): return SimpleNamespace(name=portfolio_id, properties=self.portfolios[portfolio_id])


def test_chat_answer_has_the_calculator_numbers_and_the_extra_comparison():
    agent = MortgagePayoffChat(lambda: Repo([prop()]))
    text = agent.run("If I pay $500 extra on my mortgage how soon do I finish?", uuid4(), uuid4()).text
    assert "18 years 5 months" in text and "Paying $500 more every month" in text and "sooner" in text    # 221 months = 18 years 5 months


def test_chat_without_a_login_says_what_to_do():
    reply = MortgagePayoffChat(lambda: Repo([])).run("how long is my mortgage", None, None)
    assert "log in" in reply.text


def test_chat_needs_no_portfolio_to_be_picked_first():
    text = MortgagePayoffChat(lambda: Repo([prop()])).run("how long until my mortgage is paid off?", None, uuid4()).text
    assert "18 years 5 months" in text


def test_chat_asks_for_the_missing_payment():
    text = MortgagePayoffChat(lambda: Repo([prop(payment=0)])).run("when will my mortgage be paid off", uuid4(), uuid4()).text
    assert "haven't entered a monthly payment" in text


def test_the_graph_sends_a_payoff_question_to_the_calculator_without_the_router_model():
    from unittest.mock import MagicMock
    from src.core.config import load_settings
    from src.workflow.graph import Agents, build_graph
    from langgraph.checkpoint.memory import InMemorySaver
    router = MagicMock()
    router.classify.side_effect = AssertionError("the router model must not be called")
    agents = Agents(*[MagicMock() for _ in range(6)], mortgage=MortgagePayoffChat(lambda: Repo([prop()])))
    graph = build_graph(MagicMock(), router, agents, load_settings().workflow, checkpointer=InMemorySaver())
    state = graph.invoke({"message": "how many years until my mortgage is paid off?", "session_id": "s", "owner_id": str(uuid4()), "portfolio_id": str(uuid4())},
                         {"configurable": {"thread_id": "s"}})
    assert state["route"] == "mortgage" and "18 years 5 months" in state["response"]["text"]


# ---- every portfolio, addresses, and follow-up questions
def named(name, **extra):
    return PropertyIn(name=name, purchase_price=D(300000), current_value=D(400000), mortgage_balance=D(extra.pop("balance", 200000)),
                      monthly_mortgage_payment=D(extra.pop("payment", 1500)), mortgage_rate_pct=D(extra.pop("rate", 6)), **extra)


TWO = {"Rentals": [named("Maple St")], "Retirement": [named("123 Opossum Rd", balance=100000, payment=1000)]}


def test_a_payoff_question_answers_for_the_mortgages_of_every_saved_portfolio():
    text = MortgagePayoffChat(lambda: Repo(TWO)).run("how long until my mortgage is paid off?", None, uuid4()).text
    assert "Maple St (in Rentals)" in text and "123 Opossum Rd (in Retirement)" in text and "2 properties with a mortgage" in text


def test_an_address_is_found_even_with_road_for_rd_and_a_typo():
    for said in ("what about my 123 opposum road real estate portfolios", "and 123 Opossum Rd?", "how about the 123 opossum road house"):
        text = MortgagePayoffChat(lambda: Repo(TWO)).run(said, None, uuid4()).text
        assert "123 Opossum Rd (in Retirement)" in text and "Maple St" not in text, said


def test_an_unknown_address_is_said_so_and_everything_is_shown():
    text = MortgagePayoffChat(lambda: Repo(TWO)).run("what about 999 Nowhere Lane", None, uuid4()).text
    assert "couldn't find a property matching" in text and "Maple St" in text and "Opossum" in text


def test_a_property_without_a_mortgage_is_not_given_a_made_up_payoff():
    text = MortgagePayoffChat(lambda: Repo({"A": [named("Oak Ave", balance=0)], "B": [named("Maple St")]})).run("what about Oak Avenue", None, uuid4()).text
    assert "Oak Ave" in text and "no mortgage balance" in text and "years" not in text


def test_a_house_number_must_match_exactly():
    from src.agents.mortgage_chat import mentions
    assert mentions("123 Opossum Rd", "my 123 opposum road") and not mentions("123 Opossum Rd", "my 124 opossum road")
    assert not mentions("Maple St", "what is a street address?")


def test_an_extra_amount_from_the_earlier_question_still_applies_to_the_follow_up():
    text = MortgagePayoffChat(lambda: Repo(TWO)).run("what about 123 Opossum Rd", None, uuid4(), earlier="what if I pay $500 extra on my mortgage?").text
    assert "Paying $500 more every month" in text


def test_follow_ups_are_recognized_and_other_questions_are_not():
    from src.agents.mortgage_chat import is_mortgage_follow_up
    for q in ("what about my 123 opposum road real estate portfolios", "and what if I pay $500 extra?", "how about the other property", "what about the rental?"):
        assert is_mortgage_follow_up(q), q
    for q in ("what about my taxes?", "what is an ETF?", "and how is the S&P doing today?"):
        assert not is_mortgage_follow_up(q), q


def two_turn_graph():
    from unittest.mock import MagicMock, patch
    from src.core.config import load_settings
    from src.workflow.graph import Agents, build_graph
    from langgraph.checkpoint.memory import InMemorySaver
    router = MagicMock()
    router.classify.side_effect = AssertionError("the router model must not be called for a mortgage conversation")
    agents = Agents(*[MagicMock() for _ in range(6)], mortgage=MortgagePayoffChat(lambda: Repo(TWO)))
    with patch("src.workflow.graph.rewrite_message", lambda llm, message, *a, **k: (message, None)):
        graph = build_graph(MagicMock(), router, agents, load_settings().workflow, checkpointer=InMemorySaver())
        ask = lambda message: graph.invoke({"message": message, "session_id": "s", "owner_id": str(uuid4())}, {"configurable": {"thread_id": "s"}})
        return ask


def test_a_follow_up_about_an_address_stays_on_the_mortgage_conversation():
    ask = two_turn_graph()
    first = ask("how long will it take to close my real estate mortgage?")
    second = ask("what about my 123 opposum road real estate portfolios")
    assert first["route"] == "mortgage" and "Maple St (in Rentals)" in first["response"]["text"] and "Opossum" in first["response"]["text"]
    assert second["route"] == "mortgage" and second["router"]["guard"] == "mortgage_follow_up"
    assert "123 Opossum Rd (in Retirement)" in second["response"]["text"] and "Maple St" not in second["response"]["text"]


def test_the_extra_payment_asked_earlier_carries_into_the_follow_up_through_the_graph():
    ask = two_turn_graph()
    ask("if I pay $500 extra on my mortgage how soon do I finish?")
    second = ask("what about 123 opossum road")
    assert "Paying $500 more every month" in second["response"]["text"]


def test_without_a_mortgage_conversation_the_same_words_are_not_hijacked():
    from unittest.mock import MagicMock
    from src.core.config import load_settings
    from src.workflow.graph import Agents, build_graph
    from langgraph.checkpoint.memory import InMemorySaver
    router = MagicMock()
    router.classify.return_value = SimpleNamespace(route="finance_qa", raw_route="finance_qa", confidence="high", overridden=False, latency_s=0.0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                                                   fell_back=False, as_dict=lambda: {"route": "finance_qa"})
    agents = Agents(*[MagicMock() for _ in range(6)], mortgage=MortgagePayoffChat(lambda: Repo(TWO)))
    agents.finance_qa.handle.return_value = __import__("src.core.schemas", fromlist=["AgentResponse"]).AgentResponse(text="general answer", agent_name="finance_qa")
    graph = build_graph(MagicMock(), router, agents, load_settings().workflow, checkpointer=InMemorySaver())
    state = graph.invoke({"message": "what about my 123 opposum road property", "session_id": "t", "owner_id": str(uuid4())}, {"configurable": {"thread_id": "t"}})
    assert state["route"] == "finance_qa"      # first message of a conversation, nothing about a mortgage before it
