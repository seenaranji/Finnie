import json

import numpy as np

from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.core.config import load_settings
from src.core.guardrails import DISCLAIMER_STRONG
from src.core.llm import LLMResult
from src.data.cache import TTLCache
from src.data.errors import ProviderUnavailable
from src.data.market_data import MarketDataService
from src.data.models import PriceHistory, PricePoint, Quote
from src.data.providers.base import MarketDataProvider

META = {"AAA": {"asset_class": "US stock", "sector": "Tech"}, "BBB": {"asset_class": "Bond", "sector": "Bonds"}}
CSV = "ticker,shares,avg_cost\nAAA,10,50\nBBB,20,40\n"


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, down=()):
        self.down = set(down)

    def get_quote(self, symbol):
        if symbol in self.down:
            raise ProviderUnavailable("The market data service is not responding right now.")
        return Quote(symbol=symbol, price=100.0 if symbol == "AAA" else 50.0, source=self.name, as_of="2026-10-30")

    def get_history(self, symbol, period):
        rng = np.random.default_rng(sum(map(ord, symbol)))
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 260))
        return PriceHistory(symbol=symbol, period=period, source=self.name,
                            points=[PricePoint(date=f"d{i:04d}", close=float(c)) for i, c in enumerate(closes)])


class FakeLLM:
    def __init__(self, text="Your portfolio is 50.0% in AAA."):
        self.text, self.calls = text, []

    def chat(self, prompt, system=None, **kwargs):
        self.calls.append((prompt, system))
        return LLMResult(self.text, "fake", 100, 20, 0.5, 0.001, 1)


def make_agent(llm=None, down=()):
    service = MarketDataService(FakeProvider(down), TTLCache(), load_settings(), sleep=lambda s: None)
    return PortfolioAnalysisAgent(llm or FakeLLM(), service, META)


def test_metrics_block_holds_the_computed_numbers_and_rules_stay_in_the_system_message():
    llm = FakeLLM()
    response = make_agent(llm).run(CSV, "Explain it")
    prompt, system = llm.calls[0]
    metrics = json.loads(prompt.split("METRICS (the only source of numbers):\n")[1].split("\n\nUSER QUESTION")[0])
    assert metrics["holding_count"] == 2 and metrics["total_value_usd"] == 2000        # 10*100 + 20*50
    assert [h["weight_pct"] for h in metrics["holdings"]] == [50.0, 50.0] and metrics["hhi_concentration_0_to_1"] == 0.5
    assert metrics["total_gain_pct"] == 53.8 and metrics["prices_as_of"] == "2026-10-30"   # cost 10*50 + 20*40 = 1300; 2000 / 1300 - 1 = 53.8%
    assert "Use ONLY numbers" in system and "Use ONLY numbers" not in prompt
    assert response.disclaimer == DISCLAIMER_STRONG and [s.id for s in response.sources] == ["fake"]


def test_a_bad_csv_gets_a_clear_message_and_no_model_call():
    llm = FakeLLM()
    response = make_agent(llm).run("ticker,shares,avg_cost\nAAA,-5,10\n,3,4\n")
    assert llm.calls == [] and "couldn't read your portfolio file" in response.text and response.metadata["cost_usd"] == 0.0
    assert response.text.count("line ") == 2   # both problems listed


def test_nothing_priced_means_no_model_call():
    llm = FakeLLM()
    response = make_agent(llm, down={"AAA", "BBB"}).run(CSV)
    assert llm.calls == [] and "couldn't get prices" in response.text


def test_a_partly_priced_portfolio_says_so_in_the_notes():
    llm = FakeLLM()
    make_agent(llm, down={"BBB"}).run(CSV)
    metrics = json.loads(llm.calls[0][0].split("METRICS (the only source of numbers):\n")[1].split("\n\nUSER QUESTION")[0])
    assert metrics["holding_count"] == 1 and any("BBB" in note for note in metrics["notes"])


def test_an_advice_question_adds_the_advice_instruction_and_directive_output_is_replaced():
    llm = FakeLLM("You should sell AAA now.")
    response = make_agent(llm).run(CSV, "Should I sell my biggest holding?")
    assert "personal recommendation" in llm.calls[0][1]
    assert response.metadata["output_flagged"] and "You should sell" not in response.text


def test_handle_without_a_portfolio_fails_loudly():
    import pytest
    from src.core.schemas import AgentRequest
    with pytest.raises(RuntimeError):
        make_agent().handle(AgentRequest(message="hi"))


def test_two_threads_sharing_one_agent_each_get_their_own_portfolio_explained():
    """The API shares one agent across sessions. The agent keeps per-call scratch state, so it must be per thread or answers get swapped."""
    import threading
    import time

    class SlowProvider(FakeProvider):
        def get_history(self, symbol, period):
            time.sleep(0.15 if symbol in ("AAA", "BBB") else 0.01)   # make the two calls overlap in time
            return super().get_history(symbol, period)

    class EchoLLM:
        def chat(self, prompt, system=None, **kwargs):
            time.sleep(0.2)
            holdings = json.loads(prompt.split("METRICS (the only source of numbers):\n")[1].split("\n\nUSER QUESTION")[0])["holding_count"]
            return LLMResult(f"You hold {holdings} positions.", "fake", 1, 1, 0.1, 0.0, 1)

    service = MarketDataService(SlowProvider(), TTLCache(), load_settings(), sleep=lambda s: None)
    agent = PortfolioAnalysisAgent(EchoLLM(), service, {**META, "CCC": {"asset_class": "US stock", "sector": "Tech"}})
    two, three = "ticker,shares,avg_cost\nAAA,10,50\nBBB,20,40\n", "ticker,shares,avg_cost\nAAA,10,50\nBBB,20,40\nCCC,5,30\n"
    answers = {}

    def go(name, csv):
        answers[name] = agent.run(csv).text
    threads = [threading.Thread(target=go, args=("two", two)), threading.Thread(target=go, args=("three", three))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert answers == {"two": "You hold 2 positions.", "three": "You hold 3 positions."}


# ---------- Step 25: prices are fetched side by side ----------
def test_holdings_are_fetched_concurrently():
    import threading

    class BarrierProvider(FakeProvider):
        """Every quote call waits until three calls are in flight. Sequential code can never get three in flight, so it times out and fails the test."""
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(3, timeout=3)

        def get_quote(self, symbol):
            self.barrier.wait()
            return super().get_quote(symbol)

    service = MarketDataService(BarrierProvider(), TTLCache(), load_settings(), sleep=lambda s: None)
    response = PortfolioAnalysisAgent(FakeLLM(), service, META).run("ticker,shares,avg_cost\nAAA,10,50\nBBB,20,40\nCCC,5,10\n")
    assert response.metadata["metrics"]["holding_count"] == 3


def test_a_failure_in_one_holding_is_reported_in_holding_order_and_the_rest_still_count():
    response = make_agent(down={"BBB", "CCC"}).run("ticker,shares,avg_cost\nAAA,10,50\nBBB,20,40\nCCC,5,10\n")
    notes = response.metadata["metrics"]["notes"]
    assert response.metadata["metrics"]["holding_count"] == 1
    assert [n for n in notes if "could not fetch" in n][0].index("BBB") < [n for n in notes if "could not fetch" in n][0].index("CCC")
