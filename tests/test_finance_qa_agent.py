from src.agents.finance_qa_agent import NO_INFO_TEXT, FinanceQAAgent, build_prompt
from src.core.llm import LLMResult
from src.core.schemas import AgentRequest
from src.rag.passages import Citation, Passage
from src.rag.retriever import RetrievalResult


def make_passage(chunk_id: str, text: str) -> Passage:
    citation = Citation(chunk_id.split("#")[0], "ETF Basics", "How ETFs work", "Investor.gov", "https://example.com", "2026-10-30")
    return Passage(chunk_id, text, 0.6, citation)


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def retrieve(self, question, top_k=None, threshold=None, **kwargs):
        self.calls.append((question, top_k, threshold))
        return RetrievalResult(question, self.passages, 0.6 if self.passages else 0.1, bool(self.passages),
                               len(self.passages), 0, 0, False)


class FakeLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def chat(self, prompt, system=None, **kwargs):
        self.prompts.append((prompt, system))
        return LLMResult(self.text, "fake-model", 100, 20, 0.5, 0.001, 1)


def ask(agent, message="what is an ETF"):
    return agent.handle(AgentRequest(message=message))


def test_no_passages_means_no_llm_call():
    llm = FakeLLM("should never be used")
    response = ask(FinanceQAAgent(llm, FakeRetriever([])))
    assert response.text == NO_INFO_TEXT
    assert llm.prompts == []                      # layer 1: the LLM was never called
    assert response.metadata["cost_usd"] == 0.0
    assert response.sources == []


def test_prompt_contains_ids_text_and_question_and_rules_stay_in_system():
    llm = FakeLLM("An ETF is a basket [etf-basics#0].")
    ask(FinanceQAAgent(llm, FakeRetriever([make_passage("etf-basics#0", "An ETF is a basket of investments.")])))
    prompt, system = llm.prompts[0]
    assert "[etf-basics#0]" in prompt and "An ETF is a basket of investments." in prompt and "what is an ETF" in prompt
    assert "Answer ONLY" in system and "Answer ONLY" not in prompt   # rules live in the system message


def test_citations_become_sources():
    llm = FakeLLM("An ETF is a basket [etf-basics#0]. It trades like a stock [etf-basics#1].")
    passages = [make_passage("etf-basics#0", "a"), make_passage("etf-basics#1", "b"), make_passage("etf-basics#2", "c")]
    response = ask(FinanceQAAgent(llm, FakeRetriever(passages)))
    assert [s.id for s in response.sources] == ["etf-basics#0", "etf-basics#1"]   # #2 was retrieved but not cited
    assert response.metadata["retrieved_ids"] == ["etf-basics#0", "etf-basics#1", "etf-basics#2"]
    assert response.metadata["invalid_citations"] == []


def test_invented_citation_is_flagged_and_not_turned_into_a_source():
    llm = FakeLLM("Something [made-up#9] and something real [etf-basics#0].")
    response = ask(FinanceQAAgent(llm, FakeRetriever([make_passage("etf-basics#0", "a")])))
    assert response.metadata["invalid_citations"] == ["made-up#9"]
    assert [s.id for s in response.sources] == ["etf-basics#0"]


def test_top_k_and_threshold_are_passed_to_the_retriever():
    retriever = FakeRetriever([])
    ask(FinanceQAAgent(FakeLLM("x"), retriever, top_k=3, threshold=0.0), "hello")
    assert retriever.calls == [("hello", 3, 0.0)]


def test_build_prompt_lists_every_passage():
    prompt = build_prompt("q?", [make_passage("a#0", "one"), make_passage("b#1", "two")])
    assert prompt.index("[a#0]") < prompt.index("[b#1]") and prompt.endswith("QUESTION: q?")
