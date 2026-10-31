import pytest

from src.agents.tax_agent import TAX_REDIRECT, TaxEducationAgent
from src.core.guardrails import DISCLAIMER_STRONG
from src.core.llm import LLMResult
from src.core.schemas import AgentRequest
from src.rag.passages import Citation, Passage
from src.rag.retriever import RetrievalResult


def passage(chunk_id="retirement-contribution-limits#1", text="The 2026 IRA limit is $7,500."):
    citation = Citation(chunk_id.split("#")[0], "Retirement Account Contribution Limits (Tax Year 2026)", "Contribution limits", "IRS", "https://irs.gov", "2026-10-31")
    return Passage(chunk_id, text, 0.6, citation)


class FakeRetriever:
    def __init__(self, passages):
        self.passages, self.calls = passages, []

    def retrieve(self, question, top_k=None, threshold=None, category=None, **kwargs):
        self.calls.append({"category": category})
        return RetrievalResult(question, self.passages, 0.6 if self.passages else 0.1, bool(self.passages), len(self.passages), 0, 0, False)


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, prompt, system=None, **kwargs):
        self.calls.append((prompt, system))
        return LLMResult(self.text, "fake", 100, 20, 0.5, 0.001, 1)


def ask(text, message, passages=None):
    llm, retriever = FakeLLM(text), FakeRetriever([passage()] if passages is None else passages)
    return TaxEducationAgent(llm, retriever).handle(AgentRequest(message=message)), llm, retriever


def test_retrieval_is_limited_to_the_tax_and_retirement_categories():
    _, _, retriever = ask("The IRA limit is $7,500 for tax year 2026 [retirement-contribution-limits#1].", "What is the IRA limit?")
    assert retriever.calls == [{"category": ["taxes", "retirement_accounts"]}]


def test_every_passage_shows_its_source_and_date_and_the_rules_demand_a_tax_year():
    _, llm, _ = ask("x", "What is the IRA limit?")
    prompt, system = llm.calls[0]
    assert "Source: IRS, updated 2026-10-31" in prompt and "[retirement-contribution-limits#1]" in prompt
    assert "tax year" in system and "Never state a limit that is not in the context" in system
    assert "7,500" not in system   # limits live in the knowledge base, never in the prompt


@pytest.mark.parametrize("message", [
    "How should I file my taxes this year?", "Should I convert my traditional IRA to a Roth?", "Can I deduct my IRA contribution?",
    "Am I eligible to contribute to a Roth IRA?", "How much tax will I owe on my gains?", "Which is better for me, a Roth or a traditional?",
    "Tell me exactly how to reduce my tax bill"])
def test_personal_tax_questions_get_the_personal_instruction_and_the_strong_disclaimer(message):
    response, llm, _ = ask("General rule explained [retirement-contribution-limits#1].", message)
    assert response.metadata["advice_requested"] and "OWN tax situation" in llm.calls[0][1]
    assert response.disclaimer == DISCLAIMER_STRONG


@pytest.mark.parametrize("message", [
    "What is the difference between a Roth IRA and a traditional IRA?", "What is the 401(k) contribution limit for 2026?",
    "How are short-term and long-term capital gains taxed?", "What is a catch-up contribution?"])
def test_educational_questions_are_not_treated_as_personal(message):
    response, llm, _ = ask("An explanation [retirement-contribution-limits#1].", message)
    assert not response.metadata["advice_requested"] and "OWN tax situation" not in llm.calls[0][1]


@pytest.mark.parametrize("leak", ["You should convert to a Roth now.", "You qualify for the deduction.", "You owe about $3,000.",
                                  "You are eligible to contribute.", "The best account for you is a Roth."])
def test_directive_or_eligibility_wording_is_replaced_by_the_tax_redirect(leak):
    response, _, _ = ask(leak, "What is a Roth IRA?")
    assert response.text == TAX_REDIRECT and response.metadata["output_flagged"] and response.metadata["original_text"] == leak


@pytest.mark.parametrize("fine", ["If you are 50 or older, you can add a catch-up of $1,100 [retirement-contribution-limits#1].",
                                  "You are eligible if your income is below the range [retirement-contribution-limits#1].",
                                  "Many people owe tax on short-term gains [investment-taxes-basics#0]."])
def test_general_rules_in_the_second_person_are_not_flagged(fine):
    response, _, _ = ask(fine, "What is the catch-up rule?", [passage(), passage("investment-taxes-basics#0", "text")])
    assert response.text == fine and not response.metadata["output_flagged"]


def test_nothing_retrieved_means_no_model_call():
    response, llm, _ = ask("never used", "How do I file taxes in Spain?", passages=[])
    assert llm.calls == [] and response.metadata["cost_usd"] == 0.0
    assert "licensed tax professional" in response.text and "stocks, bonds, ETFs" not in response.text   # a tax-specific message that refers on


@pytest.mark.parametrize("refusal", [
    "I can't tell you whether you should convert your IRA. In general, a conversion means paying tax now [retirement-contribution-limits#1].",
    "I can't say if you qualify; the rule uses income ranges [retirement-contribution-limits#1].",
    "What you owe depends on your income, so ask a tax professional [retirement-contribution-limits#1].",
    "It depends on how you should file, which a CPA can explain [retirement-contribution-limits#1]."])
def test_a_refusal_that_mentions_the_forbidden_phrase_is_kept_not_replaced(refusal):
    response, _, _ = ask(refusal, "Should I convert my IRA?")
    assert response.text == refusal and not response.metadata["output_flagged"]


@pytest.mark.parametrize("refusal", [
    "I can't determine which filing status you should choose. A licensed professional can help.",
    "I cannot say what you should claim on your return."])
def test_a_first_person_refusal_earlier_in_the_clause_exempts_the_match(refusal):
    response, _, _ = ask(refusal, "What filing status should I choose?")
    assert response.text == refusal and not response.metadata["output_flagged"]


@pytest.mark.parametrize("slip", [
    "I can't be sure, but you should convert your IRA this year.",
    "I can't promise anything. You should convert your IRA.",
    "You should convert. I can't tell you more."])
def test_a_refusal_in_another_clause_or_sentence_does_not_excuse_the_advice(slip):
    response, _, _ = ask(slip, "What is a Roth IRA?")
    assert response.text == TAX_REDIRECT and response.metadata["output_flagged"]
