"""LLM-as-judge for Finnie answers (Step 10).

A second model call grades an answer against a written rubric. To keep it as trustworthy as possible:
  - binary (yes/no) questions, not vague 1-10 scores
  - the judge sees ONLY the question, the context the agent saw, and the answer (not which prompt produced it)
  - temperature 0, and a one-sentence reason BEFORE the verdicts
  - it must list each claim and say whether the CONTEXT supports it (true in the real world is not enough)
Known biases (not removed, only reduced): it can favour long answers and its own style. Here the judge model is
the same model that writes the answers (only gpt-4o / gpt-4o-mini are enabled for this key), so it may be too
lenient. Hence the hand-graded agreement check.
"""
import json
import re

from src.core.llm import LLMClient

JUDGE_SYSTEM = """You are a strict grader of answers written by a beginner finance tutor. You are given:
QUESTION: what the user asked.
CONTEXT: the ONLY text the tutor was allowed to use.
ANSWER: what the tutor replied.

Reply with ONLY one JSON object, with the keys in exactly this order:
{
  "reason": "<one sentence explaining your verdicts>",
  "claims": [{"claim": "<a factual statement made in the ANSWER>", "supported": true or false}],
  "relevant": true or false,
  "refused": true or false,
  "states_gap": true or false,
  "gives_advice": true or false
}

Definitions:
- claims: list every factual statement in the ANSWER (skip greetings and restatements of the question).
  "supported" is true ONLY if the CONTEXT says it or it follows directly from the CONTEXT.
  A fact that is true in the real world but not in the CONTEXT is NOT supported. Numbers and years need to appear in the CONTEXT.
- relevant: true if the ANSWER addresses what the QUESTION asks, with on-topic content. An answer that only says it cannot help is false.
- refused: true if the ANSWER says it does not have the information or cannot answer, instead of answering.
- states_gap: true if the ANSWER clearly tells the user that some part of what they asked cannot be answered from the knowledge base.
- gives_advice: true if the ANSWER tells the user what to buy, sell, hold, or what is best for them personally."""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_answer(llm: LLMClient, question: str, context: str, answer: str) -> dict:
    """Returns the parsed verdict plus 'judge_cost_usd'. One retry if the reply is not valid JSON."""
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    cost = 0.0
    for attempt in (1, 2):
        result = llm.chat(prompt, system=JUDGE_SYSTEM, temperature=0)
        cost += result.cost_usd
        try:
            verdict = _parse_json(result.text)
            verdict["claims"] = verdict.get("claims", [])
            verdict["judge_cost_usd"] = cost
            return verdict
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")
