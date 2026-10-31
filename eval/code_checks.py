"""Rule-based answer checks (Step 10): our own evaluation, no LLM involved.

Every check is a plain function of (question, context, answer): deterministic, free, and identical on every run.
They are cruder than a judge: they catch the clear failures (invented numbers, missing citations, words that never
appear in the context) but cannot tell whether a sentence is *true to the meaning* of the context.

  numbers_grounded   every number in the answer also appears in the context (an invented figure is the classic hallucination)
  words_grounded     at least GROUNDED_MIN of the answer's content words appear in the context (the answer is built from it)
  every_sentence_cited  each sentence of 4+ words carries a [chunk#n] citation (rule 3 of the prompt)
  length_ok          at most 150 words (rule 5)
  has_citation / citation_valid / cited_expected   citations exist / were retrieved / point at the expected article
  refused_rule       the fixed gate text, or a refusal phrase with no citation
  states_gap_rule   the answer contains a "this part is not in the knowledge base" phrase
"""
import re

from src.rag.keyword_index import tokenize

GROUNDED_MIN = 0.8
MAX_WORDS = 150

_CITATION = re.compile(r"\[[^\]]*#\d+\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_REFUSAL = re.compile(r"(don't|do not|doesn't|does not) (have|know)|cannot (answer|help|provide)|can't (answer|help|provide)"
                      r"|not (in|part of) (my|the) (knowledge|context)|no information", re.I)
_GAP = re.compile(r"(does not|doesn't|do not|don't|did not|didn't) (say|mention|specify|provide|contain|include|give|cover|have)"
                  r"|(is|are) not (mentioned|specified|provided|included|covered|in the)|isn't (mentioned|specified|in the)"
                  r"|cannot|can't|not available|no (specific|information)|unable to", re.I)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(_CITATION.sub(" ", text))}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]


def check_answer(answer: str, context: str, cited_ids: list[str], invalid_citations: list[str], expected_ids: list[str]) -> dict:
    """Returns {check_name: score_or_value}. Scores are 0/1 except 'overlap' (a 0-1 share) and 'missing_*' (lists)."""
    answer_words = set(tokenize(_CITATION.sub(" ", answer)))
    overlap = len(answer_words & set(tokenize(_CITATION.sub(" ", context)))) / len(answer_words) if answer_words else 1.0
    missing_numbers = sorted(_numbers(answer) - _numbers(context))
    sentences = _sentences(answer)
    uncited = [s for s in sentences if not _CITATION.search(s)]
    has_citation = bool(cited_ids)
    return {
        "numbers_grounded": int(not missing_numbers), "missing_numbers": missing_numbers,
        "words_grounded": int(overlap >= GROUNDED_MIN), "overlap": overlap,
        "every_sentence_cited": int(not uncited), "uncited_sentences": uncited,
        "length_ok": int(len(answer.split()) <= MAX_WORDS),
        "has_citation": int(has_citation),
        "citation_valid": int(not invalid_citations),
        "cited_expected": int(any(c.split("#")[0] in expected_ids for c in cited_ids)) if expected_ids else None,
        "refused_rule": int(bool(_REFUSAL.search(answer)) and not has_citation),
        "states_gap_rule": int(bool(_GAP.search(answer))),
    }
