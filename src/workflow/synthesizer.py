"""The synthesizer (Step 21b): merges the answers of several branches into ONE answer.

The branches' answers are trusted agent output, but the model that merges them can still add or alter a fact, so its text is CHECKED IN CODE:
every number and every citation marker in the merged text must appear in the branch answers, and no link may appear. If a check fails, the merge falls back to a
code-built version (each branch's answer under a plain heading). What was NOT answered is always described by code (`missing_note`), never by the model.
"""
import re
from dataclasses import dataclass, field

from src.core.llm import LLMClient
from src.core.logging import get_logger

log = get_logger(__name__)

TITLES = {"finance_qa": "Concept", "market": "Market data", "news": "News", "portfolio": "Your portfolio", "goal": "Savings goal", "tax": "Taxes"}
_CITATION = re.compile(r"\[[^\]\s][^\]]*\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LIST_NUMERAL = re.compile(r"^\s*\d{1,2}[.)]\s+", re.M)   # "1. first point": an item number, not a fact
_URL = re.compile(r"https?://", re.I)
_WORD = re.compile(r"[a-z]{4,}")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_CHARS = 2400
MIN_SENTENCE_OVERLAP = 0.35   # a merged sentence whose content words are mostly absent from the branch answers is filler or invention
SYSTEM_PROMPT = """You combine the answers of several specialist agents into ONE answer for a beginner.

Rules:
1. Use ONLY what is in the SECTIONS. Do not add facts, numbers, advice or predictions.
2. Copy numbers, dates and citation markers (like [etf-basics#0] or [2]) exactly as they appear, and keep each marker next to the claim it supports.
3. Answer in the order the user asked. Use short headings or bullets, simple words, under 250 words.
4. Do not mention agents, sections, or that you combined anything. Do not write links.
5. Do not add a disclaimer; one is added separately.
6. If the message lists PARTS THAT COULD NOT BE ANSWERED, say NOTHING about them: no section, no heading, no general statement, no filler (for example what a company is known for). They are reported separately."""


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(_LIST_NUMERAL.sub("", _CITATION.sub(" ", text)))}


@dataclass
class Merged:
    text: str
    mode: str                       # "llm" or "concat"
    issues: list[str] = field(default_factory=list)   # why the model's merge was rejected (empty when mode == "llm")
    cost_usd: float = 0.0
    tokens: int = 0
    latency_s: float = 0.0
    model: str = "none (concat)"


def ungrounded_sentences(merged: str, source_text: str, minimum: float = MIN_SENTENCE_OVERLAP) -> list[str]:
    """Sentences of 8+ words where fewer than `minimum` of the 4+ letter words appear in the branch answers. Catches filler that has no number, marker or link
    for the other checks to see. Lexical, so it cannot judge meaning; a paraphrase with few shared words can be flagged wrongly (the merge then falls back to the code merge)."""
    source_words, flagged = set(_WORD.findall(source_text.lower())), []
    for sentence in _SENTENCE.split(merged):
        words = _WORD.findall(sentence.lower())
        if len(sentence.split()) >= 8 and words and sum(w in source_words for w in words) / len(words) < minimum:
            flagged.append(sentence.strip()[:80])
    return flagged


def concat(results: list[dict]) -> str:
    return "\n\n".join(f"**{TITLES.get(r['agent'], r['agent'])}**\n{r['text']}" for r in results)


def missing_note(missing: list[dict], everything: bool = False) -> str:
    """Written by code from fixed phrases: it names what is missing and why, in user-safe words, with no internals. `everything` = no part could be answered."""
    if not missing:
        return ""
    reasons = {"error": "that part isn't available right now", "timeout": "that part took too long", "skipped": "it depended on another part that isn't available"}
    lines = [f"- {TITLES.get(m['agent'], m['agent'])}: {m.get('degraded_reason') or reasons.get(m['status'], 'not available')}" for m in missing]
    header = "I couldn't get any part of that right now:" if everything else "I couldn't include everything you asked for:"
    return header + "\n" + "\n".join(lines) + "\nYou can ask for the missing part again in a moment."


class Synthesizer:
    def __init__(self, llm: LLMClient, mode: str = "llm", model_key: str = "answer_model") -> None:
        self.llm, self.mode, self.model_key = llm, mode, model_key

    def merge(self, question: str, results: list[dict], missing: list[dict] | None = None) -> Merged:
        """`results` are the successful branches, in the order of the plan; `missing` are the parts that could not be answered (the model is told to leave them out)."""
        if self.mode == "concat":
            return Merged(concat(results), "concat")
        if len(results) == 1:   # nothing to combine: asking a model to "merge" one answer only invites it to add filler
            return Merged(results[0]["text"], "single")
        sections = "\n\n".join(f"### {TITLES.get(r['agent'], r['agent'])}\n{r['text']}" for r in results)
        unanswered = ("\n\nPARTS THAT COULD NOT BE ANSWERED (say nothing about them): " + ", ".join(TITLES.get(m["agent"], m["agent"]) for m in missing)) if missing else ""
        try:
            result = self.llm.chat(f"USER QUESTION: {question}\n\nSECTIONS:\n{sections}{unanswered}", system=SYSTEM_PROMPT, model_key=self.model_key, temperature=0)
        except Exception as error:   # the merging model is down: the branches' answers are still good, so show them with plain headings
            log.warning("synthesizer model failed (%s); using the code-built merge", type(error).__name__)
            return Merged(concat(results), "concat", [f"the merging model failed: {type(error).__name__}"])
        source_text = " ".join(r["text"] for r in results)
        issues = []
        extra_numbers = _numbers(result.text) - _numbers(source_text)
        if extra_numbers:
            issues.append(f"numbers not in the branch answers: {sorted(extra_numbers)}")
        extra_markers = set(_CITATION.findall(result.text)) - set(_CITATION.findall(source_text))
        if extra_markers:
            issues.append(f"citation markers not in the branch answers: {sorted(extra_markers)}")
        filler = ungrounded_sentences(result.text, source_text)
        if filler:
            issues.append(f"sentences not supported by the branch answers: {filler}")
        if _URL.search(result.text):
            issues.append("contains a link")
        if len(result.text) > MAX_CHARS:
            issues.append(f"longer than {MAX_CHARS} characters")
        if issues:
            log.warning("synthesized answer rejected (%s); using the code-built merge", issues)
            return Merged(concat(results), "concat", issues, result.cost_usd, result.input_tokens + result.output_tokens, result.latency_s, result.model)
        return Merged(result.text.strip(), "llm", [], result.cost_usd, result.input_tokens + result.output_tokens, result.latency_s, result.model)
