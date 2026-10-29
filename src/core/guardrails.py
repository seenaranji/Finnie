"""Code-level guardrails: checks that run in plain Python, so they cannot be
talked out of their job the way a prompt instruction can.

Two checks:
  - is_advice_request(user_message): is the user asking us to pick for them?
  - find_advice_phrases(model_output): did the answer slip into directive advice?

These are RULES (regular expressions). They are fast and predictable but they only
see exact word patterns, so rephrasings can slip through. That is why we measure
them on a test set (eval/datasets/guardrails.jsonl) instead of trusting them.
"""
import re

DISCLAIMER_STANDARD = (
    "This is general education, not personalized financial advice. "
    "Examples are illustrative only."
)
DISCLAIMER_STRONG = (
    "This is general education, not personalized financial, investment, or tax advice. "
    "Projections are illustrative, not guaranteed, and past performance does not predict "
    "future results. Consider speaking with a licensed professional about your situation."
)

ADVICE_REDIRECT = (
    "I can't tell you what you personally should buy or sell, but I can explain how "
    "people think about this kind of decision, what the tradeoffs are, and what to "
    "consider. A licensed financial professional can look at your whole situation."
)

# Patterns that suggest the USER wants a personal recommendation.
_REQUEST_PATTERNS = [
    r"\bshould i\b.*\b(buy|sell|invest|put|move|choose|pick|open|hold)\b",
    r"\bwhere should i (put|invest|park|keep)\b",
    r"\b(good|great|best|smart|safe)\s+(buy|investment|pick|stock|move)\b",
    r"\bis \w+(\s\w+)? worth (buying|investing)\b",
    r"\bwhat (stock|etf|fund)s?\b.*\b(should i|to buy|to invest)\b",
    r"\btell me what to (buy|sell|invest)\b",
]

# Patterns that suggest the MODEL'S ANSWER is giving directive advice.
_OUTPUT_PATTERNS = [
    r"\byou should (buy|sell|invest|put|move|open|choose)\b",
    r"\bi (strongly )?(recommend|suggest|advise)\b",
    r"\b(the )?(best|right|smart) (move|choice|investment|decision)\b",
    r"\byou('d| would) be better off\b",
    r"\bi('d| would) (buy|sell|invest|put)\b",
]

_REQUEST_RE = [re.compile(p, re.IGNORECASE) for p in _REQUEST_PATTERNS]
_OUTPUT_RE = [re.compile(p, re.IGNORECASE) for p in _OUTPUT_PATTERNS]


def is_advice_request(user_message: str) -> bool:
    return any(rx.search(user_message) for rx in _REQUEST_RE)


def find_advice_phrases(model_output: str) -> list[str]:
    """Return the matched snippets (empty list = nothing suspicious found)."""
    return [m.group(0) for rx in _OUTPUT_RE if (m := rx.search(model_output))]
