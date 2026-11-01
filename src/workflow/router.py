"""The router: a cheap model that reads a message and picks ONE route plus a confidence. It is a classifier, not an agent: it answers nothing.

Choices that make it dependable:
  - a small, cheap model (router_model) at temperature 0, so the same message gets the same route and the cost stays tiny
  - STRUCTURED OUTPUT: a strict JSON schema with an enum of routes and an enum of confidence levels, so it cannot invent either
  - a POLICY in code (not in the prompt): a "low" confidence route becomes `clarify`, because asking is cheaper than answering the wrong question
  - a safe default if the reply is somehow unusable: `clarify`, and say so (`fell_back`)
Prompts are versioned in PROMPTS so two versions can be compared with the same evaluation code.
"""
import json
from dataclasses import asdict, dataclass

from src.core.llm import LLMClient
from src.core.logging import get_logger

log = get_logger(__name__)
ROUTES = ("finance_qa", "market", "portfolio", "goal", "news", "tax", "clarify")
CONFIDENCE = ("high", "medium", "low")
DEFAULT_ROUTE = "clarify"

SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": list(ROUTES)}, "confidence": {"type": "string", "enum": list(CONFIDENCE)}},
    "required": ["route", "confidence"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": SCHEMA}}

_DEFINITIONS = """You route a user's message to exactly one agent and say how confident you are. Reply with the route and the confidence only.

Routes:
- finance_qa: a concept, definition, comparison or how-it-works question, answered from a knowledge base (what is an ETF, how do bonds work, what is compound interest).
- market: current or recent market DATA: a price or quote, how a named stock, fund or index is doing, a change, a trend, a moving average, volatility, a 52-week range, or a comparison of tickers' performance.
- portfolio: the user's OWN holdings: analyze, review or explain their portfolio (diversification, concentration, volatility or correlation of what they hold).
- goal: planning a savings goal: how much to save, by when, for a target amount.
- news: recent news or headlines about a company, a ticker, the Fed or the markets.
- tax: tax rules for investing and retirement accounts: contribution limits, how gains, dividends or withdrawals are taxed, tax treatment of IRAs and 401(k)s.
- clarify: the message is about money or finance but is too vague, or depends on something said earlier that is not available (for example only "it", "that one" or "tell me more").

Confidence: high = clearly one route; medium = probably this route; low = it could reasonably be more than one route, or you are guessing."""

_V2_TAX = ("- tax: anything about TAXES on investing and retirement accounts: contribution limits, how gains, dividends, interest or withdrawals are taxed, "
           "penalties and deductions, tax rules and tax terms (qualified dividend, cost basis, wash sale, tax-loss harvesting, net investment income tax), tax forms. "
           "Route a tax TERM or tax RULE here even when it is phrased as a plain concept question (\"what is ...\").")
_V2_DEFINITIONS = _DEFINITIONS.replace(
    "- tax: tax rules for investing and retirement accounts: contribution limits, how gains, dividends or withdrawals are taxed, tax treatment of IRAs and 401(k)s.", _V2_TAX)
assert _V2_DEFINITIONS != _DEFINITIONS

_V2_EXAMPLES = """

Tie-breakers:
- If the question is mostly about what is TAXED or what a tax rule or penalty is, choose tax, not finance_qa.
- A concept with no tax angle (an account type, a bond term, an index) is finance_qa, even if a ticker or company is mentioned as an example.
- A question about the user's OWN holdings is portfolio; a question about how a named company or fund is doing right now is market; "what happened / what did X say" is news.

Examples:
"What is the capital gains tax rate?" -> tax, high
"What does the alternative minimum tax do?" -> tax, high
"What is Form 8949 used for?" -> tax, high
"What is asset allocation?" -> finance_qa, high
"What is beta?" -> finance_qa, high
"What is Intel's beta right now?" -> market, high
"What did the Fed decide this week?" -> news, high
"How is my retirement account allocated?" -> portfolio, medium"""
PROMPTS = {"v1": _DEFINITIONS, "v2": _V2_DEFINITIONS + _V2_EXAMPLES}


@dataclass
class RouteDecision:
    route: str                 # the route the graph will follow (after the low-confidence policy)
    confidence: str            # what the model said: high | medium | low
    raw_route: str             # what the model chose, before the policy
    overridden: bool           # True = the policy changed raw_route to clarify
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    fell_back: bool = False    # True = the reply was unusable and DEFAULT_ROUTE was used

    def as_dict(self) -> dict:
        return asdict(self)


class Router:
    def __init__(self, llm: LLMClient, model_key: str = "router_model", prompt_version: str = "v2", low_confidence_to_clarify: bool = True) -> None:
        if prompt_version not in PROMPTS:
            raise ValueError(f"prompt_version must be one of {sorted(PROMPTS)}")
        self.llm, self.model_key, self.prompt_version = llm, model_key, prompt_version
        self.low_confidence_to_clarify = low_confidence_to_clarify

    def classify(self, message: str) -> RouteDecision:
        result = self.llm.chat(f"MESSAGE: {message}", system=PROMPTS[self.prompt_version], model_key=self.model_key, temperature=0,
                               max_output_tokens=30, response_format=RESPONSE_FORMAT)
        try:
            data = json.loads(result.text)
            raw_route, confidence, fell_back = data["route"], data["confidence"], False
            if raw_route not in ROUTES or confidence not in CONFIDENCE:
                raise ValueError(result.text)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.warning("unusable router reply %r; falling back to %s", result.text[:80], DEFAULT_ROUTE)
            raw_route, confidence, fell_back = DEFAULT_ROUTE, "low", True
        route = raw_route
        overridden = self.low_confidence_to_clarify and confidence == "low" and raw_route != "clarify"
        if overridden:
            route = "clarify"
        return RouteDecision(route, confidence, raw_route, overridden, result.model, result.input_tokens, result.output_tokens,
                             result.cost_usd, result.latency_s, fell_back)
