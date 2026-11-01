"""The router: a cheap model that reads a message and picks ONE route plus a confidence. It is a classifier, not an agent: it answers nothing.

Choices that make it dependable:
  - a small, cheap model (router_model) at temperature 0, so the same message gets the same route and the cost stays tiny
  - STRUCTURED OUTPUT: a strict JSON schema with an enum of routes and an enum of confidence levels, so it cannot invent either
  - a POLICY in code (not in the prompt): a "low" confidence route becomes `clarify`, because asking is cheaper than answering the wrong question
  - a safe default if the reply is somehow unusable: `clarify`, and say so (`fell_back`)
Prompts are versioned in PROMPTS so two versions can be compared with the same evaluation code.
"""
import json
import re
from dataclasses import asdict, dataclass

from src.core.llm import LLMClient
from src.core.logging import get_logger

log = get_logger(__name__)
ROUTES_V2 = ("finance_qa", "market", "portfolio", "goal", "news", "tax", "clarify")   # the labels prompts v1 and v2 know
ROUTES_V5 = ROUTES_V2 + ("out_of_scope",)                                              # prompts v3-v5 add the scope gate (Step 21a)
ROUTES = ROUTES_V5 + ("multi",)                                                         # prompt v6 adds multi-intent routing (Step 21b)
CONFIDENCE = ("high", "medium", "low")
DEFAULT_ROUTE = "clarify"
# A code-level veto (Step 21a). If the model says out_of_scope but the message contains money words, ASK (clarify) instead of blocking: wrongly blocking a real
# finance question is worse than a wasted call. It costs some catches on off-topic messages that happen to use a money word ("bank the fire"); those still stop at clarify.
FINANCE_WORDS = re.compile(r"\b(invest\w*|stocks?|shares?|shareholders?|bonds?|etfs?|funds?|dividends?|portfolios?|retire\w*|401\(?k\)?|iras?|roth|tax\w*|loans?|debts?|mortgages?|"
                           r"interest|inflation|savings?|saving|saved|budget\w*|crypto\w*|bitcoin|finance\w*|financial|credit|brokers?\w*|equity|assets?|income|salary|wealth|"
                           r"money|futures|commodit\w*|earnings|profits?|returns?|net worth|paycheck|refund)\b", re.I)



def response_format_for(labels: tuple[str, ...]) -> dict:
    """A strict schema whose route enum is exactly the labels that prompt version knows, so an old prompt can never answer with a label it was not told about."""
    schema = {"type": "object", "properties": {"route": {"type": "string", "enum": list(labels)}, "confidence": {"type": "string", "enum": list(CONFIDENCE)}},
              "required": ["route", "confidence"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": schema}}


RESPONSE_FORMAT = response_format_for(ROUTES_V2)   # kept for the v1/v2 comparison

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
_V2 = _V2_DEFINITIONS + _V2_EXAMPLES
_OUT_OF_SCOPE = ("\n- out_of_scope: the message is NOT about money, investing, saving, taxes, markets or personal finance "
                 "(cooking, sports, poems, jokes, weather, trivia, coding, translation, health, travel).")
_V3_EXTRA = """

Scope rules:
- out_of_scope is only for messages with NO connection to money or finance. A message about the price or stock of a company, inflation or the cost of living, budgeting, saving, taxes, or investing in something is IN scope even if it mentions food, sports or everyday items.
- When unsure whether a message is about finance, choose a finance route (or clarify), NOT out_of_scope. Blocking a real finance question is worse than letting an off-topic one through.

More examples:
"How much does a haircut cost?" -> out_of_scope, high
"Tell me a joke" -> out_of_scope, high
"Who is the best player in the NBA?" -> out_of_scope, high
"Is inflation raising my rent?" -> finance_qa, high
"How is Starbucks stock doing?" -> market, high
"Should I budget for travel?" -> finance_qa, medium
"What is a bond?" -> finance_qa, high"""
_V3 = _V2.replace("\n\nConfidence:", _OUT_OF_SCOPE + "\n\nConfidence:", 1) + _V3_EXTRA
# v4 (written after the stress set showed a real finance question blocked): explicit rules for finance questions that wear everyday clothes, and "only when SURE"
_V4_EXTRA = """

More scope rules:
- A question about whether something is a good INVESTMENT or business to put money into, whether to pay off debt or invest, how a purchase or habit adds up over time, or how earnings are TAXED is a finance question, whatever the everyday topic (a restaurant, a franchise, a loan, a latte habit).
- Choose out_of_scope only when you are SURE that nothing about money is being asked (recipes, sports scores, poems, weather, trivia, coding, translation, health, travel).
More examples:
"Is buying a self-storage unit a good investment?" -> finance_qa, high
"Should I pay down my student loan or save for a house?" -> finance_qa, high"""
# v5 (written after the router test split and then stress3 showed uploaded-file requests blocked): in this app a message about the user's own file is a portfolio request
_V5_EXTRA = """
- This app lets users upload their holdings. A message about the user's uploaded file, CSV, spreadsheet, statement, holdings or account data is a PORTFOLIO request. If you cannot tell what a file contains, choose portfolio or clarify, NEVER out_of_scope.
More examples:
"Check the spreadsheet I attached" -> portfolio, high
"Go through my account statement" -> portfolio, medium"""
_V5 = _V3 + _V4_EXTRA + _V5_EXTRA
# v6 (Step 21b): a `multi` route for messages that need SEVERAL agents. The planner decides what to run; a false `multi` costs one cheap planner call (it can still
# downgrade to one task), while a missed `multi` loses half the answer, so when two distinct sources are clearly asked for the router leans to multi.
_MULTI = ("\n- multi: the message asks for TWO OR MORE separate things that need DIFFERENT agents (for example a concept explained AND a live price, news AND the user's "
          "portfolio, tax rules AND how a stock is doing).")
_V6_EXTRA = """

Multi rules:
- multi is for requests with separate parts that different agents must answer. A single question that one agent can answer, even if it has several sentences or parts, is NOT multi.
- When two clearly different sources are asked for (data plus an explanation, news plus prices, the user's holdings plus the market), choose multi.
More examples:
"Explain what a bond is and tell me how TLT did this month" -> multi, high
"Is my portfolio exposed to the news about the banking sector?" -> multi, medium
"What are the tax rules for IRAs and how much can I contribute?" -> tax, high
"How is Intel doing and what is its beta?" -> market, medium"""
PROMPTS = {"v1": _DEFINITIONS, "v2": _V2, "v3": _V3, "v4": _V3 + _V4_EXTRA, "v5": _V5,
           "v6": _V5.replace("\n\nConfidence:", _MULTI + "\n\nConfidence:", 1) + _V6_EXTRA}
assert "out_of_scope" in PROMPTS["v3"] and "out_of_scope" not in PROMPTS["v2"]
LABELS = {"v1": ROUTES_V2, "v2": ROUTES_V2, "v3": ROUTES, "v4": ROUTES_V5, "v5": ROUTES_V5, "v6": ROUTES}   # which labels each prompt version may produce
DEFAULT_PROMPT_VERSION = "v6"   # v6 adds multi (Step 21b). Before that: v3 blocked a real finance question and asked too often (stress sets) -> v4; v4 blocked uploaded-file requests -> v5. The finance-word veto stays OFF (it cost catches and added nothing with v4)


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
    vetoed: bool = False       # True = the model said out_of_scope but the message has money words, so it was sent to clarify instead of blocked

    def as_dict(self) -> dict:
        return asdict(self)


class Router:
    def __init__(self, llm: LLMClient, model_key: str = "router_model", prompt_version: str = DEFAULT_PROMPT_VERSION, low_confidence_to_clarify: bool = True,
                 finance_veto: bool = False) -> None:
        if prompt_version not in PROMPTS:
            raise ValueError(f"prompt_version must be one of {sorted(PROMPTS)}")
        self.llm, self.model_key, self.prompt_version = llm, model_key, prompt_version
        self.labels, self.response_format = LABELS[prompt_version], response_format_for(LABELS[prompt_version])
        self.low_confidence_to_clarify, self.finance_veto = low_confidence_to_clarify, finance_veto

    def classify(self, message: str) -> RouteDecision:
        result = self.llm.chat(f"MESSAGE: {message}", system=PROMPTS[self.prompt_version], model_key=self.model_key, temperature=0,
                               max_output_tokens=30, response_format=self.response_format)
        try:
            data = json.loads(result.text)
            raw_route, confidence, fell_back = data["route"], data["confidence"], False
            if raw_route not in self.labels or confidence not in CONFIDENCE:
                raise ValueError(result.text)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.warning("unusable router reply %r; falling back to %s", result.text[:80], DEFAULT_ROUTE)
            raw_route, confidence, fell_back = DEFAULT_ROUTE, "low", True
        route = raw_route
        overridden = self.low_confidence_to_clarify and confidence == "low" and raw_route != "clarify"
        if overridden:
            route = "clarify"
        vetoed = self.finance_veto and route == "out_of_scope" and bool(FINANCE_WORDS.search(message))
        if vetoed:
            route = "clarify"
        return RouteDecision(route, confidence, raw_route, overridden, result.model, result.input_tokens, result.output_tokens,
                             result.cost_usd, result.latency_s, fell_back, vetoed)
